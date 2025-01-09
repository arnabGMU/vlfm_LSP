# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from torch import Tensor
import sys
import networkx as nx

from vlfm.mapping.frontier_map import FrontierMap
from vlfm.mapping.value_map import ValueMap
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.policy.lsp import get_lowest_cost_ordering, Frontier_LSP
from vlfm.utils.geometry_utils import closest_point_within_threshold
from vlfm.utils.img_utils import crop_white_border, resize_images
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.vlm.detections import ObjectDetections
from matplotlib import pyplot as plt

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

PROMPT_SEPARATOR = "|"
EPISODE = 0

class BaseITMPolicy(BaseObjectNavPolicy):
    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    _circle_marker_radius: int = 5
    _last_value: float = float("-inf")
    _last_frontier: np.ndarray = np.zeros(2)

    @staticmethod
    def _vis_reduce_fn(i: np.ndarray) -> np.ndarray:
        return np.max(i, axis=-1)

    def __init__(
        self,
        text_prompt: str,
        use_max_confidence: bool = True,
        sync_explored_areas: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._itm = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        self._text_prompt = text_prompt
        self._value_map: ValueMap = ValueMap(
            value_channels=len(text_prompt.split(PROMPT_SEPARATOR)),
            use_max_confidence=use_max_confidence,
            obstacle_map=self._obstacle_map if sync_explored_areas else None,
        )
        self._acyclic_enforcer = AcyclicEnforcer()
        self.LSP = True
        self.navigable_map_LSP = None
        self.navigable_nodes_LSP = None
        self.navigable_graph_LSP = None

        self.episode = 2

    def _reset(self) -> None:
        super()._reset()
        self._value_map.reset()
        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_value = float("-inf")
        self._last_frontier = np.zeros(2)

        self.navigable_map_LSP = None
        self.navigable_nodes_LSP = None
        self.navigable_graph_LSP = None

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        frontiers = self._observations_cache["frontier_sensor"]
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            print("No frontiers found during exploration, stopping.")
            return self._stop_action
        best_frontier, best_value = self._get_best_frontier(observations, frontiers, LSP=self.LSP)
        os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}%"
        print(f"Best value: {best_value*100:.2f}%")

        pointnav_action = self._pointnav(best_frontier, stop=False)

        return pointnav_action


    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
        LSP=False
    ) -> Tuple[np.ndarray, float]:
        """Returns the best frontier and its value based on self._value_map.

        Args:
            observations (Union[Dict[str, Tensor], "TensorDict"]): The observations from
                the environment.
            frontiers (np.ndarray): The frontiers to choose from, array of 2D points.

        Returns:
            Tuple[np.ndarray, float]: The best frontier and its value.
        """
        # The points and values will be sorted in descending order
        sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)
        robot_xy = self._observations_cache["robot_xy"]
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2])

        os.environ["DEBUG_INFO"] = ""
        # If there is a last point pursued, then we consider sticking to pursuing it
        # if it is still in the list of frontiers and its current value is not much
        # worse than self._last_value.
        if not np.array_equal(self._last_frontier, np.zeros(2)):
            curr_index = None

            for idx, p in enumerate(sorted_pts):
                if np.array_equal(p, self._last_frontier):
                    # Last point is still in the list of frontiers
                    curr_index = idx
                    break

            if curr_index is None:
                closest_index = closest_point_within_threshold(sorted_pts, self._last_frontier, threshold=0.5)

                if closest_index != -1:
                    # There is a point close to the last point pursued
                    curr_index = closest_index

            if curr_index is not None:
                curr_value = sorted_values[curr_index]
                if curr_value + 0.01 > self._last_value:
                    # The last point pursued is still in the list of frontiers and its
                    # value is not much worse than self._last_value
                    print("Sticking to last point.")
                    os.environ["DEBUG_INFO"] += "Sticking to last point. "
                    best_frontier_idx = curr_index

        

        # If there is no last point pursued, then just take the best point, given that
        # it is not cyclic.
        if best_frontier_idx is None:
            if LSP:
                cost, ordering = self._get_best_frontier_lsp(sorted_pts, sorted_values, robot_xy)
                best_frontier_LSP = ordering[0]
                
                for frontier_LSP in ordering:
                    frontier = self._value_map._px_to_xy(frontier_LSP.centroid.reshape(1,2))[0]
                    frontier_value = frontier_LSP.prob_feasible
                    
                    cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier, top_two_values)
                    if cyclic:
                        print("Suppressed cyclic frontier. LSP")
                        continue
                    best_frontier = frontier
                    best_value = frontier_value
                    self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
                    self._last_value = best_value
                    self._last_frontier = best_frontier
                    os.environ["DEBUG_INFO"] += f" Best value: {best_value*100:.2f}%"

                    return best_frontier, best_value
            else:
                for idx, frontier in enumerate(sorted_pts):
                    cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier, top_two_values)
                    if cyclic:
                        print("Suppressed cyclic frontier.")
                        continue
                    best_frontier_idx = idx
                    break

        if best_frontier_idx is None:
            print("All frontiers are cyclic. Just choosing the closest one.")
            os.environ["DEBUG_INFO"] += "All frontiers are cyclic. "
            best_frontier_idx = max(
                range(len(frontiers)),
                key=lambda i: np.linalg.norm(frontiers[i] - robot_xy),
            )

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]
        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier
        os.environ["DEBUG_INFO"] += f" Best value: {best_value*100:.2f}%"

        return best_frontier, best_value
    
    def _get_best_frontier_lsp(self, frontiers, values, robot_xy):
        frontiers_px = self._value_map._xy_to_px(frontiers)
        print("frontiers", frontiers)
        print("frontiers_px", frontiers_px)
        robot_px = tuple(self._value_map._xy_to_px(robot_xy.reshape(1,2))[0])
        print("robot_xy", robot_xy)
        print("robot_px", robot_px)
        if self._last_frontier is not None:
            last_frontier_px = self._value_map._xy_to_px(self._last_frontier.reshape(1,2))[0]

        subgoals = []
        for frontier, prob in zip(frontiers_px, values):
            frontier_lsp = Frontier_LSP(frontier)
            frontier_lsp.set_props(prob)

            if self._last_frontier is not None and tuple(last_frontier_px) == tuple(frontier):
                frontier_lsp.is_from_last_chosen = True 
            subgoals.append(frontier_lsp)
        
        # Calculate robot-frontier and frontier-frontier distances
        distances = self._calculate_distances_LSP(subgoals, robot_px)
        cost, ordering = get_lowest_cost_ordering(subgoals, distances)
        print("lsp cost", cost)
        print("ordering", ordering)
        return cost, ordering

    def _add_edges_to_graph(self, graph, grid, nodes):
        grid_max_x, grid_max_y = grid.shape
        for node in nodes:
            i = node[0]
            j = node[1]
            if i+1 < grid_max_x and grid[i + 1][j] == 1:
                graph.add_edge((i, j), (i + 1, j), weight=1)
            if j+1 < grid_max_y and grid[i][j + 1] == 1:
                graph.add_edge((i, j), (i, j + 1), weight=1)
            if i+1 < grid_max_x and j+1 < grid_max_y and grid[i+1][j+1] == 1:
                graph.add_edge((i, j), (i+1, j+1), weight=1.41)
            if 0 <= i-1 and 0 <= j-1 and grid[i-1][j-1] == 1:
                graph.add_edge((i,j), (i-1,j-1), weight=1.41)

    def _update_navigable_graph_LSP(self):
        if self.navigable_graph_LSP is None:
            self.navigable_map_LSP = self._obstacle_map._navigable_map.astype(np.uint8).copy()
            self.navigable_nodes_LSP = list(map(tuple, np.transpose(np.where(self.navigable_map_LSP==1))))
            cv2.imshow("navigable map", self._obstacle_map._navigable_map)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("nodes", self.navigable_map_LSP)
            print(len(self.navigable_nodes_LSP))
            
            self.navigable_graph_LSP = nx.Graph()
            self.navigable_graph_LSP.add_nodes_from(self.navigable_nodes_LSP)
            self._add_edges_to_graph(self.navigable_graph_LSP, self.navigable_map_LSP, self.navigable_nodes_LSP)
        
        else:
            current_nav_map = self._obstacle_map._navigable_map.astype(np.uint8).copy()
            new_nav_nodes = list(map(tuple, np.transpose(np.where(np.logical_and(current_nav_map==1, current_nav_map != self.navigable_map_LSP)==1))))
            plt.figure()
            plt.imshow(current_nav_map);plt.show()
            plt.imshow(np.logical_and(current_nav_map==1, current_nav_map != self.navigable_map_LSP)==1);plt.show()
            #print("new nodes", new_nav_nodes)

            self.navigable_graph_LSP.add_nodes_from(new_nav_nodes)
            self._add_edges_to_graph(self.navigable_graph_LSP, current_nav_map, new_nav_nodes)

            self.navigable_map_LSP = current_nav_map

    def _calculate_distances_LSP(self, frontiers_LSP, robot_px):
        self._update_navigable_graph_LSP()
        distances = {'goal': {}, 'robot': {}, 'frontier': {}}
        for i in range(len(frontiers_LSP)):
            frontier = frontiers_LSP[i]
            distances['goal'][frontier] = 0
            distances['robot'][frontier] = nx.shortest_path_length(self.navigable_graph_LSP, \
                                                                   robot_px, \
                                                                    tuple(frontier.centroid)) / self._value_map.pixels_per_meter
            for j in range(i+1, len(frontiers_LSP)):
                frontier2 = frontiers_LSP[j]
                distances['frontier'][frozenset([frontier, frontier2])] = nx.shortest_path_length(self.navigable_graph_LSP, \
                                                                                                tuple(frontier.centroid), \
                                                                                                    tuple(frontier2.centroid)) / self._value_map.pixels_per_meter
        return distances
    
    def save_figure(self):
        policy_info = {}
        markers = []

        # Draw frontiers on to the cost map
        frontiers = self._observations_cache["frontier_sensor"]
        if not(np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0):

            for i, frontier in enumerate(frontiers):
                marker_kwargs = {
                    "radius": self._circle_marker_radius,
                    "thickness": self._circle_marker_thickness,
                    "color": self._frontier_color,
                }
                markers.append((frontier[:2], marker_kwargs))

            if not np.array_equal(self._last_goal, np.zeros(2)):
                # Draw the pointnav goal on to the cost map
                if any(np.array_equal(self._last_goal, frontier) for frontier in frontiers):
                    color = self._selected__frontier_color
                else:
                    color = self._target_object_color
                marker_kwargs = {
                    "radius": 7,
                    "thickness": self._circle_marker_thickness,
                    "color": color,
                }
                markers.append((self._last_goal, marker_kwargs))

        policy_info["value_map"] = cv2.cvtColor(
            self._value_map.visualize(markers, reduce_fn=self._vis_reduce_fn),
            cv2.COLOR_BGR2RGB,
        )
        policy_info["value_map"] = crop_white_border(policy_info["value_map"])
        policy_info["obstacle_map"] = cv2.cvtColor(self._obstacle_map.visualize(), cv2.COLOR_BGR2RGB)
        policy_info["obstacle_map"] = crop_white_border(policy_info["obstacle_map"])

        #rgb_canvas = np.zeros(policy_info["obstacle_map"].shape)
        rgb = cv2.cvtColor(self._observations_cache["object_map_rgbd"][0][0], cv2.COLOR_BGR2RGB)
        # rgb_canvas[rgb_canvas.shape[0]//2 - rgb.shape[0] // 2: rgb_canvas.shape[0]//2 + rgb.shape[0] // 2, \
        #            rgb_canvas.shape[1]//2 - rgb.shape[1] // 2:rgb_canvas.shape[1]//2 + rgb.shape[1] // 2, :] = rgb

        resized_images = resize_images([rgb, policy_info['obstacle_map'], policy_info["value_map"]])
        #image_horizontal = np.concatenate((rgb_canvas, policy_info['obstacle_map'], policy_info["value_map"]), axis=1)
        image_horizontal = np.concatenate(resized_images, axis=1)
        if self.LSP:
            path = f"./figures/LSP/{self.episode}"
        else:
            path = f"./figures/{self.episode}"
        if not os.path.exists(path):
            os.mkdir(path) 
        path = f'{path}/{self._num_steps}.png'       
        cv2.imwrite(path, image_horizontal)

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections)

        if not self._visualize:
            return policy_info

        markers = []

        # Draw frontiers on to the cost map
        frontiers = self._observations_cache["frontier_sensor"]
        for frontier in frontiers:
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }
            markers.append((frontier[:2], marker_kwargs))

        if not np.array_equal(self._last_goal, np.zeros(2)):
            # Draw the pointnav goal on to the cost map
            if any(np.array_equal(self._last_goal, frontier) for frontier in frontiers):
                color = self._selected__frontier_color
            else:
                color = self._target_object_color
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": color,
            }
            markers.append((self._last_goal, marker_kwargs))
        policy_info["value_map"] = cv2.cvtColor(
            self._value_map.visualize(markers, reduce_fn=self._vis_reduce_fn),
            cv2.COLOR_BGR2RGB,
        )

        return policy_info

    def _update_value_map(self) -> None:
        all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
        cosines = [
            [
                self._itm.cosine(
                    rgb,
                    p.replace("target_object", self._target_object.replace("|", "/")),
                )
                for p in self._text_prompt.split(PROMPT_SEPARATOR)
            ]
            for rgb in all_rgb
        ]
        for cosine, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
            cosines, self._observations_cache["value_map_rgbd"]
        ):
            self._value_map.update_map(np.array(cosine), depth, tf, min_depth, max_depth, fov)

        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        raise NotImplementedError


class ITMPolicy(BaseITMPolicy):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frontier_map: FrontierMap = FrontierMap()

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        self._pre_step(observations, masks)
        if self._visualize:
            self._update_value_map()
        
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _reset(self) -> None:
        super()._reset()
        self._frontier_map.reset()

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        rgb = self._observations_cache["object_map_rgbd"][0][0]
        text = self._text_prompt.replace("target_object", self._target_object)
        self._frontier_map.update(frontiers, rgb, text)  # type: ignore
        return self._frontier_map.sort_waypoints()


class ITMPolicyV2(BaseITMPolicy):
    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        self._pre_step(observations, masks)
        self._update_value_map()
        self.save_figure()
        #detections = self._get_object_detections(rgb)
        #sys.exit()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5)
        return sorted_frontiers, sorted_values


class ITMPolicyV3(ITMPolicyV2):
    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._exploration_thresh = exploration_thresh

        def visualize_value_map(arr: np.ndarray) -> np.ndarray:
            # Get the values in the first channel
            first_channel = arr[:, :, 0]
            # Get the max values across the two channels
            max_values = np.max(arr, axis=2)
            # Create a boolean mask where the first channel is above the threshold
            mask = first_channel > exploration_thresh
            # Use the mask to select from the first channel or max values
            result = np.where(mask, first_channel, max_values)

            return result

        self._vis_reduce_fn = visualize_value_map  # type: ignore

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5, reduce_fn=self._reduce_values)

        return sorted_frontiers, sorted_values

    def _reduce_values(self, values: List[Tuple[float, float]]) -> List[float]:
        """
        Reduce the values to a single value per frontier

        Args:
            values: A list of tuples of the form (target_value, exploration_value). If
                the highest target_value of all the value tuples is below the threshold,
                then we return the second element (exploration_value) of each tuple.
                Otherwise, we return the first element (target_value) of each tuple.

        Returns:
            A list of values, one per frontier.
        """
        target_values = [v[0] for v in values]
        max_target_value = max(target_values)

        if max_target_value < self._exploration_thresh:
            explore_values = [v[1] for v in values]
            return explore_values
        else:
            return [v[0] for v in values]
