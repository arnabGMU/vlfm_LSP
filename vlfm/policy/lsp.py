import numpy as np
import itertools
import lsp_accel

class Frontier_LSP(object):
    def __init__(self, centroid):
        """Initialized with a 2xN numpy array of points (the grid cell
        coordinates of all points on frontier boundary)."""
        #inds = np.lexsort((points[0, :], points[1, :]))
        #sorted_points = points[:, inds]
        self._centroid = centroid
        self.props_set = False
        self.is_from_last_chosen = False
        self.is_obstructed = False
        self.prob_feasible = 1.0
        self.delta_success_cost = 1.0
        self.exploration_cost = 1.0
        self.negative_weighting = 0.0
        self.positive_weighting = 0.0

        self.counter = 0
        self.last_observed_pose = None

        # Any duplicate points should be eliminated (would interfere with
        # equality checking).
        # dupes = []
        # for ii in range(1, sorted_points.shape[1]):
        #     if (sorted_points[:, ii - 1] == sorted_points[:, ii]).all():
        #         dupes += [ii]
        # self.points = np.delete(sorted_points, dupes, axis=1)

        # Compute and cache the hash
        self.hash = hash(self._centroid.tobytes())

    def set_props(self,
                  prob_feasible,
                  is_obstructed=False,
                  delta_success_cost=1,
                  exploration_cost=1,
                  positive_weighting=0,
                  negative_weighting=0,
                  counter=0,
                  last_observed_pose=None,
                  did_set=True):
        self.props_set = did_set
        self.just_set = did_set
        self.prob_feasible = prob_feasible
        self.is_obstructed = is_obstructed
        self.delta_success_cost = delta_success_cost
        self.exploration_cost = exploration_cost
        self.positive_weighting = positive_weighting
        self.negative_weighting = negative_weighting
        self.counter = counter
        self.last_observed_pose = last_observed_pose

    @property
    def centroid(self):
        return self.get_centroid()

    def get_centroid(self):
        """Returns the point that is the centroid of the frontier"""
        # centroid = np.mean(self.points, axis=1)
        # return centroid
        return self._centroid

    def get_frontier_point(self):
        # """Returns the point that is on the frontier that is closest to the
        # actual centroid"""
        # center_point = np.mean(self.points, axis=1)
        # norm = np.linalg.norm(self.points - center_point[:, None], axis=0)
        # ind = np.argmin(norm)
        # return self.points[:, ind]
        return self._centroid

    def get_distance_to_point(self, point):
        norm = np.linalg.norm(self.points - point[:, None], axis=0)
        return norm.min()

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return hash(self) == hash(other)

class FState(object):
    """Used to conviently store the 'state' during recursive cost search.
    """
    def __init__(self, new_frontier, distances, old_state=None):
        nf = new_frontier
        p = nf.prob_feasible
        # Success cost
        try:
            sc = nf.delta_success_cost + distances['goal'][nf]
        except KeyError:
            sc = nf.delta_success_cost + distances['goal'][nf.id]
        # Exploration cost
        ec = nf.exploration_cost

        if old_state is not None:
            self.frontier_list = old_state.frontier_list + [nf]
            # Store the old frontier
            of = old_state.frontier_list[-1]
            # Known cost (travel between frontiers)
            try:
                kc = distances['frontier'][frozenset([nf, of])]
            except KeyError:
                kc = distances['frontier'][frozenset([nf.id, of.id])]
            self.cost = old_state.cost + old_state.prob * (kc + p * sc +
                                                           (1 - p) * ec)
            self.prob = old_state.prob * (1 - p)
        else:
            # This is the first frontier, so the robot must accumulate a cost of getting to the frontier
            self.frontier_list = [nf]
            # Known cost (travel to frontier)
            try:
                kc = distances['robot'][nf]
            except KeyError:
                kc = distances['robot'][nf.id]

            # if nf.is_from_last_chosen:
            #     kc -= IS_FROM_LAST_CHOSEN_REWARD
            self.cost = kc + p * sc + (1 - p) * ec
            self.prob = (1 - p)

    def __lt__(self, other):
        return self.cost < other.cost
    
def get_ordering_cost(subgoals, distances):
    """A helper function to compute the expected cost of a particular ordering.
    The function takes an ordered list of subgoals (the order in which the robot
    aims to explore beyond them). Consistent with the subgoal planning API,
    'distances' is a dictionary with three keys: 'robot' (a dict of the
    robot-subgoal distances), 'goal' (a dict of the goal-subgoal distances), and
    'frontier' (a dict of the frontier-frontier distances)."""
    fstate = None
    for s in subgoals:
        fstate = FState(s, distances, fstate)

    return fstate.cost

def get_lowest_cost_ordering(subgoals, distances, do_sort=True):
    """This computes the lowest cost ordering (the policy) the robot will follow
    for navigation under uncertainty. It wraps a branch-and-bound search
    function implemented in C++ in 'lsp_accel'. As is typical of
    branch-and-bound functions, function evaluation is fastest if the high-cost
    plans can be ruled out quickly: i.e., if the first expansion is already of
    relatively low cost, many of the other branches can be pruned. When
    'do_sort' is True, a handful of common-sense heuristics are run to find an
    initial ordering that is of low cost to take advantage of this property. The
    subgoals are sorted by the various heuristics and the ordering that
    minimizes the expected cost is chosen. That ordering is used as an input to
    the search function, which searches it first."""

    if len(subgoals) == 0:
        return None, None

    if do_sort:
        order_heuristics = []
        order_heuristics.append({
            s: ii for ii, s in enumerate(subgoals)
        })
        order_heuristics.append({
            s: 1 - s.prob_feasible for s in subgoals
        })
        order_heuristics.append({
            s: distances['goal'][s] + distances['robot'][s] +
            s.prob_feasible * s.delta_success_cost +
            (1 - s.prob_feasible) * s.exploration_cost
            for s in subgoals
        })
        order_heuristics.append({
            s: distances['goal'][s] + distances['robot'][s]
            for s in subgoals
        })
        order_heuristics.append({
            s: distances['goal'][s] + distances['robot'][s] +
            s.delta_success_cost
            for s in subgoals
        })
        order_heuristics.append({
            s: distances['goal'][s] + distances['robot'][s] +
            s.exploration_cost
            for s in subgoals
        })

        heuristic_ordering_dat = []
        for heuristic in order_heuristics:
            ordered_subgoals = sorted(subgoals, reverse=False, key=lambda s: heuristic[s])
            ordering_cost = get_ordering_cost(ordered_subgoals, distances)
            heuristic_ordering_dat.append((ordering_cost, ordered_subgoals))

        subgoals = min(heuristic_ordering_dat, key=lambda hod: hod[0])[1]

    s_dict = {hash(s): s for s in subgoals}
    rd_cpp = {hash(s): distances['robot'][s] for s in subgoals}
    gd_cpp = {hash(s): distances['goal'][s] for s in subgoals}
    fd_cpp = {(hash(sp[0]), hash(sp[1])): distances['frontier'][frozenset(sp)]
              for sp in itertools.permutations(subgoals, 2)}
    s_cpp = [
        lsp_accel.FrontierData(s.prob_feasible, s.delta_success_cost,
                               s.exploration_cost, hash(s),
                               s.is_from_last_chosen) for s in subgoals
    ]

    cost, ordering = lsp_accel.get_lowest_cost_ordering(
        s_cpp, rd_cpp, gd_cpp, fd_cpp)
    ordering = [s_dict[sid] for sid in ordering]

    return cost, ordering