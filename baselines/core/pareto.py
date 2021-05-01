from typing import List
import numpy as np


def pareto(costs: np.ndarray):
    """
    Find the pareto-optimal points
    :param costs: (n_points, m_cost_values) array
    :return: (n_points, 1) indicator if point is on pareto front or not.
    """
    assert type(costs) == np.ndarray
    assert costs.ndim == 2

    # first assume all points are pareto optimal
    is_pareto = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_pareto[i]:
            # determine all points that have a smaller cost
            all_with_lower_costs = np.any(costs < c, axis=1)
            keep_on_front = np.logical_and(all_with_lower_costs, is_pareto)
            is_pareto = keep_on_front
            is_pareto[i] = True  # keep self
    return is_pareto

def pareto_index(costs: np.ndarray, index_list):
    """
    Find the pareto-optimal points
    :param costs: (n_points, m_cost_values) array
    :return: (n_points, 1) indicator if point is on pareto front or not.
    """
    # first assume all points are pareto optimal
    is_pareto = np.ones(costs.shape[0], dtype=bool)

    for i, c in enumerate(costs):

        if is_pareto[i]:
            # determine all points that have a smaller cost
            all_with_lower_costs = np.any(costs < c, axis=1)
            keep_on_front = np.logical_and(all_with_lower_costs, is_pareto)
            is_pareto = keep_on_front
            is_pareto[i] = True  # keep self

    index_return = index_list[is_pareto]

    return is_pareto, index_return

def nDS_index(costs, index_list):
    """
    Implementation of the non-dominated sorting method
    :param costs: (n_points, m_cost_values) array
    :list of indeces
    :return: list of all fronts, sorted indeces
    """

    dominating_list = []
    index_return_list = []
    fronts = []
    while costs.size > 0:
        dominating, index_return = pareto_index(costs, index_list)
        fronts.append(costs[dominating])
        costs = costs[~dominating]
        index_list = index_list[~dominating]
        dominating_list.append(dominating)
        index_return_list.append(index_return)

    return fronts, index_return_list


def crowdingDist(fronts, index_list):
    """
    Implementation of the crowding distance
    :param front: (n_points, m_cost_values) array
    :return: sorted_front and corresponding distance value of each element in the sorted_front
    """
    dist_list = []
    index_return_list = []
    
    for g in range(len(fronts)):
        front = fronts[g]
        index_ = index_list[g]

        sorted_front = np.sort(front.view([('', front.dtype)] * front.shape[1]),
                               axis=0).view(np.float)

        _, sorted_index = (list(t) for t in zip(*sorted(zip([f[0] for f in front], index_))))

        normalized_front = np.copy(sorted_front)

        for column in range(normalized_front.shape[1]):
            ma, mi = np.max(normalized_front[:, column]), np.min(normalized_front[:, column])
            normalized_front[:, column] -= mi
            normalized_front[:, column] /= (ma - mi)

        dists = np.empty((sorted_front.shape[0], ), dtype=np.float)
        dists[0] = np.inf
        dists[-1] = np.inf

        for elem_idx in range(1, dists.shape[0] - 1):
            dist_left = np.linalg.norm(normalized_front[elem_idx] - normalized_front[elem_idx - 1])
            dist_right = np.linalg.norm(normalized_front[elem_idx + 1] - normalized_front[elem_idx])
            dists[elem_idx] = dist_left + dist_right

        dist_list.append((sorted_front, dists))
        _, index_sorted_max = (list(t) for t in zip(*sorted(zip(dists, sorted_index))))
        index_sorted_max.reverse()

        index_return_list.append(index_sorted_max)

    return dist_list, index_return_list



def nDS(costs: np.ndarray):
    """
    Implementation of the non-dominated sorting method
    :param costs: (n_points, m_cost_values) array
    :return: list of all fronts
    """
    assert type(costs) == np.ndarray
    assert costs.ndim == 2

    # Stepwise compute the pareto front without all prior dominating points
    my_costs = costs.copy()
    remain = np.ones(len(my_costs), dtype=np.bool)
    fronts = []
    while np.any(remain):
        front_i = pareto(my_costs)
        fronts.append(my_costs[front_i, :])
        my_costs[front_i, :] = np.inf
        remain = np.logical_and(remain, np.logical_not(front_i))
    return fronts


def computeHV2D(front: np.ndarray, ref: List[float]):
    """
    Compute the Hypervolume for the pareto front  (only implement it for 2D)
    :param front: (n_points, m_cost_values) array for which to compute the volume
    :param ref: coordinates of the reference point
    :returns: Hypervolume of the polygon spanned by all points in the front + the reference point
    """

    front = np.asarray(front)
    assert front.ndim == 2
    assert len(ref) == 2


    # We assume all points already sorted
    list_ = [ref]
    for x in front:
        elem_at = len(list_) -1
        list_.append([list_[elem_at][0], x[1]])  # add intersection points by keeping the x constant
        list_.append(x)
    list_.append([list_[-1][0], list_[0][1]])
    sorted_front = np.array(list_)

    def shoelace(x_y): # taken from https://stackoverflow.com/a/58515054
        x_y = np.array(x_y)
        x_y = x_y.reshape(-1,2)

        x = x_y[:, 0]
        y = x_y[:, 1]

        S1 = np.sum(x*np.roll(y,-1))
        S2 = np.sum(y*np.roll(x,-1))

        area = .5*np.absolute(S1 - S2)

        return area
    return shoelace(sorted_front)
