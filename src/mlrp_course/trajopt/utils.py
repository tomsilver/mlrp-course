"""Utils for trajectory optimization."""

from typing import List

import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray

from mlrp_course.trajopt.trajopt_problem import (
    TrajOptAction,
    TrajOptState,
    TrajOptTraj,
    UnconstrainedTrajOptProblem,
)
from mlrp_course.utils import Trajectory, point_sequence_to_trajectory


def spline_to_trajopt_trajectory(
    problem: UnconstrainedTrajOptProblem,
    solution: Trajectory[TrajOptAction],
    initial_state: TrajOptState,
    horizon: int,
) -> TrajOptTraj:
    """Roll out a spline to create a trajectory."""
    assert np.isclose(solution.duration, horizon)
    state_list = [initial_state]
    state = initial_state
    action_list: List[TrajOptAction] = []
    for t in range(horizon):
        action = solution(t)
        action_list.append(action)
        state = problem.get_next_state(state, action)
        state_list.append(state)
    state_arr = np.array(state_list, dtype=np.float32)
    action_arr = np.array(action_list, dtype=np.float32)
    return TrajOptTraj(state_arr, action_arr)


def sample_spline_from_box_space(
    box: Box, num_points: int, horizon: int
) -> Trajectory[NDArray[np.float32]]:
    """Sample a spline by sampling points and interpolating."""
    points = [box.sample() for _ in range(num_points)]
    dt = horizon / (len(points) - 1)
    return point_sequence_to_trajectory(points, dt=dt)
