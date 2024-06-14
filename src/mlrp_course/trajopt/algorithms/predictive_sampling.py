"""The simple trajopt method described in https://arxiv.org/abs/2212.00541"""

from dataclasses import dataclass
from typing import List

import numpy as np

from mlrp_course.structs import Hyperparameters
from mlrp_course.trajopt.algorithms.trajopt_solver import UnconstrainedTrajOptSolver
from mlrp_course.trajopt.trajopt_problem import (
    TrajOptAction,
    TrajOptState,
    TrajOptTraj,
)
from mlrp_course.utils import Trajectory, point_sequence_to_trajectory


@dataclass(frozen=True)
class PredictiveSamplingHyperparameters(Hyperparameters):
    """Hyperparameters for predictive sampling."""

    num_rollouts: int = 100
    noise_scale: float = 1.0
    control_interval: int = 10  # 1 + number of steps in between control points


class PredictiveSamplingSolver(UnconstrainedTrajOptSolver):
    """The simple method described in https://arxiv.org/abs/2212.00541"""

    def __init__(
        self,
        seed: int,
        config: PredictiveSamplingHyperparameters | None = None,
        warm_start: bool = True,
    ) -> None:
        self._config = config or PredictiveSamplingHyperparameters()
        super().__init__(seed, warm_start)

    def _solve(
        self,
        initial_state: TrajOptState,
        horizon: int,
    ) -> Trajectory[TrajOptAction]:
        # Warm start by advancing the last solution by one step.
        sample_list: List[Trajectory[TrajOptAction]] = []
        if self._warm_start and self._last_solution is not None:
            assert isinstance(self._last_solution, Trajectory)
            assert np.isclose(self._last_solution.duration, horizon + 1)
            nominal = self._last_solution.get_sub_trajectory(1, horizon + 1)
        else:
            nominal = self._get_initialization(horizon)
        sample_list.append(nominal)
        # Sample new candidates around the nominal trajectory.
        for _ in range(self._config.num_rollouts - len(sample_list)):
            sample = self._sample_from_nominal(nominal)
            sample_list.append(sample)
        # Pick the best one.
        return min(
            sample_list, key=lambda s: self._score_sample(s, initial_state, horizon)
        )

    def _get_initialization(self, horizon: int) -> Trajectory[TrajOptAction]:
        assert self._problem is not None
        num_control_points = int(np.ceil(horizon / self._config.control_interval))
        actions = [
            self._problem.action_space.sample() for _ in range(num_control_points)
        ]
        dt = horizon / len(actions)
        return point_sequence_to_trajectory(actions, dt=dt)

    def _sample_from_nominal(
        self, nominal: Trajectory[TrajOptAction]
    ) -> Trajectory[TrajOptAction]:
        assert self._problem is not None
        # Sample by adding Gaussian noise around the nominal trajectory.
        actions = [
            self._rng.multivariate_normal(mean=nominal(t), cov=self._config.noise_scale)
            for t in np.arange(
                0, nominal.duration + 1, step=self._config.control_interval
            )
        ]
        # Clip to obey action limits.
        low = self._problem.action_space.low
        high = self._problem.action_space.high
        actions = [np.clip(a, low, high) for a in actions]
        # Interpolate the control points to create a final trajectory.
        dt = nominal.duration / len(actions)
        return point_sequence_to_trajectory(actions, dt=dt)

    def _score_sample(
        self,
        sample: Trajectory[TrajOptAction],
        initial_state: TrajOptState,
        horizon: int,
    ) -> float:
        assert self._problem is not None
        traj = self._solution_to_trajectory(sample, initial_state, horizon)
        return self._problem.get_traj_cost(traj)

    def _solution_to_trajectory(
        self,
        solution: Trajectory[TrajOptAction],
        initial_state: TrajOptState,
        horizon: int,
    ) -> TrajOptTraj:
        assert self._problem is not None
        assert np.isclose(solution.duration, horizon)
        state_list = [initial_state]
        state = initial_state
        action_list: List[TrajOptAction] = []
        for t in range(horizon):
            action = solution(t)
            action_list.append(action)
            state = self._problem.get_next_state(state, action)
            state_list.append(state)
        state_arr = np.array(state_list, dtype=np.float32)
        action_arr = np.array(action_list, dtype=np.float32)
        return TrajOptTraj(state_arr, action_arr)
