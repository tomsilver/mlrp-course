"""A gradient-descent solver for unconstrained trajopt problems."""

from dataclasses import dataclass
from typing import List, Tuple

import jax
import jax.numpy as jnp

from mlrp_course.structs import Hyperparameters
from mlrp_course.trajopt.algorithms.trajopt_solver import UnconstrainedTrajOptSolver
from mlrp_course.trajopt.trajopt_problem import (
    TrajOptAction,
    TrajOptState,
    TrajOptTraj,
)
from mlrp_course.trajopt.utils import (
    sample_spline_from_box_space,
    spline_to_trajopt_trajectory,
)
from mlrp_course.utils import Trajectory, point_sequence_to_trajectory


@dataclass(frozen=True)
class GradientDescentHyperparameters(Hyperparameters):
    """Hyperparameters for gradient descent."""

    num_control_points: int = 10
    num_descent_steps: int = 1
    learning_rates: Tuple[float] = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


class GradientDescentSolver(UnconstrainedTrajOptSolver):
    """A gradient-descent solver for unconstrained trajopt problems."""

    def __init__(
        self,
        seed: int,
        config: GradientDescentHyperparameters | None = None,
        warm_start: bool = True,
    ) -> None:
        self._config = config or GradientDescentHyperparameters()
        super().__init__(seed, warm_start)

    def _solve(
        self,
        initial_state: TrajOptState,
        horizon: int,
    ) -> Trajectory[TrajOptAction]:
        # Warm start by advancing the last solution by one step.
        if self._warm_start and self._last_solution is not None:
            assert isinstance(self._last_solution, Trajectory)
            nominal = self._last_solution.get_sub_trajectory(1, horizon + 1)
        else:
            nominal = self._get_initialization(horizon)
        # Optimize.
        return self._optimize_trajectory(nominal, initial_state, horizon)

    def _optimize_trajectory(
        self,
        init_traj: Trajectory[TrajOptAction],
        initial_state: TrajOptState,
        horizon: int,
    ) -> Trajectory[TrajOptAction]:
        assert self._problem is not None

        # Extract optimization parameter initialization.
        dt = horizon / (self._config.num_control_points - 1)
        init_params = jnp.array(
            [init_traj(t) for t in self._get_control_times(horizon)]
        )

        def _objective(params):
            spline = point_sequence_to_trajectory(params, dt=dt)
            traj = self._solution_to_trajectory(spline, initial_state, horizon)
            return self._problem.get_traj_cost(traj)

        grad_objective = jax.grad(_objective)

        best_params = init_params
        best_loss = _objective(init_params)
        for learning_rate in self._config.learning_rates:
            params = jnp.copy(init_params)
            for _ in range(self._config.num_descent_steps):
                gradients = grad_objective(params)
                params = params - learning_rate * gradients
                loss = _objective(params)
                if loss < best_loss:
                    best_params = jnp.copy(params)
                    best_loss = loss

        return point_sequence_to_trajectory(best_params, dt=dt)

    def _get_control_times(self, horizon: float) -> List[float]:
        control_times = jnp.linspace(
            0,
            horizon,
            num=self._config.num_control_points,
            endpoint=True,
        )
        return list(control_times)

    def _get_initialization(self, horizon: int) -> Trajectory[TrajOptAction]:
        assert self._problem is not None
        return sample_spline_from_box_space(
            self._problem.action_space, self._config.num_control_points, horizon
        )

    def _solution_to_trajectory(
        self,
        solution: Trajectory[TrajOptAction],
        initial_state: TrajOptState,
        horizon: int,
    ) -> TrajOptTraj:
        assert self._problem is not None
        return spline_to_trajopt_trajectory(
            self._problem, solution, initial_state, horizon
        )
