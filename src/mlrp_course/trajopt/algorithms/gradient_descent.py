"""A gradient-descent solver for unconstrained trajopt problems."""

from dataclasses import dataclass
from typing import List, Tuple

import jax.numpy as jnp
from jax import grad
from numpy.typing import NDArray

from mlrp_course.structs import Hyperparameters
from mlrp_course.trajopt.algorithms.trajopt_solver import UnconstrainedTrajOptSolver
from mlrp_course.trajopt.trajopt_problem import (
    TrajOptAction,
    TrajOptState,
    TrajOptTraj,
)


@dataclass(frozen=True)
class GradientDescentHyperparameters(Hyperparameters):
    """Hyperparameters for gradient descent."""

    num_descent_steps: int = 1
    learning_rates: Tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0)


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
    ) -> List[TrajOptAction]:
        # Warm start by advancing the last solution by one step.
        if self._warm_start and self._last_solution is not None:
            nominal = self._last_solution[1:]
        else:
            nominal = self._get_initialization(horizon)
        # Optimize.
        return self._optimize_trajectory(nominal, initial_state, horizon)

    def _optimize_trajectory(
        self,
        init_traj: List[TrajOptAction],
        initial_state: TrajOptState,
        horizon: int,
    ) -> List[TrajOptAction]:
        assert self._problem is not None
        _get_traj_cost = self._problem.get_traj_cost

        def _objective(params: NDArray[jnp.float32]) -> float:
            traj = self._solution_to_trajectory(params, initial_state, horizon)
            return _get_traj_cost(traj)

        grad_objective = grad(_objective)

        init_params = jnp.array(init_traj)
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

        return list(best_params)

    def _get_initialization(self, horizon: int) -> List[TrajOptAction]:
        assert self._problem is not None
        return [self._problem.action_space.sample() for _ in range(horizon)]

    def _solution_to_trajectory(
        self,
        solution: List[TrajOptAction],
        initial_state: TrajOptState,
        horizon: int,
    ) -> TrajOptTraj:
        assert self._problem is not None
        assert len(solution) == horizon
        state_list = [initial_state]
        state = initial_state
        for action in solution:
            state = self._problem.get_next_state(state, action)
            state_list.append(state)
        state_arr = jnp.array(state_list, dtype=jnp.float32)
        action_arr = jnp.array(solution, dtype=jnp.float32)
        return TrajOptTraj(state_arr, action_arr)
