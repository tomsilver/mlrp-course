"""A solver that uses a jaxopt optimizer for unconstrained trajopt problems."""

from dataclasses import dataclass
from typing import Any, Dict, List, Type

import jax.numpy as jnp
from jaxopt import GradientDescent
from jaxopt._src.base import Solver as JaxOptSolver
from numpy.typing import NDArray

from mlrp_course.structs import Hyperparameters
from mlrp_course.trajopt.algorithms.trajopt_solver import UnconstrainedTrajOptSolver
from mlrp_course.trajopt.trajopt_problem import (
    TrajOptAction,
    TrajOptState,
    TrajOptTraj,
)
from mlrp_course.trajopt.utils import (
    sample_standard_normal_spline,
    spline_to_trajopt_trajectory,
)
from mlrp_course.utils import Trajectory, point_sequence_to_trajectory


@dataclass(frozen=True)
class JaxOptTrajOptSolverHyperparameters(Hyperparameters):
    """Hyperparameters for JaxOptTrajOptSolver."""

    num_control_points: int = 10


class JaxOptTrajOptSolver(UnconstrainedTrajOptSolver):
    """A jaxopt-based solver for unconstrained trajopt problems."""

    def __init__(
        self,
        seed: int,
        solver_cls: Type[JaxOptSolver],
        solver_kwargs: Dict[str, Any],
        config: JaxOptTrajOptSolverHyperparameters | None = None,
        warm_start: bool = True,
    ) -> None:
        self._solver_cls = solver_cls or GradientDescent
        self._solver_kwargs = solver_kwargs or {}
        self._config = config or JaxOptTrajOptSolverHyperparameters()
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
        # Create optimization objective.
        assert self._problem is not None
        _get_traj_cost = self._problem.get_traj_cost

        def _objective(params: NDArray[jnp.float32]) -> float:
            spline = point_sequence_to_trajectory(params, dt=dt)
            traj = self._solution_to_trajectory(spline, initial_state, horizon)
            return _get_traj_cost(traj)

        # Create initialization.
        dt = horizon / (self._config.num_control_points - 1)
        init_params = jnp.array(
            [init_traj(t) for t in self._get_control_times(horizon)]
        )

        # Create solver.
        solver = self._solver_cls(fun=_objective, **self._solver_kwargs)

        # Solve.
        params, _ = solver.run(init_params)

        return point_sequence_to_trajectory(params, dt=dt)

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
        assert self._problem.action_space.shape == (1,)
        return sample_standard_normal_spline(
            self._rng, self._config.num_control_points, horizon
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


class GradientDescentTrajOptSolver(JaxOptTrajOptSolver):
    """Gradient descent trajopt solver."""

    def __init__(
        self,
        seed: int,
        maxiter: int = 500,
        jit: bool = True,
        config: JaxOptTrajOptSolverHyperparameters | None = None,
        warm_start: bool = True,
    ) -> None:
        solver_cls = GradientDescent
        solver_kwargs = {"maxiter": maxiter, "jit": jit}
        super().__init__(seed, solver_cls, solver_kwargs, config, warm_start)
