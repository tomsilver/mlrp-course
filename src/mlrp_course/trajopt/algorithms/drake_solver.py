"""A solver that uses drake for unconstrained trajopt problems."""

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray
from pydrake.all import Expression, Formula, MathematicalProgram, SnoptSolver, eq  # pylint: disable=no-name-in-module

from mlrp_course.structs import Hyperparameters
from mlrp_course.trajopt.algorithms.trajopt_solver import UnconstrainedTrajOptSolver
from mlrp_course.trajopt.trajopt_problem import (
    TrajOptAction,
    TrajOptState,
    TrajOptTraj,
    UnconstrainedTrajOptProblem,
)
from mlrp_course.utils import Trajectory


@dataclass
class DrakeTrajOptTraj:
    """A trajectory of states and actions."""

    states: NDArray  # array of drake variables
    actions: NDArray  # array of drake variables

    def __post_init__(self) -> None:
        assert len(self.states) == len(self.actions) + 1


class DrakeProblem(UnconstrainedTrajOptProblem):
    """A drake version of a trajopt problem."""

    def create_drake_transition_constraints(
        self,
        state: TrajOptState,
        action: TrajOptAction,
        next_state: TrajOptState,
    ) -> Iterator[Formula]:
        """Create transition constraints."""
        predicted_next_state = self.get_next_state(state, action)
        yield eq(next_state, predicted_next_state)

    def create_drake_cost(self, traj: DrakeTrajOptTraj) -> Expression:
        """Create cost functions for the whole trajectory."""
        return self.get_traj_cost(traj)


@dataclass(frozen=True)
class DrakeTrajOptSolverHyperparameters(Hyperparameters):
    """Hyperparameters for DrakeTrajOptSolver."""


class DrakeTrajOptSolver(UnconstrainedTrajOptSolver):
    """A drake-based solver for trajopt problems."""

    def __init__(
        self,
        seed: int,
        config: DrakeTrajOptSolverHyperparameters | None = None,
        warm_start: bool = True,
    ) -> None:
        self._config = config or DrakeTrajOptSolverHyperparameters()
        super().__init__(seed, warm_start)

    def _solve(
        self,
        initial_state: TrajOptState,
        horizon: int,
    ) -> Trajectory[TrajOptAction]:
        # Warm start by advancing the last solution by one step.
        if self._warm_start and self._last_solution is not None:
            nominal = TrajOptTraj(
                self._last_solution.states[1:], self._last_solution.actions[1:]
            )
        else:
            nominal = self._get_initialization(initial_state, horizon)
        # Optimize.
        return self._optimize_trajectory(nominal, initial_state, horizon)

    def _optimize_trajectory(
        self,
        init_traj: TrajOptTraj,
        initial_state: TrajOptState,
        horizon: int,
    ) -> TrajOptTraj:
        assert isinstance(self._problem, DrakeProblem)

        # Create drake program.
        program = MathematicalProgram()
        state_shape = initial_state.shape
        assert len(state_shape) == 1
        states = program.NewContinuousVariables(horizon + 1, state_shape[0])
        action_shape = self._problem.action_space.shape
        assert len(action_shape) == 1
        actions = program.NewContinuousVariables(horizon, action_shape[0])
        drake_traj = DrakeTrajOptTraj(states, actions)

        # Create constraints.
        # Add initial state constraint.
        initial_state_constraint = eq(states[0], initial_state)
        program.AddConstraint(initial_state_constraint)
        # Add dynamic constraints.
        for s_t, a_t, s_t1 in zip(states[:-1], actions, states[1:], strict=True):
            for c in self._problem.create_drake_transition_constraints(s_t, a_t, s_t1):
                program.AddConstraint(c)

        # Create cost.
        cost = self._problem.create_drake_cost(drake_traj)
        program.AddCost(cost)

        # Set initialization.
        program.SetInitialGuess(states, init_traj.states)
        program.SetInitialGuess(actions, init_traj.actions)

        # Solve.
        solver = SnoptSolver()
        result = solver.Solve(program)

        assert result.is_success()  # TODO
        result_states = result.GetSolution(states)
        result_actions = result.GetSolution(actions)

        if len(result_actions.shape) == 1:
            result_actions = result_actions.reshape((-1, 1))

        return TrajOptTraj(
            result_states.astype(np.float32), result_actions.astype(np.float32)
        )

    def _get_initialization(
        self, initial_state: TrajOptState, horizon: int
    ) -> TrajOptTraj:
        assert self._problem is not None
        assert self._problem.action_space.shape == (1,)
        actions = self._rng.standard_normal(size=(horizon, 1))
        states = [initial_state]
        for action in actions:
            states.append(self._problem.get_next_state(states[-1], action))
        return TrajOptTraj(
            np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32)
        )

    def _solution_to_trajectory(
        self,
        solution: TrajOptTraj,
        initial_state: TrajOptState,
        horizon: int,
    ) -> TrajOptTraj:
        return solution
