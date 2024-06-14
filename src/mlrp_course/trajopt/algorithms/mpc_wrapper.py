"""Model-predictive controller."""

from mlrp_course.trajopt.algorithms.trajopt_solver import UnconstrainedTrajOptSolver
from mlrp_course.trajopt.trajopt_problem import (
    TrajOptAction,
    TrajOptState,
    UnconstrainedTrajOptProblem,
)


class MPCWrapper:
    """Re-run a given trajectory optimization solver at every timestep."""

    def __init__(
        self, problem: UnconstrainedTrajOptProblem, solver: UnconstrainedTrajOptSolver
    ) -> None:
        self._problem = problem
        self._solver = solver
        self._timestep = 0

    def reset(self) -> None:
        """Reset the timestep to 0."""
        self._timestep = 0
        self._solver.reset()

    def step(self, state: TrajOptState) -> TrajOptAction:
        """Get the most recent state and return an action."""
        # Run the solver.
        traj = self._solver.solve(
            problem=self._problem,
            initial_state=state,
            horizon=(self._problem.horizon - self._timestep),
        )
        # Take the first action.
        action = traj.actions[0]
        # Advance time.
        self._timestep += 1
        return action
