"""The simple trajopt method described in https://arxiv.org/abs/2212.00541"""

from mlrp_course.structs import Hyperparameters
from dataclasses import dataclass
from mlrp_course.trajopt.algorithms.trajopt_solver import UnconstrainedTrajOptSolver
from mlrp_course.trajopt.trajopt_problem import (
    TrajOptState,
    TrajOptTraj,
    UnconstrainedTrajOptProblem,
)

@dataclass(frozen=True)
class PredictiveSamplingHyperparameters(Hyperparameters):
    """Hyperparameters for predictive sampling."""
    
    num_rollouts: int = 100
    noise_scale: float = 1.0


class PredictiveSamplingSolver(UnconstrainedTrajOptSolver):
    """The simple method described in https://arxiv.org/abs/2212.00541"""

    def __init__(self, config: PredictiveSamplingHyperparameters | None = None, warm_start: bool = True) -> None:
        self._config = config or PredictiveSamplingHyperparameters()
        super().__init__(self, warm_start)
        # TODO did you warm start?

    def _solve(
        self,
        problem: UnconstrainedTrajOptProblem,
        initial_state: TrajOptState,
        horizon: int,
    ) -> TODO:
        import ipdb; ipdb.set_trace()

    def _solution_to_trajectory(self, solution: TODO, problem: UnconstrainedTrajOptProblem,
        initial_state: TrajOptState,
        horizon: int,
) -> TrajOptTraj:
        import ipdb; ipdb.set_trace()

    
