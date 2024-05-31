"""Run online planning with MCTS in the belief MDP for a POMDP."""

import numpy as np

from mlrp_course.mdp.algorithms.mcts import (
    MCTSHyperparameters,
    mcts,
)
from mlrp_course.mdp.discrete_mdp import DiscreteAction
from mlrp_course.pomdp.discrete_pomdp import BeliefState, DiscretePOMDP
from mlrp_course.pomdp.utils import BeliefMDP, DiscretePOMDPAgent


def pomdp_mcts(
    initial_belief_state: BeliefState,
    pomdp: DiscretePOMDP,
    rng: np.random.Generator,
    config: MCTSHyperparameters,
) -> DiscreteAction:
    """Returns a single action to take."""
    mdp = BeliefMDP(pomdp)
    return mcts(initial_belief_state, mdp, rng, config)


class POMDPMCTSAgent(DiscretePOMDPAgent):
    """An agent that runs POMDP MCTS at every timestep."""

    def __init__(
        self,
        pomdp: DiscretePOMDP,
        seed: int,
        mcts_hyperparameters: MCTSHyperparameters | None = None,
    ) -> None:
        self._mcts_hyperparameters = mcts_hyperparameters or MCTSHyperparameters()
        super().__init__(pomdp, seed)

    def _get_action(self) -> DiscreteAction:
        return pomdp_mcts(
            self._belief_state,
            self._pomdp,
            self._rng,
            self._mcts_hyperparameters,
        )
