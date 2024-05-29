"""Sparse sampling."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState
from mlrp_course.mdp.utils import DiscreteMDPAgent
from mlrp_course.structs import Hyperparameters


@dataclass(frozen=True)
class SparseSamplingHyperparameters(Hyperparameters):
    """Hyperparameters for sparse sampling."""

    search_horizon: int = 10
    num_samples: int = 5


def sparse_sampling(
    initial_state: DiscreteState,
    mdp: DiscreteMDP,
    rng: np.random.Generator,
    config: SparseSamplingHyperparameters,
) -> DiscreteAction:
    """Returns a single action to take."""
    # Note: no iteration over state space.
    A = mdp.action_space
    R = mdp.get_reward
    gamma = mdp.temporal_discount_factor

    @lru_cache(maxsize=None)
    def V(s, h):
        """Shorthand for the value function."""
        if h == config.search_horizon:
            return 0
        return max(Q(s, a, h) for a in A)

    @lru_cache(maxsize=None)
    def Q(s, a, h):
        """Shorthand for the action-value function."""
        qsa = 0.0
        for _ in range(config.num_samples):
            ns = mdp.sample_next_state(s, a, rng)
            qsa += 1 / config.num_samples * (R(s, a, ns) + gamma * V(ns, h + 1))
        return qsa

    return max(A, key=lambda a: Q(initial_state, a, 0))


class SparseSamplingAgent(DiscreteMDPAgent):
    """An agent that runs RTDP at every timestep."""

    def __init__(
        self,
        mdp: DiscreteMDP,
        seed: int,
        sparse_sampling_hyperparameters: SparseSamplingHyperparameters | None = None,
    ) -> None:
        self._sparse_sampling_hyperparameters = (
            sparse_sampling_hyperparameters or SparseSamplingHyperparameters()
        )
        super().__init__(mdp, seed)

    def _get_action(self) -> DiscreteAction:
        assert self._last_observation is not None
        return sparse_sampling(
            self._last_observation,
            self._mdp,
            self._rng,
            self._sparse_sampling_hyperparameters,
        )
