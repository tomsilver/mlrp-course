"""Q Learning."""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from mlrp_course.agents import DiscreteMDPAgent
from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState
from mlrp_course.structs import AlgorithmConfig


@dataclass(frozen=True)
class QLearningConfig(AlgorithmConfig):
    """Hyperparameters for Q Learning."""

    explore_strategy: str = "epsilon-greedy"
    epsilon: float = 0.1
    learning_rate: float = 0.1


class QLearningAgent(DiscreteMDPAgent):
    """An agent that learns with Q-learning."""

    def __init__(self, planner_config: QLearningConfig, *args, **kwargs) -> None:
        self._planner_config = planner_config
        super().__init__(*args, **kwargs)

    
