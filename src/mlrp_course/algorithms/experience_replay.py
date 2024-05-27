"""Experience replay."""

import abc
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Tuple, TypeAlias

from mlrp_course.agents import Agent
from mlrp_course.structs import AlgorithmConfig


@dataclass(frozen=True)
class ExperienceReplayConfig(AlgorithmConfig):
    """Hyperparameters for experience replay."""

    num_replays_per_update: int = 10
    buffer_length: int = 1000


_Obs: TypeAlias = Any
_Action: TypeAlias = Any


class ExperienceReplayAgent(Agent[_Obs, _Action], abc.ABC):
    """A mix-in that implements experience replay."""

    def __init__(self, config: ExperienceReplayConfig, seed: int) -> None:
        self._experience_replay_config = config
        self._buffer: Deque[Tuple[_Obs, _Action, _Obs, float, bool]] = deque(
            [], maxlen=config.buffer_length
        )
        Agent.__init__(self, seed)

    def _learn_from_transition(
        self, obs: _Obs, act: _Action, next_obs: _Obs, reward: float, done: bool
    ) -> None:
        self._buffer.append((obs, act, next_obs, reward, done))
        for _ in range(self._experience_replay_config.num_replays_per_update):
            idx = self._rng.choice(len(self._buffer))
            o, a, no, r, d = self._buffer[idx]
            super()._learn_from_transition(o, a, no, r, d)
        return super()._learn_from_transition(obs, act, next_obs, reward, done)
