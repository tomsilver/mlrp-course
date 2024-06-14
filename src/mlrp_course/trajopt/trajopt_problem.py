"""A discrete-time finite-horizon trajectory optimization problem."""

import abc
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mlrp_course.structs import Image

TrajOptState: TypeAlias = NDArray[np.float32]
TrajOptAction: TypeAlias = NDArray[np.float32]


@dataclass
class TrajOptTraj:
    """A trajectory of states and actions."""

    states: ArrayLike[TrajOptState]
    actions: ArrayLike[TrajOptAction]

    def __post_init__(self) -> None:
        assert len(self.states) == len(self.actions) + 1


class UnconstrainedTrajOptProblem(abc.ABC):
    """An unconstrained discrete-time finite-horizon trajopt problem."""

    @abc.abstractmethod
    @property
    def horizon(self) -> int:
        """The time horizon of the problem."""

    @abc.abstractmethod
    @property
    def state_dim(self) -> int:
        """The dimension of the state vector."""

    @abc.abstractmethod
    @property
    def action_dim(self) -> int:
        """The dimension of the action vector."""

    @abc.abstractmethod
    def get_next_state(
        self, state: TrajOptState, action: TrajOptAction
    ) -> TrajOptState:
        """The discrete-time dynamics: get the next state."""

    @abc.abstractmethod
    def get_traj_cost(self, traj: TrajOptTraj) -> float:
        """Get the cost of a full trajectory."""

    @abc.abstractmethod
    def render_state(self, state: TrajOptState) -> Image:
        """Optional rendering function."""
