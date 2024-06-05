"""Classical planning domains and problems."""

import abc
from dataclasses import dataclass
from typing import Callable, Generic, Set, TypeAlias, TypeVar

from mlrp_course.structs import HashableComparable, Image

DiscreteState: TypeAlias = HashableComparable
DiscreteAction: TypeAlias = HashableComparable

_S = TypeVar("_S", bound=DiscreteState)
_A = TypeVar("_A", bound=DiscreteAction)


class ClassicalPlanningDomain(Generic[_S, _A]):
    """A classical planning domain."""

    @property
    @abc.abstractmethod
    def state_space(self) -> Set[_S]:
        """Representation of the state set."""

    @property
    @abc.abstractmethod
    def action_space(self) -> Set[_A]:
        """Representation of the action set."""

    @abc.abstractmethod
    def initiable(self, state: _S, action: _A) -> bool:
        """Determines if the given action can be initiated in the state."""

    @abc.abstractmethod
    def get_cost(self, state: _S, action: _A, next_state: _S) -> float:
        """The (non-negative) cost function."""

    @abc.abstractmethod
    def get_next_state(self, state: _S, action: _A) -> _S:
        """The transition function."""

    @abc.abstractmethod
    def render_state(self, state: _S, goal: Callable[[_S], bool]) -> Image:
        """Optional rendering function for visualizations."""


@dataclass(frozen=True)
class ClassicalPlanningProblem(Generic[_S, _A]):
    """A classical planning problem."""

    domain: ClassicalPlanningDomain[_S, _A]
    initial_state: _S
    goal: Callable[[_S], bool]
