"""Classical planning problems."""

import abc
from typing import Generic, Iterator, Set, Tuple, TypeAlias, TypeVar

from mlrp_course.structs import HashableComparable, Image

DiscreteState: TypeAlias = HashableComparable
DiscreteAction: TypeAlias = HashableComparable

_S = TypeVar("_S", bound=DiscreteState)
_A = TypeVar("_A", bound=DiscreteAction)


class ClassicalPlanningProblem(Generic[_S, _A]):
    """A classical planning problem."""

    @property
    @abc.abstractmethod
    def state_space(self) -> Set[_S]:
        """Representation of the state set."""

    @property
    @abc.abstractmethod
    def action_space(self) -> Set[_A]:
        """Representation of the action set."""

    @property
    @abc.abstractmethod
    def initial_state(self) -> _S:
        """The initial state."""

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
    def check_goal(self, state: _S) -> bool:
        """The goal function."""

    @abc.abstractmethod
    def render_state(self, state: _S) -> Image:
        """Optional rendering function for visualizations."""

    def get_successors(self, state: _S) -> Iterator[Tuple[_A, _S, float]]:
        """Convenience method that subclasses might override for efficiency."""
        actions = {a for a in self.action_space if self.initiable(state, a)}
        ordered_actions = sorted(actions)  # sort for determinism
        for action in ordered_actions:
            next_state = self.get_next_state(state, action)
            cost = self.get_cost(state, action, next_state)
            yield action, next_state, cost
