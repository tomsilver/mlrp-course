"""The Chase MDP described in lecture."""

from collections import defaultdict
from typing import ClassVar, DefaultDict, Dict, Set, Tuple, TypeAlias

import numpy as np
from numpy.typing import NDArray

from mlrp_course.mdp.discrete_mdp import DiscreteMDP
from mlrp_course.structs import Image

# Define the state and action types.
ChaseState: TypeAlias = Tuple[Tuple[int, int], Tuple[int, int]]
ChaseAction: TypeAlias = str


class ChaseMDP(DiscreteMDP[ChaseState, ChaseAction]):
    """The Chase MDP described in lecture."""

    # The map is defined based on the obstacles, which by default are empty.
    _obstacles: ClassVar[NDArray[np.bool_]] = np.zeros((2, 3), dtype=np.bool_)
    _height: ClassVar[int] = _obstacles.shape[0]
    _width: ClassVar[int] = _obstacles.shape[1]
    _goal_reward: ClassVar[float] = 1.0
    _living_reward: ClassVar[float] = 0.0

    @property
    def state_space(self) -> Set[ChaseState]:
        pos = [(r, c) for r in range(self._height) for c in range(self._width)]
        return {(p1, p2) for p1 in pos for p2 in pos}

    @property
    def action_space(self) -> Set[ChaseAction]:
        return {"up", "down", "left", "right"}

    @property
    def temporal_discount_factor(self) -> float:
        return 0.9

    def state_is_terminal(self, state: ChaseState) -> bool:
        agent_pos, goal_pos = state
        return agent_pos == goal_pos

    def _action_to_delta(self, action: ChaseAction) -> Tuple[int, int]:
        """Helper for transition distribution construction."""
        return {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }[action]

    def get_transition_distribution(
        self, state: ChaseState, action: ChaseAction
    ) -> Dict[ChaseState, float]:
        # Discrete distributions, represented with a dict
        # mapping next states to probs.
        next_state_dist: DefaultDict[ChaseState, float] = defaultdict(float)

        agent_pos, goal_pos = state

        # Get next agent state.
        row, col = agent_pos
        dr, dc = self._action_to_delta(action)
        r, c = row + dr, col + dc
        # Stay in place if out of bounds or obstacle
        if not (0 <= r < self._height and 0 <= c < self._width):
            r, c = row, col
        elif self._obstacles[r, c]:
            r, c = row, col
        next_agent_pos = (r, c)

        # Get next bunny state.
        # Stay in same place with probability 0.5.
        next_state_dist[(next_agent_pos, goal_pos)] += 0.5
        # Otherwise move.
        row, col = goal_pos
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row + dr, col + dc
            # Stay in place if out of bounds or obstacle.
            if not (0 <= r < self._height and 0 <= c < self._width):
                r, c = row, col
            elif self._obstacles[r, c]:
                r, c = row, col
            next_goal_pos = (r, c)
            next_state_dist[(next_agent_pos, next_goal_pos)] += 0.5 * 0.25

        return dict(next_state_dist)

    def get_reward(
        self, state: ChaseState, action: ChaseAction, next_state: ChaseState
    ) -> float:
        agent_pos, goal_pos = next_state
        if agent_pos == goal_pos:
            return self._goal_reward
        return self._living_reward

    def render_state(self, state: ChaseState) -> Image:
        image = np.ones((self._height, self._width, 4))
        (agent_r, agent_c), (goal_r, goal_c) = state
        image[agent_r, agent_c] = (0.0, 0.0, 0.9, 1.0)
        image[goal_r, goal_c] = (0.0, 0.9, 0.0, 1.0)
        image[np.argwhere(self._obstacles), :3] = 0
        return (255 * image).astype(np.uint8)
