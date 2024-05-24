"""The Chase MDP described in lecture."""

from collections import defaultdict
from functools import lru_cache
from typing import ClassVar, DefaultDict, Dict, Set, Tuple, TypeAlias

import numpy as np
from numpy.typing import NDArray
from skimage.transform import resize  # pylint: disable=no-name-in-module

from mlrp_course.mdp.discrete_mdp import DiscreteMDP
from mlrp_course.structs import Image
from mlrp_course.utils import load_image_asset

# Define the state and action types.
ChaseState: TypeAlias = Tuple[Tuple[int, int], Tuple[int, int]]
ChaseAction: TypeAlias = str


class ChaseMDP(DiscreteMDP[ChaseState, ChaseAction]):
    """The Chase MDP described in lecture."""

    # The map is defined based on the obstacles, which by default are empty.
    _obstacles: ClassVar[NDArray[np.bool_]] = np.zeros((2, 3), dtype=np.bool_)
    _goal_reward: ClassVar[float] = 1.0
    _living_reward: ClassVar[float] = 0.0

    @classmethod
    def get_height(cls) -> int:
        """The height, i.e., number of rows."""
        return cls._obstacles.shape[0]

    @classmethod
    def get_width(cls) -> int:
        """The width, i.e., number of columns."""
        return cls._obstacles.shape[1]

    @property
    def state_space(self) -> Set[ChaseState]:
        pos = [
            (r, c) for r in range(self.get_height()) for c in range(self.get_width())
        ]
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
        if not (0 <= r < self.get_height() and 0 <= c < self.get_width()):
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
            if not (0 <= r < self.get_height() and 0 <= c < self.get_width()):
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

    @lru_cache(maxsize=None)
    def _get_token_image(self, cell_type: str) -> Image:
        if cell_type == "robot":
            return load_image_asset("robot.png")
        if cell_type == "bunny":
            return load_image_asset("bunny.png")
        if cell_type == "obstacle":
            return load_image_asset("obstacle.png")
        raise ValueError(f"No asset for {cell_type} known")

    def render_state(self, state: ChaseState) -> Image:
        tilesize = 64
        height, width = self.get_height(), self.get_width()
        canvas = np.zeros((height * tilesize, width * tilesize, 3))

        for r in range(height):
            for c in range(width):
                if (r, c) == state[0]:
                    cell_type = "robot"
                elif (r, c) == state[1]:
                    cell_type = "bunny"
                elif self._obstacles[(r, c)]:
                    cell_type = "obstacle"
                else:
                    continue
                im = self._get_token_image(cell_type)
                canvas[
                    r * tilesize : (r + 1) * tilesize,
                    c * tilesize : (c + 1) * tilesize,
                ] = resize(im[:, :, :3], (tilesize, tilesize, 3), preserve_range=True)

        return (255 * canvas).astype(np.uint8)


class ChaseWithRoomsMDP(ChaseMDP):
    """A variation with many "rooms" that are disconnected."""

    _obstacles: ClassVar[NDArray[np.bool_]] = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=np.bool_,
    )


class ChaseWithLargeRoomsMDP(ChaseMDP):
    """A variation with many large "rooms" that are disconnected."""

    _obstacles: ClassVar[NDArray[np.bool_]] = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=np.bool_,
    )
