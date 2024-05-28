"""The Chase MDP described in lecture."""

import itertools
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import (
    Any,
    ClassVar,
    DefaultDict,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeAlias,
)

import numpy as np
from numpy.typing import NDArray
from skimage.transform import resize  # pylint: disable=no-name-in-module

from mlrp_course.mdp.discrete_mdp import DiscreteMDP
from mlrp_course.structs import Image
from mlrp_course.utils import load_image_asset

# Define the state and action types.
ChaseAction: TypeAlias = str
BunnyPosition: TypeAlias = Optional[Tuple[int, int]]


@dataclass(frozen=True)
class ChaseState:
    """The state consists of the robot position and the bunny positions."""

    robot_pos: Tuple[int, int]
    bunny_positions: Tuple[BunnyPosition, ...]  # location or gone

    def __lt__(self, other: Any) -> bool:
        assert isinstance(other, ChaseState)
        return str(self) < str(other)


class ChaseMDP(DiscreteMDP[ChaseState, ChaseAction]):
    """The Chase MDP described in lecture."""

    # The map is defined based on the obstacles, which by default are empty.
    _obstacles: ClassVar[NDArray[np.bool_]] = np.zeros((2, 3), dtype=np.bool_)
    _goal_reward: ClassVar[float] = 1.0
    _capture_reward: ClassVar[float] = 0.0
    _living_reward: ClassVar[float] = 0.0
    _bunnies_move: ClassVar[bool] = True

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
        # Subclasses may have multiple bunnies, but by default there is just one.
        pos = [
            (r, c) for r in range(self.get_height()) for c in range(self.get_width())
        ]
        return {ChaseState(p1, (p2,)) for p1 in pos for p2 in pos + [None]}

    @property
    def action_space(self) -> Set[ChaseAction]:
        return {"up", "down", "left", "right"}

    @property
    def temporal_discount_factor(self) -> float:
        return 0.9

    def state_is_terminal(self, state: ChaseState) -> bool:
        return all(p is None for p in state.bunny_positions)

    def _action_to_delta(self, action: ChaseAction) -> Tuple[int, int]:
        """Helper for transition distribution construction."""
        return {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }[action]

    def _get_next_robot_pos(
        self, state: ChaseState, action: ChaseAction
    ) -> Tuple[int, int]:
        row, col = state.robot_pos
        dr, dc = self._action_to_delta(action)
        r, c = row + dr, col + dc
        # Stay in place if out of bounds or obstacle
        if not (0 <= r < self.get_height() and 0 <= c < self.get_width()):
            r, c = row, col
        elif self._obstacles[r, c]:
            r, c = row, col
        return (r, c)

    def _get_next_bunny_distributions(
        self, state: ChaseState, next_robot_pos: Tuple[int, int]
    ) -> List[Dict[BunnyPosition, float]]:
        # Get next bunny state distributions.
        next_bunny_pos_dists: List[Dict[BunnyPosition, float]] = []
        for bunny_pos in state.bunny_positions:
            dist: DefaultDict[BunnyPosition, float] = defaultdict(float)
            # If the bunny is already gone, it will definitely be gone.
            if bunny_pos is None:
                dist[None] = 1.0
            else:
                stick_prob = 0.5 if self._bunnies_move else 1.0
                # Stay in same place with probability 0.5.
                if bunny_pos == next_robot_pos:
                    dist[None] += stick_prob
                else:
                    dist[bunny_pos] += stick_prob
                # Otherwise move.
                row, col = bunny_pos
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    r, c = row + dr, col + dc
                    # Stay in place if out of bounds or obstacle.
                    if not (0 <= r < self.get_height() and 0 <= c < self.get_width()):
                        r, c = row, col
                    elif self._obstacles[r, c]:
                        r, c = row, col
                    next_bunny_pos = (r, c)
                    if next_bunny_pos == next_robot_pos:
                        dist[None] += (1.0 - stick_prob) * 0.25
                    else:
                        dist[next_bunny_pos] += (1.0 - stick_prob) * 0.25
            next_bunny_pos_dists.append(dist)
        return next_bunny_pos_dists

    def get_transition_distribution(
        self, state: ChaseState, action: ChaseAction
    ) -> Dict[ChaseState, float]:
        # Discrete distributions, represented with a dict
        # mapping next states to probs.
        next_state_dist: Dict[ChaseState, float] = {}
        next_robot_pos = self._get_next_robot_pos(state, action)
        next_bunny_pos_dists = self._get_next_bunny_distributions(state, next_robot_pos)

        # Combine bunny distributions together.
        for outcome in itertools.product(*[d.items() for d in next_bunny_pos_dists]):
            bunny_positions = []
            prob = 1.0
            for pos, p in outcome:
                bunny_positions.append(pos)
                prob *= p
            state = ChaseState(next_robot_pos, tuple(bunny_positions))
            next_state_dist[state] = prob

        return next_state_dist

    def get_reward(
        self, state: ChaseState, action: ChaseAction, next_state: ChaseState
    ) -> float:
        rew = 0.0
        num_bunnies_before = sum(l is not None for l in state.bunny_positions)
        num_bunnies_after = sum(l is not None for l in next_state.bunny_positions)
        num_capture = num_bunnies_before - num_bunnies_after
        rew += num_capture * self._capture_reward
        if self.state_is_terminal(next_state):
            rew += self._goal_reward
        else:
            rew += self._living_reward
        return rew

    @lru_cache(maxsize=None)
    def _get_token_image(self, cell_type: str, tilesize: int) -> Image:
        if cell_type == "robot":
            im = load_image_asset("robot.png")
        elif cell_type == "bunny":
            im = load_image_asset("bunny.png")
        elif cell_type == "obstacle":
            im = load_image_asset("obstacle.png")
        else:
            raise ValueError(f"No asset for {cell_type} known")
        return resize(im[:, :, :3], (tilesize, tilesize, 3), preserve_range=True)

    def render_state(self, state: ChaseState) -> Image:
        height, width = self.get_height(), self.get_width()
        tilesize = 64
        canvas = np.zeros((height * tilesize, width * tilesize, 3))

        for r in range(height):
            for c in range(width):
                if (r, c) == state.robot_pos:
                    cell_type = "robot"
                elif (r, c) in state.bunny_positions:
                    cell_type = "bunny"
                elif self._obstacles[(r, c)]:
                    cell_type = "obstacle"
                else:
                    continue
                im = self._get_token_image(cell_type, tilesize)
                canvas[
                    r * tilesize : (r + 1) * tilesize,
                    c * tilesize : (c + 1) * tilesize,
                ] = im

        return (255 * canvas).astype(np.uint8)


class StaticBunnyChaseMDP(ChaseMDP):
    """Variation where the bunnies don't move."""

    _bunnies_move: ClassVar[bool] = False


class TwoBunnyChaseMDP(ChaseMDP):
    """A small chase MDP with two bunnies in it."""

    @property
    def state_space(self) -> Set[ChaseState]:
        # Subclasses may have multiple bunnies, but by default there is just one.
        pos = [
            (r, c) for r in range(self.get_height()) for c in range(self.get_width())
        ]
        return {
            ChaseState(p1, (p2, p3))
            for p1 in pos
            for p2 in pos + [None]
            for p3 in pos + [None]
        }


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


class LargeChaseMDP(ChaseMDP):
    """A variation with arbitrarily many bunnies in a large free space."""

    _obstacles: ClassVar[NDArray[np.bool_]] = np.zeros((16, 16), dtype=np.bool_)
    _capture_reward: ClassVar[float] = 1.0

    @property
    def state_space(self) -> Set[ChaseState]:
        raise NotImplementedError("State space too large to enumerate.")

    def get_transition_distribution(
        self, state: ChaseState, action: ChaseAction
    ) -> Dict[ChaseState, float]:
        raise NotImplementedError("Transition distribution too large.")

    def sample_next_state(
        self, state: ChaseState, action: ChaseAction, rng: np.random.Generator
    ) -> ChaseState:
        next_robot_pos = self._get_next_robot_pos(state, action)
        next_bunny_pos_dists = self._get_next_bunny_distributions(state, next_robot_pos)
        sampled_bunny_positions = []
        for dist in next_bunny_pos_dists:
            options, probs = zip(*dist.items())
            idxs = len(options)
            idx = rng.choice(idxs, p=probs)
            pos = options[idx]
            sampled_bunny_positions.append(pos)
        return ChaseState(next_robot_pos, tuple(sampled_bunny_positions))
