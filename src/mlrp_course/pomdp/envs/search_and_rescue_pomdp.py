"""Search and Rescue POMDP."""

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Set, Tuple

import numpy as np

from mlrp_course.pomdp.discrete_pomdp import DiscretePOMDP
from mlrp_course.structs import CategoricalDistribution, Hyperparameters, Image
from mlrp_course.utils import render_avatar_grid


@dataclass(frozen=True, eq=True, order=True)
class SearchAndRescueObs:
    """An observation in the SearchAndRescuePOMDP."""

    robot_loc: Tuple[int, int]
    scan_response: str | None = None  # got-response, got-no-response


@dataclass(frozen=True, eq=True, order=True)
class SearchAndRescueState:
    """A state in the SearchAndRescuePOMDP."""

    robot_loc: Tuple[int, int]
    person_loc: Tuple[int, int]


@dataclass(frozen=True, eq=True, order=True)
class SearchAndRescueAction:
    """An action in the SearchAndRescuePOMDP."""

    type: str  # move or scan
    direction: Tuple[int, int]  # up, down, left, right


@dataclass(frozen=True)
class SearchAndRescuePOMDPHyperparameters(Hyperparameters):
    """Hyperparameters for the SearchAndRescuePOMDP."""

    scan_noise_probability: float = 0.1
    move_noise_probability: float = 0.05
    living_reward: float = -1.0
    rescue_reward: float = 100.0
    fire_reward: float = -100.0


class SearchAndRescuePOMDP(
    DiscretePOMDP[SearchAndRescueObs, SearchAndRescueState, SearchAndRescueAction]
):
    """Search and Rescue POMDP."""

    # Short-hand for grid cells.
    E, W, F, H = _EMPTY, _WALL, _FIRE, _HIDDEN = range(4)

    # Default medium-sized grid.
    _grid = np.array(
        [
            [W, W, H, E, E, E, E],
            [W, F, E, W, E, W, E],
            [F, F, E, E, E, E, H],
            [E, F, E, E, E, F, E],
            [H, E, E, E, E, F, F],
            [E, W, E, W, E, F, W],
            [E, E, E, E, H, W, W],
        ]
    )

    # Subclasses may disable some action directions.
    _action_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(
        self, config: SearchAndRescuePOMDPHyperparameters | None = None
    ) -> None:
        self._config = config or SearchAndRescuePOMDPHyperparameters()

    @cached_property
    def _possible_robot_locs(self) -> List[Tuple[int, int]]:
        mask = self._grid == self._EMPTY
        mask |= self._grid == self._HIDDEN
        mask |= self._grid == self._FIRE
        return [tuple(loc) for loc in np.argwhere(mask)]

    @cached_property
    def _possible_person_locs(self) -> List[Tuple[int, int]]:
        mask = self._grid == self._HIDDEN
        return [tuple(loc) for loc in np.argwhere(mask)]

    @property
    def observation_space(self) -> Set[SearchAndRescueObs]:
        observations: Set[SearchAndRescueObs] = set()
        for scan_response in ["got-response", "got-no-response", None]:
            for robot_loc in self._possible_robot_locs:
                observations.add(SearchAndRescueObs(robot_loc, scan_response))
        return observations

    @property
    def state_space(self) -> Set[SearchAndRescueState]:
        states: Set[SearchAndRescueState] = set()
        for person_loc in self._possible_person_locs:
            for robot_loc in self._possible_robot_locs:
                states.add(SearchAndRescueState(robot_loc, person_loc))
        return states

    @property
    def action_space(self) -> Set[SearchAndRescueAction]:
        actions: Set[SearchAndRescueAction] = set()
        for d in self._action_directions:
            for t in ["move", "scan"]:
                actions.add(SearchAndRescueAction(t, d))
        return actions

    def get_observation_distribution(
        self,
        action: SearchAndRescueAction,
        next_state: SearchAndRescueState,
    ) -> CategoricalDistribution[SearchAndRescueObs]:
        robot_loc = next_state.robot_loc
        if action.type == "move":
            return CategoricalDistribution({SearchAndRescueObs(robot_loc): 1.0})
        assert action.type == "scan"
        # If the action was a scan, get a noisy response.
        # First figure out the "correct" directions to scan.
        person_loc = next_state.person_loc
        correct_scans: List[Tuple[int, int]] = []
        dr, dc = np.subtract(person_loc, robot_loc)
        if dr < 0:
            correct_scans.append((-1, 0))
        elif dr > 0:
            correct_scans.append((1, 0))
        if dc < 0:
            correct_scans.append((0, -1))
        elif dc > 0:
            correct_scans.append((0, 1))
        assert len(correct_scans) <= 2
        if action.direction in correct_scans:
            correct_response = "got-response"
            incorrect_response = "got-no-response"
        else:
            correct_response = "got-no-response"
            incorrect_response = "got-response"
        # Return a noisy response.
        noise_prob = self._config.scan_noise_probability
        dist = {
            SearchAndRescueObs(robot_loc, correct_response): 1.0 - noise_prob,
            SearchAndRescueObs(robot_loc, incorrect_response): noise_prob,
        }
        return CategoricalDistribution(dist)

    def get_initial_observation_distribution(
        self, initial_state: SearchAndRescueState
    ) -> CategoricalDistribution[SearchAndRescueObs]:
        robot_loc = initial_state.robot_loc
        return CategoricalDistribution({SearchAndRescueObs(robot_loc): 1.0})

    def state_is_terminal(self, state: SearchAndRescueState) -> bool:
        return state.robot_loc == state.person_loc

    def get_reward(
        self,
        state: SearchAndRescueState,
        action: SearchAndRescueAction,
        next_state: SearchAndRescueState,
    ) -> float:
        reward = self._config.living_reward
        if self.state_is_terminal(next_state):
            reward += self._config.rescue_reward
        # If in fire, big negative reward.
        next_r, next_c = next_state.robot_loc
        if self._grid[next_r, next_c] == self._FIRE:
            reward += self._config.fire_reward
        return reward

    def get_transition_distribution(
        self, state: SearchAndRescueState, action: SearchAndRescueAction
    ) -> CategoricalDistribution[SearchAndRescueState]:
        # Scanning does not change the state.
        if action.type == "scan":
            return CategoricalDistribution({state: 1.0})
        assert action.type == "move"
        robot_r, robot_c = state.robot_loc
        person_loc = state.person_loc
        # Moving is deterministically null if we're in a fire.
        if self._grid[robot_r, robot_c] == self._FIRE:
            return CategoricalDistribution({state: 1.0})
        # Moving is stochastic otherwise.
        dist: Dict[SearchAndRescueState, float] = defaultdict(float)
        for dr, dc in self._action_directions:
            if (robot_r + dr, robot_c + dc) in self._possible_robot_locs:
                next_robot_loc = (robot_r + dr, robot_c + dc)
            else:
                next_robot_loc = (robot_r, robot_c)
            next_state = SearchAndRescueState(next_robot_loc, person_loc)
            if (dr, dc) == action.direction:
                dist[next_state] += 1.0 - self._config.move_noise_probability
            else:
                N = len(self._action_directions) - 1
                dist[next_state] += self._config.move_noise_probability / N
        return CategoricalDistribution(dist)

    def render_state(self, state: SearchAndRescueState) -> Image:
        avatar_grid = np.full(self._grid.shape, None, dtype=object)
        avatar_grid[self._grid == self._FIRE] = "fire"
        avatar_grid[self._grid == self._HIDDEN] = "hidden"
        avatar_grid[self._grid == self._WALL] = "obstacle"
        if state.robot_loc == state.person_loc:
            avatar_grid[state.robot_loc] = "bunny"
        else:
            avatar_grid[state.robot_loc] = "robot"
        return render_avatar_grid(avatar_grid)


class TinySearchAndRescuePOMDP(SearchAndRescuePOMDP):
    """Tiny version of the SearchAndRescuePOMDP with reduced action space."""

    E, H = SearchAndRescuePOMDP._EMPTY, SearchAndRescuePOMDP._HIDDEN
    _grid = np.array([[H, E, H]])

    _action_directions = [(0, -1), (0, 1)]
