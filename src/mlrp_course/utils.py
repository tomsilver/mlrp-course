"""Utilities."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from functools import cached_property, lru_cache, singledispatch
from pathlib import Path
from typing import Generic, Iterator, List, Sequence, Tuple, TypeVar

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from skimage.transform import resize  # pylint: disable=no-name-in-module
from spatialmath import SE2

from mlrp_course.agent import Agent
from mlrp_course.structs import HashableComparable, Image

_O = TypeVar("_O", bound=HashableComparable)
_A = TypeVar("_A", bound=HashableComparable)


def run_episodes(
    agent: Agent[_O, _A],
    env: gym.Env,
    num_episodes: int,
    max_episode_length: int,
) -> List[Tuple[List[_O], List[_A], List[float]]]:
    """Run episodic interactions between an environment and agent."""
    traces: List[Tuple[List[_O], List[_A], List[float]]] = []
    for _ in range(num_episodes):
        observations: List[_O] = []
        actions: List[_A] = []
        rewards: List[float] = []
        obs, _ = env.reset()
        agent.reset(obs)
        observations.append(obs)
        for _ in range(max_episode_length):
            action = agent.step()
            actions.append(action)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            done = terminated or truncated
            agent.update(next_obs, reward, done)
            obs = next_obs
            observations.append(obs)
            if done:
                break
        traces.append((observations, actions, rewards))
    return traces


def load_avatar_asset(filename: str) -> Image:
    """Load an image of an avatar."""
    asset_dir = Path(__file__).parent / "assets" / "avatars"
    image_file = asset_dir / filename
    return plt.imread(image_file)


def load_pddl_asset(filename: str) -> str:
    """Load a PDDL string from assets."""
    asset_dir = Path(__file__).parent / "assets" / "pddl"
    pddl_file = asset_dir / filename
    with open(pddl_file, "r", encoding="utf-8") as f:
        s = f.read()
    return s


@lru_cache(maxsize=None)
def get_avatar_by_name(avatar_name: str, tilesize: int) -> Image:
    """Helper for rendering."""
    if avatar_name == "robot":
        im = load_avatar_asset("robot.png")
    elif avatar_name == "bunny":
        im = load_avatar_asset("bunny.png")
    elif avatar_name == "obstacle":
        im = load_avatar_asset("obstacle.png")
    elif avatar_name == "fire":
        im = load_avatar_asset("fire.png")
    elif avatar_name == "hidden":
        im = load_avatar_asset("hidden.png")
    else:
        raise ValueError(f"No asset for {avatar_name} known")
    return resize(im[:, :, :3], (tilesize, tilesize, 3), preserve_range=True)


def render_avatar_grid(avatar_grid: NDArray, tilesize: int = 64) -> Image:
    """Helper for rendering."""
    height, width = avatar_grid.shape
    canvas = np.zeros((height * tilesize, width * tilesize, 3))

    for r in range(height):
        for c in range(width):
            avatar_name: str | None = avatar_grid[r, c]
            if avatar_name is None:
                continue
            im = get_avatar_by_name(avatar_name, tilesize)
            canvas[
                r * tilesize : (r + 1) * tilesize,
                c * tilesize : (c + 1) * tilesize,
            ] = im

    return (255 * canvas).astype(np.uint8)


def fig2data(fig: plt.Figure) -> Image:
    """Convert matplotlib figure into Image."""
    fig.canvas.draw()
    return np.array(fig.canvas.renderer.buffer_rgba())


def wrap_angle(angle: float) -> float:
    """Wrap angles between -np.pi and np.pi."""
    return np.arctan2(np.sin(angle), np.cos(angle))


TrajectoryState = TypeVar("TrajectoryState")


class Trajectory(Generic[TrajectoryState]):
    """A continuous-time trajectory."""

    @property
    @abc.abstractmethod
    def duration(self) -> float:
        """The length of the trajectory in time."""

    @property
    @abc.abstractmethod
    def distance(self) -> float:
        """The length of the trajectory in distance."""

    @abc.abstractmethod
    def __call__(self, time: float) -> TrajectoryState:
        """Get the state at the given time."""

    def __getitem__(self, key: float | slice):
        """Shorthand for indexing or sub-trajectory creation."""
        if isinstance(key, float):
            return self(key)
        assert isinstance(key, slice)
        assert key.step is None
        start = key.start or 0
        end = key.stop or self.duration
        return self.get_sub_trajectory(start, end)

    @abc.abstractmethod
    def get_sub_trajectory(
        self, start_time: float, end_time: float
    ) -> Trajectory[TrajectoryState]:
        """Create a new trajectory with time re-indexed."""

    @abc.abstractmethod
    def reverse(self) -> Trajectory[TrajectoryState]:
        """Create a new trajectory with time re-indexed."""


@dataclass(frozen=True)
class TrajectorySegment(Trajectory[TrajectoryState]):
    """A trajectory defined by a single start and end point."""

    start: TrajectoryState
    end: TrajectoryState
    _duration: float = 1.0

    @classmethod
    def from_max_velocity(
        cls, start: TrajectoryState, end: TrajectoryState, max_velocity: float
    ) -> TrajectorySegment:
        """Create a segment from a given max velocity."""
        distance = get_trajectory_state_distance(start, end)
        duration = distance / max_velocity
        return TrajectorySegment(start, end, duration)

    @cached_property
    def duration(self) -> float:
        return self._duration

    @cached_property
    def distance(self) -> float:
        return get_trajectory_state_distance(self.start, self.end)

    def __call__(self, time: float) -> TrajectoryState:
        # Avoid numerical issues.
        time = np.clip(time, 0, self.duration)
        s = time / self.duration
        return interpolate_trajectory_states(self.start, self.end, s)

    def get_sub_trajectory(
        self, start_time: float, end_time: float
    ) -> Trajectory[TrajectoryState]:
        elapsed_time = end_time - start_time
        frac = elapsed_time / self.duration
        new_duration = frac * self.duration
        return TrajectorySegment(self(start_time), self(end_time), new_duration)

    def reverse(self) -> Trajectory[TrajectoryState]:
        return TrajectorySegment(self.end, self.start, self.duration)


@dataclass(frozen=True)
class ConcatTrajectory(Trajectory[TrajectoryState]):
    """A trajectory that concatenates other trajectories."""

    trajs: Sequence[Trajectory[TrajectoryState]]

    @cached_property
    def duration(self) -> float:
        return sum(t.duration for t in self.trajs)

    @cached_property
    def distance(self) -> float:
        return sum(t.distance for t in self.trajs)

    def __call__(self, time: float) -> TrajectoryState:
        # Avoid numerical issues.
        time = np.clip(time, 0, self.duration)
        start_time = 0.0
        for traj in self.trajs:
            end_time = start_time + traj.duration
            if time <= end_time:
                assert time >= start_time
                return traj(time - start_time)
            start_time = end_time
        raise ValueError(f"Time {time} exceeds duration {self.duration}")

    def get_sub_trajectory(
        self, start_time: float, end_time: float
    ) -> Trajectory[TrajectoryState]:
        new_trajs = []
        st = 0.0
        keep_traj = False
        for traj in self.trajs:
            et = st + traj.duration
            # Start keeping trajectories.
            if st <= start_time <= et:
                keep_traj = True
                # Shorten the current trajectory so it starts at start_time.
                traj = traj.get_sub_trajectory(start_time - st, traj.duration)
                st = start_time
            # Stop keeping trajectories.
            if st <= end_time <= et:
                # Shorten the current trajectory so it ends at end_time.
                traj = traj.get_sub_trajectory(0, end_time - st)
                # Finish.
                assert keep_traj
                new_trajs.append(traj)
                break
            if keep_traj:
                new_trajs.append(traj)
            st = et
        return concatenate_trajectories(new_trajs)

    def reverse(self) -> Trajectory[TrajectoryState]:
        return ConcatTrajectory([t.reverse() for t in self.trajs][::-1])


def concatenate_trajectories(
    trajectories: Sequence[Trajectory[TrajectoryState]],
) -> Trajectory[TrajectoryState]:
    """Concatenate one or more trajectories."""
    inner_trajs: List[Trajectory[TrajectoryState]] = []
    for traj in trajectories:
        if isinstance(traj, ConcatTrajectory):
            inner_trajs.extend(traj.trajs)
        else:
            inner_trajs.append(traj)
    return ConcatTrajectory(inner_trajs)


@singledispatch
def interpolate_trajectory_states(
    start: TrajectoryState, end: TrajectoryState, s: float
) -> TrajectoryState:
    """Get a point on the interpolated path between start and end.

    The argument is a value between 0 and 1.
    """
    raise NotImplementedError


@interpolate_trajectory_states.register
def _(start: SE2, end: SE2, s: float) -> SE2:
    return start.interp(end, s)


@singledispatch
def get_trajectory_state_distance(
    start: TrajectoryState, end: TrajectoryState
) -> float:
    """Get the distance between two trajectory states."""
    raise NotImplementedError


@get_trajectory_state_distance.register
def _(start: SE2, end: SE2) -> float:
    # Many choices are possible. Here we take the maximum of the translation
    # distance and a scaled-down angular distance.
    angular_scale = 0.1
    difference = start.inv() * end
    assert isinstance(difference, SE2)
    translate_distance = np.sqrt(difference.x**2 + difference.y**2)
    angular_distance = angular_scale * abs(difference.theta())
    return max(translate_distance, angular_distance)


def iter_traj_with_max_distance(
    traj: Trajectory[TrajectoryState],
    max_distance: float,
    include_start: bool = True,
    include_end: bool = True,
) -> Iterator[TrajectoryState]:
    """Iterate through the trajectory while guaranteeing that the distance in
    each step is no more than the given max distance."""
    num_steps = int(np.ceil(traj.distance / max_distance)) + 1
    ts = np.linspace(0, traj.duration, num=num_steps, endpoint=True)
    if not include_start:
        ts = ts[1:]
    if not include_end:
        ts = ts[:-1]
    for t in ts:
        yield traj(t)


def state_sequence_to_trajectory(
    state_sequence: List[TrajectoryState], max_velocity: float
) -> Trajectory[TrajectoryState]:
    """Convert a sequence of states to a trajectory."""
    segments = []
    for t in range(len(state_sequence) - 1):
        seg = TrajectorySegment.from_max_velocity(
            state_sequence[t], state_sequence[t + 1], max_velocity
        )
        segments.append(seg)
    return concatenate_trajectories(segments)
