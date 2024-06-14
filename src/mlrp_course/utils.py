"""Utilities."""

from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, TypeVar

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from skimage.transform import resize  # pylint: disable=no-name-in-module

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
