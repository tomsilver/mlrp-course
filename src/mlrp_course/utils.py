"""Utilities."""

from pathlib import Path
from typing import Dict, List, Tuple, TypeVar

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

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


_T = TypeVar("_T", bound=HashableComparable)


def sample_from_categorical(dist: Dict[_T, float], rng: np.random.Generator) -> _T:
    """Sample from a categorical distribution."""
    candidates, probs = zip(*dist.items(), strict=True)
    idx = rng.choice(len(candidates), p=probs)
    return candidates[idx]


def load_image_asset(filename: str) -> Image:
    """Load an image asset."""
    asset_dir = Path(__file__).parent / "assets"
    image_file = asset_dir / filename
    return plt.imread(image_file)


def fig2data(fig: plt.Figure) -> Image:
    """Convert matplotlib figure into Image."""
    fig.canvas.draw()
    return np.array(fig.canvas.renderer.buffer_rgba())
