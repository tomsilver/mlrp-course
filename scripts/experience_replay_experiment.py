"""Experiment comparing Q learning with and without experience replay."""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from mlrp_course.agents import Agent
from mlrp_course.algorithms.experience_replay import ExperienceReplayConfig
from mlrp_course.algorithms.q_learning import (
    QLearningAgent,
    QLearningConfig,
    QLearningExperienceReplayAgent,
)
from mlrp_course.mdp.chase_mdp import ChaseState, TwoBunnyChaseMDP
from mlrp_course.utils import DiscreteMDPGymEnv, run_episodes


def _main(
    start_seed: int,
    num_seeds: int,
    max_horizon: int,
    num_episodes: int,
    eval_interval: int,
    num_evals: int,
    outdir: Path,
) -> None:
    columns = ["Seed", "Agent", "Episode", "Returns"]
    results: List[Tuple[int, str, int, float]] = []
    for seed in range(start_seed, start_seed + num_seeds):
        seed_results = _run_single_seed(
            seed, max_horizon, num_episodes, eval_interval, num_evals
        )
        for agent, agent_results in seed_results.items():
            for episode, returns in agent_results.items():
                results.append((seed, agent, episode, returns))
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(outdir / "experience_replay_experiments.csv")
    grouped = df.groupby(["Agent", "Episode"]).agg({"Returns": ["mean", "sem"]})
    grouped.columns = grouped.columns.droplevel(0)
    grouped = grouped.rename(columns={"mean": "Returns_mean", "sem": "Returns_sem"})
    grouped = grouped.reset_index()
    plt.figure(figsize=(10, 6))

    for agent in grouped["Agent"].unique():
        agent_data = grouped[grouped["Agent"] == agent]
        plt.plot(agent_data["Episode"], agent_data["Returns_mean"], label=agent)
        plt.fill_between(
            agent_data["Episode"],
            agent_data["Returns_mean"] - agent_data["Returns_sem"],
            agent_data["Returns_mean"] + agent_data["Returns_sem"],
            alpha=0.2,
        )

    plt.xlabel("Episode")
    plt.ylabel("Returns")
    plt.title("Agent Performance over Episodes")
    plt.legend(title="Agent")
    plt.grid(True)
    fig_file = outdir / "experience_replay_experiments.png"
    plt.savefig(fig_file)
    print(f"Wrote out to {fig_file}")


def _run_single_seed(
    seed: int,
    max_horizon: int,
    num_episodes: int,
    eval_interval: int,
    num_evals: int,
) -> Dict[str, Dict[int, float]]:
    # Set up environment.
    mdp = TwoBunnyChaseMDP()
    sample_initial_state = lambda _: ChaseState((0, 0), ((1, 2), (1, 1)))
    env = DiscreteMDPGymEnv(mdp, sample_initial_state)
    # Set up agents.
    q_learning_config = QLearningConfig()
    experience_replay_config = ExperienceReplayConfig()
    no_replay_agent = QLearningAgent(
        mdp.action_space, mdp.temporal_discount_factor, q_learning_config, seed
    )
    replay_agent = QLearningExperienceReplayAgent(
        mdp.action_space,
        mdp.temporal_discount_factor,
        q_learning_config,
        experience_replay_config,
        seed,
    )
    agents: Dict[str, Agent] = {
        "Q Learning (No Replay)": no_replay_agent,
        "Q Learning (Replay)": replay_agent,
    }
    results: Dict[str, Dict[int, float]] = {}
    for agent_name, agent in agents.items():
        print(f"Starting {agent_name} (seed={seed})")
        results[agent_name] = {}
        episode = 0
        env.reset(seed=seed)
        with tqdm(total=num_episodes) as pbar:
            while True:
                # Run eval episodes.
                # TODO: why do lines not start at the same place when x=0?
                agent.eval()
                eval_results = run_episodes(
                    agent, env, num_evals, max_episode_length=max_horizon
                )
                episode_returns = 0.0
                for _, _, rewards in eval_results:
                    episode_returns += sum(rewards)
                results[agent_name][episode] = episode_returns / num_evals
                if episode >= num_episodes:
                    break
                # Run training episodes.
                agent.train()
                num_train_episodes = min(eval_interval, num_episodes - episode)
                episode += num_train_episodes
                pbar.update(num_train_episodes)
                run_episodes(
                    agent, env, num_train_episodes, max_episode_length=max_horizon
                )
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_seeds", default=25, type=int)
    parser.add_argument("--max_horizon", default=10, type=int)
    parser.add_argument("--num_episodes", default=2500, type=int)
    parser.add_argument("--eval_interval", default=100, type=int)
    parser.add_argument("--num_evals", default=10, type=int)
    parser.add_argument("--outdir", default=Path("."), type=Path)
    args = parser.parse_args()
    _main(
        args.seed,
        args.num_seeds,
        args.max_horizon,
        args.num_episodes,
        args.eval_interval,
        args.num_evals,
        args.outdir,
    )
