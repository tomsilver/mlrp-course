"""Experiment comparing Q learning with and without experience replay."""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from mlrp_course.agents import Agent
from mlrp_course.mdp.algorithms.experience_replay import ExperienceReplayHyperparameters
from mlrp_course.mdp.algorithms.q_learning import (
    QLearningAgent,
    QLearningExperienceReplayAgent,
    QLearningHyperparameters,
)
from mlrp_course.mdp.envs.chase_mdp import ChaseState, TwoBunnyChaseMDP
from mlrp_course.utils import DiscreteMDPGymEnv, run_episodes


def _main(
    start_seed: int,
    num_seeds: int,
    max_horizon: int,
    num_episodes: int,
    eval_interval: int,
    num_evals: int,
    num_replays_per_update: int,
    outdir: Path,
    load: bool,
) -> None:
    csv_file = outdir / "experience_replay_experiments.csv"
    if load:
        assert csv_file.exists()
        df = pd.read_csv(csv_file)
        return _df_to_plot(df, outdir)
    columns = ["Seed", "Agent", "Episode", "Returns"]
    results: List[Tuple[int, str, int, float]] = []
    for seed in range(start_seed, start_seed + num_seeds):
        seed_results = _run_single_seed(
            seed,
            max_horizon,
            num_episodes,
            eval_interval,
            num_evals,
            num_replays_per_update,
        )
        for agent, agent_results in seed_results.items():
            for episode, returns in agent_results.items():
                results.append((seed, agent, episode, returns))
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(csv_file)
    return _df_to_plot(df, outdir)


def _df_to_plot(df: pd.DataFrame, outdir: Path) -> None:
    matplotlib.rcParams.update({"font.size": 20})
    fig_file = outdir / "experience_replay_experiments.png"

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
    plt.title("Two Bunny Chase")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_file, dpi=150)
    print(f"Wrote out to {fig_file}")


def _run_single_seed(
    seed: int,
    max_horizon: int,
    num_episodes: int,
    eval_interval: int,
    num_evals: int,
    num_replays_per_update: int,
) -> Dict[str, Dict[int, float]]:
    # Set up environment.
    mdp = TwoBunnyChaseMDP()
    sample_initial_state = lambda _: ChaseState((0, 0), ((1, 2), (1, 1)))
    env = DiscreteMDPGymEnv(mdp, sample_initial_state)
    # Set up agents.
    q_learning_config = QLearningHyperparameters()
    experience_replay_config = ExperienceReplayHyperparameters(
        num_replays_per_update=num_replays_per_update
    )
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
                    agent,
                    env,
                    num_train_episodes,
                    max_episode_length=max_horizon,
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
    parser.add_argument("--num_replays_per_update", default=50, type=int)
    parser.add_argument("--outdir", default=Path("."), type=Path)
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()
    _main(
        args.seed,
        args.num_seeds,
        args.max_horizon,
        args.num_episodes,
        args.eval_interval,
        args.num_evals,
        args.num_replays_per_update,
        args.outdir,
        args.load,
    )
