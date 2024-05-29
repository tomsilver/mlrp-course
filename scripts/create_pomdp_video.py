"""Create videos of various POMDP planning approaches in various POMDPs."""

from pathlib import Path
from typing import List, Tuple

import imageio.v2 as iio
import numpy as np
from tqdm import tqdm

from mlrp_course.agent import Agent
from mlrp_course.mdp.discrete_mdp import DiscreteState
from mlrp_course.pomdp.algorithms.expectimax_search import (
    POMDPExpectimaxSearchAgent,
)
from mlrp_course.pomdp.discrete_pomdp import DiscretePOMDP
from mlrp_course.pomdp.envs.search_and_rescue_pomdp import (
    SearchAndRescuePOMDP,
    SearchAndRescueState,
    TinySearchAndRescuePOMDP,
)
from mlrp_course.structs import Image


def _sample_search_and_rescue_initial_state(
    pomdp: SearchAndRescuePOMDP,
    rng: np.random.Generator,
) -> SearchAndRescueState:
    # pylint: disable=protected-access
    possible_person_locs = sorted(pomdp._possible_person_locs)
    person_loc = possible_person_locs[rng.choice(len(possible_person_locs))]
    possible_robot_locs = sorted(
        set(pomdp._possible_robot_locs) - set(pomdp._possible_person_locs)
    )
    robot_loc = possible_robot_locs[rng.choice(len(possible_robot_locs))]
    return SearchAndRescueState(robot_loc, person_loc)


def _create_pomdp_and_initial_state(
    name: str, rng: np.random.Generator
) -> Tuple[DiscretePOMDP, DiscreteState]:
    if name == "tiny-search-and-rescue":
        pomdp = TinySearchAndRescuePOMDP()
        initial_state = _sample_search_and_rescue_initial_state(pomdp, rng)
        return pomdp, initial_state

    raise NotImplementedError("MDP not supported")


def _create_agent(
    name: str,
    pomdp: DiscretePOMDP,
    seed: int,
) -> Agent:

    if name == "expectimax_search":
        return POMDPExpectimaxSearchAgent(pomdp, seed)

    raise NotImplementedError("Approach not found.")


def _main(
    pomdp_name: str,
    approach_name: str,
    seed: int,
    max_horizon: int,
    outdir: Path,
    fps: int,
) -> None:
    rng = np.random.default_rng(seed)
    pomdp, initial_state = _create_pomdp_and_initial_state(pomdp_name, rng)
    agent = _create_agent(approach_name, pomdp, seed)
    outfile = outdir / f"{pomdp_name}_{approach_name}_{seed}.gif"
    if pomdp.horizon is not None:
        max_horizon = min(max_horizon, pomdp.horizon)
    initial_obs = pomdp.sample_initial_observation(initial_state, rng)
    agent.reset(initial_obs)
    states: List[DiscreteState] = [initial_state]
    state = initial_state
    assert not pomdp.state_is_terminal(state)
    print("Sampling trajectory...")
    for _ in range(max_horizon):
        action = agent.step()
        next_state = pomdp.sample_next_state(state, action, rng)
        obs = pomdp.sample_observation(action, next_state, rng)
        reward = pomdp.get_reward(state, action, next_state)
        done = pomdp.state_is_terminal(next_state)
        agent.update(obs, reward, done)
        state = next_state
        states.append(state)
        if done:
            break
    print("Done.")
    print("Rendering...")
    imgs: List[Image] = []
    for s in tqdm(states):
        img = pomdp.render_state(s)
        imgs.append(img)
    iio.mimsave(outfile, imgs, fps=fps)
    print(f"Wrote out to {outfile}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pomdp", type=str)
    parser.add_argument("approach", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max_horizon", default=100, type=int)
    parser.add_argument("--outdir", default=Path("."), type=Path)
    parser.add_argument("--fps", default=2, type=int)
    args = parser.parse_args()
    _main(args.pomdp, args.approach, args.seed, args.max_horizon, args.outdir, args.fps)
