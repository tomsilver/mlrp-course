"""Create videos of various MDP planning approaches running in various MDPs."""

from pathlib import Path
from typing import List, Tuple

import imageio.v2 as iio
import numpy as np
from tqdm import tqdm

from mlrp_course.agents import Agent
from mlrp_course.mdp.algorithms.expectimax_search import (
    ExpectimaxSearchAgent,
)
from mlrp_course.mdp.algorithms.finite_horizon_dp import (
    FiniteHorizonDPAgent,
)
from mlrp_course.mdp.algorithms.mcts import MCTSAgent
from mlrp_course.mdp.algorithms.policy_iteration import (
    PolicyIterationAgent,
)
from mlrp_course.mdp.algorithms.rtdp import RTDPAgent
from mlrp_course.mdp.algorithms.sparse_sampling import (
    SparseSamplingAgent,
)
from mlrp_course.mdp.algorithms.value_iteration import (
    ValueIterationAgent,
)
from mlrp_course.mdp.discrete_mdp import DiscreteMDP, DiscreteState
from mlrp_course.mdp.envs.chase_mdp import (
    ChaseMDP,
    ChaseState,
    ChaseWithLargeRoomsMDP,
    ChaseWithRoomsMDP,
    LargeChaseMDP,
    TwoBunnyChaseMDP,
)
from mlrp_course.structs import Image


def _sample_chase_initial_state(
    mdp: ChaseMDP, rng: np.random.Generator, num_bunnies: int = 1
) -> ChaseState:
    obstacles = mdp._obstacles  # pylint: disable=protected-access
    free_spaces = [tuple(loc) for loc in np.argwhere(~obstacles)]
    # Sample a random starting location for the robot.
    robot_loc = free_spaces[rng.choice(len(free_spaces))]
    # Sample a random location for the rabbit that is in the same room, but
    # not equal to the location of the robot.
    reachable_locs = {robot_loc}
    queue = [robot_loc]
    while queue:
        r, c = queue.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < obstacles.shape[0]
                and 0 <= nc < obstacles.shape[1]
                and not obstacles[nr, nc]
                and (nr, nc) not in reachable_locs
            ):
                reachable_locs.add((nr, nc))
                queue.append((nr, nc))
    ordered_reachable_locs = sorted(reachable_locs)
    ordered_reachable_locs.remove(robot_loc)
    idxs = rng.choice(len(ordered_reachable_locs), size=num_bunnies)
    rabbit_locs = tuple(ordered_reachable_locs[i] for i in idxs)
    return ChaseState(robot_loc, rabbit_locs)


def _create_mdp_and_initial_state(
    name: str, rng: np.random.Generator
) -> Tuple[DiscreteMDP, DiscreteState]:
    if name == "chase":
        mdp = ChaseMDP()
        initial_state = _sample_chase_initial_state(mdp, rng)
        return mdp, initial_state

    if name == "chase-two-bunnies":
        mdp = TwoBunnyChaseMDP()
        initial_state = _sample_chase_initial_state(mdp, rng, num_bunnies=2)
        return mdp, initial_state

    if name == "chase-with-rooms":
        mdp = ChaseWithRoomsMDP()
        initial_state = _sample_chase_initial_state(mdp, rng)
        return mdp, initial_state

    if name == "chase-with-large-rooms":
        mdp = ChaseWithLargeRoomsMDP()
        initial_state = _sample_chase_initial_state(mdp, rng)
        return mdp, initial_state

    if name == "chase-large":
        mdp = LargeChaseMDP()
        initial_state = _sample_chase_initial_state(mdp, rng, num_bunnies=5)
        return mdp, initial_state

    raise NotImplementedError("MDP not supported")


def _create_agent(
    name: str,
    mdp: DiscreteMDP,
    seed: int,
) -> Agent:

    if name == "finite_horizon_dp":
        return FiniteHorizonDPAgent(mdp, seed)

    if name == "value_iteration":
        return ValueIterationAgent(mdp, seed)

    if name == "policy_iteration":
        return PolicyIterationAgent(mdp, seed)

    if name == "expectimax_search":
        return ExpectimaxSearchAgent(mdp, seed)

    if name == "sparse_sampling":
        return SparseSamplingAgent(mdp, seed)

    if name == "rtdp":
        return RTDPAgent(mdp, seed)

    if name == "mcts":
        return MCTSAgent(mdp, seed)

    raise NotImplementedError("Approach not found.")


def _main(
    mdp_name: str,
    approach_name: str,
    seed: int,
    max_horizon: int,
    outdir: Path,
    fps: int,
) -> None:
    rng = np.random.default_rng(seed)
    mdp, initial_state = _create_mdp_and_initial_state(mdp_name, rng)
    agent = _create_agent(approach_name, mdp, seed)
    outfile = outdir / f"{mdp_name}_{approach_name}_{seed}.gif"
    if mdp.horizon is not None:
        max_horizon = min(max_horizon, mdp.horizon)
    agent.reset(initial_state)
    states: List[DiscreteState] = [initial_state]
    state = initial_state
    assert not mdp.state_is_terminal(state)
    print("Sampling trajectory...")
    for _ in range(max_horizon):
        action = agent.step()
        next_state = mdp.sample_next_state(state, action, rng)
        reward = mdp.get_reward(state, action, next_state)
        done = mdp.state_is_terminal(next_state)
        agent.update(next_state, reward, done)
        state = next_state
        states.append(state)
        if done:
            break
    print("Done.")
    print("Rendering...")
    imgs: List[Image] = []
    for s in tqdm(states):
        img = mdp.render_state(s)
        imgs.append(img)
    iio.mimsave(outfile, imgs, fps=fps)
    print(f"Wrote out to {outfile}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mdp", type=str)
    parser.add_argument("approach", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max_horizon", default=100, type=int)
    parser.add_argument("--outdir", default=Path("."), type=Path)
    parser.add_argument("--fps", default=2, type=int)
    args = parser.parse_args()
    _main(args.mdp, args.approach, args.seed, args.max_horizon, args.outdir, args.fps)
