"""Create videos of various MDP planning approaches running in various MDPs."""

from pathlib import Path
from typing import Callable, Tuple

import imageio.v2 as iio
import numpy as np

from mlrp_course.algorithms.expectimax_search import expectimax_search
from mlrp_course.algorithms.finite_horizon_dp import finite_horizon_dp
from mlrp_course.algorithms.policy_iteration import policy_iteration
from mlrp_course.algorithms.rtdp import rtdp
from mlrp_course.algorithms.value_iteration import value_iteration
from mlrp_course.mdp.chase_mdp import ChaseMDP, ChaseState, ChaseWithRoomsMDP
from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState
from mlrp_course.utils import sample_trajectory, value_function_to_greedy_policy


def _sample_chase_initial_state(mdp: ChaseMDP, rng: np.random.Generator) -> ChaseState:
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
    rabbit_loc = ordered_reachable_locs[rng.choice(len(ordered_reachable_locs))]
    return (robot_loc, rabbit_loc)


def _create_mdp_and_initial_state(
    name: str, rng: np.random.Generator
) -> Tuple[DiscreteMDP, DiscreteState]:
    if name == "chase":
        mdp = ChaseMDP()
        initial_state = _sample_chase_initial_state(mdp, rng)
        return mdp, initial_state

    if name == "chase-with-rooms":
        mdp = ChaseWithRoomsMDP()
        initial_state = _sample_chase_initial_state(mdp, rng)
        return mdp, initial_state

    raise NotImplementedError("MDP not supported")


def _create_approach(
    name: str, mdp: DiscreteMDP, rng: np.random.Generator
) -> Callable[[DiscreteState], DiscreteAction]:

    if name == "finite_horizon_dp":
        print("Running finite-horizon DP...")
        V_timed = finite_horizon_dp(mdp)
        print("Done.")
        t = 0

        def dp_pi(s: DiscreteState) -> DiscreteAction:
            """Assume that the policy is called once per time step."""
            nonlocal t
            V = V_timed[t]
            t += 1
            return value_function_to_greedy_policy(V, mdp, rng)(s)

        return dp_pi

    if name == "value_iteration":
        print("Running value iteration...")
        Vs = value_iteration(mdp)
        print("Done.")
        return value_function_to_greedy_policy(Vs[-1], mdp, rng)

    if name == "policy_iteration":
        print("Running policy iteration...")
        Vs = policy_iteration(mdp)
        print("Done.")
        return value_function_to_greedy_policy(Vs[-1], mdp, rng)

    if name == "expectimax_search":

        def expectimax_pi(s: DiscreteState) -> DiscreteAction:
            """Run expectimax search on every step."""
            return expectimax_search(s, mdp, search_horizon=10)

        return expectimax_pi

    if name == "rtdp":

        def rtdp_pi(s: DiscreteState) -> DiscreteAction:
            """Run RTDP on every step."""
            return rtdp(s, mdp, search_horizon=10, rng=rng)

        return rtdp_pi

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
    policy = _create_approach(approach_name, mdp, rng)
    outfile = outdir / f"{mdp_name}_{approach_name}_{seed}.gif"
    if mdp.horizon is not None:
        max_horizon = min(max_horizon, mdp.horizon)
    states, _ = sample_trajectory(initial_state, policy, mdp, max_horizon, rng)
    imgs = [mdp.render_state(s) for s in states]
    iio.mimsave(outfile, imgs, fps=fps)
    print(f"Wrote out to {outfile}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mdp", type=str)
    parser.add_argument("approach", type=str)
    parser.add_argument("--seed", default=0)
    parser.add_argument("--max_horizon", default=100)
    parser.add_argument("--outdir", default=Path("."), type=Path)
    parser.add_argument("--fps", default=2)
    args = parser.parse_args()
    _main(args.mdp, args.approach, args.seed, args.max_horizon, args.outdir, args.fps)
