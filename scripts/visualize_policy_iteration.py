"""Create a visualization of policy iteration running in Chase."""

from typing import Dict, List

import imageio.v2 as iio
from visualize_value_iteration import render_chase_value_function

from mlrp_course.algorithms.policy_iteration import policy_iteration
from mlrp_course.mdp.chase_mdp import ChaseMDP, ChaseState


def _main(outfile: str, fps: int) -> None:
    mdp = ChaseMDP()
    Vs: List[Dict[ChaseState, float]] = policy_iteration(
        mdp,
        max_num_iterations=100,
    )
    imgs = [render_chase_value_function(V) for V in Vs]
    iio.mimsave(outfile, imgs, fps=fps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile", default="policy_iteration.gif")
    parser.add_argument("--fps", default=2)
    args = parser.parse_args()
    _main(args.outfile, args.fps)
