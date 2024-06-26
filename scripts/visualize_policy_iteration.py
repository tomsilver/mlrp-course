"""Create a visualization of policy iteration running in Chase."""

import imageio.v2 as iio
from visualize_value_iteration import render_chase_value_function

from mlrp_course.mdp.algorithms.policy_iteration import (
    PolicyIterationHyperparameters,
    policy_iteration,
)
from mlrp_course.mdp.envs.chase_mdp import ChaseMDP


def _main(outfile: str, fps: int) -> None:
    mdp = ChaseMDP()
    config = PolicyIterationHyperparameters(max_num_iterations=100)
    Vs = policy_iteration(mdp, config)
    imgs = [render_chase_value_function(V) for V in Vs]
    iio.mimsave(outfile, imgs, fps=fps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile", default="policy_iteration.gif")
    parser.add_argument("--fps", default=2)
    args = parser.parse_args()
    _main(args.outfile, args.fps)
