"""Create a visualization of value iteration running in Chase."""

from typing import Dict, Tuple

import imageio.v2 as iio
import numpy as np
from matplotlib import pyplot as plt

from mlrp_course.mdp.algorithms.value_iteration import (
    ValueIterationConfig,
    value_iteration,
)
from mlrp_course.mdp.discrete_mdp import DiscreteState
from mlrp_course.mdp.envs.chase_mdp import ChaseMDP
from mlrp_course.structs import Image
from mlrp_course.utils import fig2data

HEIGHT, WIDTH = ChaseMDP.get_height(), ChaseMDP.get_width()


def render_chase_value_function(value_function: Dict[DiscreteState, float]) -> Image:
    """Render a value function in the Chase MDP."""
    fig, axes = plt.subplots(HEIGHT, WIDTH)
    for r in range(HEIGHT):
        for c in range(WIDTH):
            ax = axes[r][c]
            _render_value_function_rabbit_pos(ax, value_function, (r, c))
    fig.tight_layout()
    return fig2data(fig)


def _render_value_function_rabbit_pos(
    ax: plt.Axes,
    value_function: Dict[DiscreteState, float],
    rabbit_pos: Tuple[int, int],
) -> None:
    value_grid = np.zeros((HEIGHT, WIDTH))
    for r in range(HEIGHT):
        for c in range(WIDTH):
            v = value_function[((r, c), rabbit_pos)]
            value_grid[r, c] = v
            if v < 0.5:
                color = "white"
            else:
                color = "black"
            ax.text(c, r, np.round(v, 4), ha="center", va="center", color=color)
    ax.imshow(value_grid)
    ax.axis("off")
    ax.axis("off")
    ax.set_title(f"Rabbit Pos={rabbit_pos}")


def _main(outfile: str, fps: int) -> None:
    mdp = ChaseMDP()
    config = ValueIterationConfig(max_num_iterations=100, print_every=1)
    Vs = value_iteration(mdp, config)
    imgs = [render_chase_value_function(V) for V in Vs]
    iio.mimsave(outfile, imgs, fps=fps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile", default="value_iteration.gif")
    parser.add_argument("--fps", default=2)
    args = parser.parse_args()
    _main(args.outfile, args.fps)
