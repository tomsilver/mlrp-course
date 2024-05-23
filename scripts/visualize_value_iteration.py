"""Create a visualization of value iteration running in Chase."""

from typing import Dict, List, Tuple

import imageio.v2 as iio
import numpy as np
from matplotlib import pyplot as plt

from mlrp_course.algorithms.value_iteration import value_iteration
from mlrp_course.mdp.chase_mdp import ChaseMDP, ChaseState
from mlrp_course.structs import Image
from mlrp_course.utils import fig2data

HEIGHT, WIDTH = ChaseMDP._height, ChaseMDP._width  # pylint: disable=protected-access


def _render_value_function(value_function: Dict[ChaseState, float]) -> Image:
    fig, axes = plt.subplots(HEIGHT, WIDTH)
    for r in range(HEIGHT):
        for c in range(WIDTH):
            ax = axes[r][c]
            _render_value_function_rabbit_pos(ax, value_function, (r, c))
    fig.tight_layout()
    return fig2data(fig)


def _render_value_function_rabbit_pos(
    ax: plt.Axes, value_function: Dict[ChaseState, float], rabbit_pos: Tuple[int, int]
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
    Vs: List[Dict[ChaseState, float]] = value_iteration(
        mdp, max_num_iterations=100, print_every=1
    )
    imgs = [_render_value_function(V) for V in Vs]
    iio.mimsave(outfile, imgs, fps=fps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile", default="value_iteration.gif")
    parser.add_argument("--fps", default=2)
    args = parser.parse_args()
    _main(args.outfile, args.fps)
