"""Script to visualize RRT."""

from pathlib import Path

import imageio.v2 as iio
import numpy as np
from matplotlib import pyplot as plt
from spatialmath import SE2
from tomsgeoms2d.structs import Circle, Rectangle
from tqdm import tqdm

from mlrp_course.motion.algorithms.rrt import RRTHyperparameters, _build_rrt
from mlrp_course.motion.envs.geom2d_problem import (
    Geom2DMotionPlanningProblem,
    _copy_geom_with_pose,
)
from mlrp_course.utils import fig2data


def _main(outdir: Path, fps: int) -> None:

    world_x_bounds = (0, 10)
    world_y_bounds = (0, 10)
    robot_init_geom = Rectangle.from_center(1, 1, 1, 1, np.pi / 4)
    robot_goal = SE2(9, 9, 0)
    obstacle_geoms = {
        Circle(2.5, 2.5, 1.0),
        Rectangle.from_center(0.5, 4, 1, 2, 0),
        Rectangle.from_center(1, 8, 1, 4, np.pi / 3),
        Rectangle.from_center(7, 7, 1, 4, -np.pi / 3),
    }
    problem = Geom2DMotionPlanningProblem(
        world_x_bounds,
        world_y_bounds,
        robot_init_geom,
        robot_goal,
        obstacle_geoms,
        seed=123,
    )
    rng = np.random.default_rng(123)
    hyperparameters = RRTHyperparameters()
    nodes = _build_rrt(problem, rng, hyperparameters)
    imgs = []

    print("Creating video...")
    world_min_x, world_max_x = world_x_bounds
    world_min_y, world_max_y = world_y_bounds

    figsize = (
        world_max_x - world_min_x,
        world_max_y - world_min_y,
    )
    fig, ax = plt.subplots(
        1, 1, figsize=figsize, dpi=Geom2DMotionPlanningProblem.render_dpi
    )
    pad_x = (world_max_x - world_min_x) / 25
    pad_y = (world_max_y - world_min_y) / 25
    ax.set_xlim(world_min_x - pad_x, world_max_x + pad_x)
    ax.set_ylim(world_min_y - pad_y, world_max_y + pad_y)
    ax.set_xticks([])
    ax.set_yticks([])

    for obstacle_geom in obstacle_geoms:
        obstacle_geom.plot(ax, **Geom2DMotionPlanningProblem.obstacle_render_kwargs)
    robot_goal_geom = _copy_geom_with_pose(robot_init_geom, robot_goal)
    robot_goal_geom.plot(ax, **Geom2DMotionPlanningProblem.robot_goal_render_kwargs)
    robot_init_geom.plot(ax, **Geom2DMotionPlanningProblem.robot_current_render_kwargs)
    plt.tight_layout()

    for node in tqdm(nodes):
        geom = _copy_geom_with_pose(robot_init_geom, node.conf)
        geom.plot(ax, fc=(0.7, 0.1, 0.9, 0.5), ec=(0.3, 0.3, 0.3))
        img = fig2data(fig)
        imgs.append(img)

    outfile = outdir / "rrt.gif"
    iio.mimsave(outfile, imgs, fps=fps)
    print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results", type=Path)
    parser.add_argument("--fps", default=30)
    args = parser.parse_args()
    _main(args.outdir, args.fps)
