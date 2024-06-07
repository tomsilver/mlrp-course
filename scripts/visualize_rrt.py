"""Script to visualize RRT."""

from pathlib import Path

import imageio.v2 as iio
import numpy as np
from spatialmath import SE2
from tomsgeoms2d.structs import Circle, Rectangle
from tqdm import tqdm

from mlrp_course.motion.algorithms.rrt import RRTHyperparameters, _build_rrt
from mlrp_course.motion.envs.geom2d_problem import Geom2DMotionPlanningProblem


def _main(outdir: Path, fps: int) -> None:

    world_x_bounds = (0, 5)
    world_y_bounds = (0, 5)
    robot_init_geom = Rectangle.from_center(1, 1, 1, 1, np.pi / 4)
    robot_goal = SE2(4, 4, 0)
    obstacle_geoms = {
        Circle(2.5, 2.5, 1.0),
        Rectangle.from_center(0.5, 4, 1, 2, 0),
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
    for t in tqdm(range(len(nodes))):
        confs = [n.conf for n in nodes[: t + 1]]
        img = problem.render(confs=confs)
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
