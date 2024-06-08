"""Motion planning with rapidly-explored random trees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, List

import numpy as np

from mlrp_course.motion.motion_planning_problem import MotionPlanningProblem, RobotConf
from mlrp_course.motion.utils import (
    ConcatRobotConfTraj,
    RobotConfSegment,
    RobotConfTraj,
    get_robot_conf_distance,
    iter_traj_with_max_distance,
)
from mlrp_course.structs import Hyperparameters


@dataclass(frozen=True)
class RRTHyperparameters(Hyperparameters):
    """Hyperparameters for RRT."""

    max_velocity: float = 1.0
    collision_check_max_distance: float = 1.0
    num_attempts: int = 10
    num_iters: int = 100
    sample_goal_prob: float = 0.25


@dataclass(frozen=True)
class _RRTNode(Generic[RobotConf]):
    """A node for RRT."""

    conf: RobotConf
    parent: _RRTNode[RobotConf] | None = None


def run_rrt(
    mpp: MotionPlanningProblem[RobotConf],
    rng: np.random.Generator,
    hyperparameters: RRTHyperparameters | None = None,
) -> RobotConfTraj[RobotConf] | None:
    """Run RRT to get a collision-free path from start to goal.

    If none is found, returns None.
    """
    if hyperparameters is None:
        hyperparameters = RRTHyperparameters()

    # Check if the problem is obviously impossible.
    if mpp.has_collision(mpp.initial_configuration) or mpp.has_collision(
        mpp.goal_configuration
    ):
        return None

    # Try to just take a direct path from start to goal.
    direct_path = _try_direct_path(
        mpp.initial_configuration,
        mpp.goal_configuration,
        mpp.has_collision,
        hyperparameters,
    )
    if direct_path is not None:
        return direct_path

    # Run a number of independent attempts.
    for _ in range(hyperparameters.num_attempts):
        nodes = _build_rrt(mpp, rng, hyperparameters)
        if nodes[-1].conf == mpp.goal_configuration:
            return _finish_plan(nodes[-1], hyperparameters.max_velocity)

    # No path found, fail.
    return None


def _build_rrt(
    mpp: MotionPlanningProblem[RobotConf],
    rng: np.random.Generator,
    hyperparameters: RRTHyperparameters,
) -> List[_RRTNode[RobotConf]]:
    root = _RRTNode(mpp.initial_configuration)
    nodes = [root]
    for _ in range(hyperparameters.num_iters):
        # Try to go directly to the goal every so often.
        if rng.uniform() < hyperparameters.sample_goal_prob:
            target = mpp.goal_configuration
        # Otherwise, sample from the configuration space.
        else:
            target = mpp.configuration_space.sample()
        # Find the closest node in the tree.
        node = _get_closest_node(nodes, target)
        # Extend the tree in the direction of the target until collision.
        extension = RobotConfSegment(node.conf, target)
        for waypoint in iter_traj_with_max_distance(
            extension,
            hyperparameters.collision_check_max_distance,
            include_start=False,
        ):
            if mpp.has_collision(waypoint):
                break
            node = _RRTNode(waypoint, parent=node)
            nodes.append(node)
        # Check if we reached the goal and quit early if so.
        leaf = nodes[-1]
        if leaf.conf == mpp.goal_configuration:
            break
    return nodes


def _try_direct_path(
    start: RobotConf,
    end: RobotConf,
    has_collision: Callable[[RobotConf], bool],
    hyperparameters: RRTHyperparameters,
) -> RobotConfTraj[RobotConf] | None:
    traj = RobotConfSegment.from_max_velocity(start, end, hyperparameters.max_velocity)
    for waypoint in iter_traj_with_max_distance(
        traj,
        hyperparameters.collision_check_max_distance,
        include_start=False,
    ):
        if has_collision(waypoint):
            return None
    return traj


def _get_closest_node(nodes: List[_RRTNode[RobotConf]], target: RobotConf) -> _RRTNode:
    return min(nodes, key=lambda n: get_robot_conf_distance(n.conf, target))


def _finish_plan(
    node: _RRTNode[RobotConf], max_velocity: float
) -> RobotConfTraj[RobotConf]:
    rev_node_sequence = [node]
    while node.parent is not None:
        node = node.parent
        rev_node_sequence.append(node)
    node_sequence = rev_node_sequence[::-1]
    conf_sequence = [n.conf for n in node_sequence]
    segments = []
    for t in range(len(node_sequence) - 1):
        seg = RobotConfSegment.from_max_velocity(
            conf_sequence[t], conf_sequence[t + 1], max_velocity
        )
        segments.append(seg)
    return ConcatRobotConfTraj(segments)
