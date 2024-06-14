"""Motion planning with rapidly-explored random trees."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Generic, List

import numpy as np

from mlrp_course.motion.algorithms.direct_path import run_direct_path_motion_planning
from mlrp_course.motion.motion_planning_problem import MotionPlanningProblem, RobotConf
from mlrp_course.motion.utils import (
    MotionPlanningHyperparameters,
    find_trajectory_shortcuts,
    iter_traj_with_max_distance,
)
from mlrp_course.utils import (
    Trajectory,
    TrajectorySegment,
    get_trajectory_state_distance,
    point_sequence_to_trajectory,
)


@dataclass(frozen=True)
class RRTHyperparameters(MotionPlanningHyperparameters):
    """Hyperparameters for RRT."""

    num_attempts: int = 10
    num_iters: int = 100
    sample_goal_prob: float = 0.25


# Give nodes unique IDs.
_NODE_ID_COUNT = itertools.count()


@dataclass(frozen=True)
class _RRTNode(Generic[RobotConf]):
    """A node for RRT."""

    conf: RobotConf
    parent: _RRTNode[RobotConf] | None = None
    node_id: int = field(default_factory=lambda: next(_NODE_ID_COUNT))


def run_rrt(
    mpp: MotionPlanningProblem[RobotConf],
    rng: np.random.Generator,
    hyperparameters: RRTHyperparameters | None = None,
) -> Trajectory[RobotConf] | None:
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
    direct_path = run_direct_path_motion_planning(mpp, hyperparameters)
    if direct_path is not None:
        return direct_path

    # Run a number of independent attempts.
    for _ in range(hyperparameters.num_attempts):
        nodes = _build_rrt(mpp, rng, hyperparameters)
        if nodes[-1].conf == mpp.goal_configuration:
            traj = _finish_plan(nodes[-1], hyperparameters.max_velocity)
            # Smooth the trajectory before returning.
            return find_trajectory_shortcuts(traj, rng, mpp, hyperparameters)

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
        extension = TrajectorySegment(node.conf, target)
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


def _get_closest_node(nodes: List[_RRTNode[RobotConf]], target: RobotConf) -> _RRTNode:
    return min(nodes, key=lambda n: get_trajectory_state_distance(n.conf, target))


def _finish_plan(
    node: _RRTNode[RobotConf], max_velocity: float
) -> Trajectory[RobotConf]:
    rev_node_sequence = [node]
    while node.parent is not None:
        node = node.parent
        rev_node_sequence.append(node)
    node_sequence = rev_node_sequence[::-1]
    conf_sequence = [n.conf for n in node_sequence]
    return point_sequence_to_trajectory(conf_sequence, max_velocity)
