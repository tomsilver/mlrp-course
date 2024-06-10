"""Motion planning with rapidly-explored random trees."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from mlrp_course.motion.algorithms.direct_path import run_direct_path_motion_planning
from mlrp_course.motion.algorithms.rrt import (
    RRTHyperparameters,
    _finish_plan,
    _get_closest_node,
    _RRTNode,
)
from mlrp_course.motion.motion_planning_problem import MotionPlanningProblem, RobotConf
from mlrp_course.motion.utils import (
    RobotConfSegment,
    RobotConfTraj,
    concatenate_robot_conf_trajectories,
    find_trajectory_shortcuts,
    iter_traj_with_max_distance,
)


def run_birrt(
    mpp: MotionPlanningProblem[RobotConf],
    rng: np.random.Generator,
    hyperparameters: RRTHyperparameters | None = None,
) -> RobotConfTraj[RobotConf] | None:
    """Run BiRRT to get two trees, one from init and one from goal.

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
        nodes_from_init, nodes_from_goal = _build_birrt(mpp, hyperparameters)
        assert nodes_from_init[0].conf == mpp.initial_configuration
        assert nodes_from_goal[0].conf == mpp.goal_configuration
        if nodes_from_init[-1].conf == nodes_from_goal[-1].conf:
            # Combine the two trajectories.
            traj_from_init = _finish_plan(
                nodes_from_init[-1], hyperparameters.max_velocity
            )
            traj_from_goal = _finish_plan(
                nodes_from_goal[-1], hyperparameters.max_velocity
            )
            traj_to_goal = traj_from_goal.reverse()
            traj = concatenate_robot_conf_trajectories([traj_from_init, traj_to_goal])
            # Smooth the trajectory before returning.
            return find_trajectory_shortcuts(traj, rng, mpp, hyperparameters)

    # No path found, fail.
    return None


def _build_birrt(
    mpp: MotionPlanningProblem[RobotConf],
    hyperparameters: RRTHyperparameters,
) -> Tuple[List[_RRTNode[RobotConf]], List[_RRTNode[RobotConf]]]:
    nodes_from_init = [_RRTNode(mpp.initial_configuration)]
    nodes_from_goal = [_RRTNode(mpp.goal_configuration)]
    for _ in range(hyperparameters.num_iters):
        target = mpp.configuration_space.sample()
        # Extend each tree in the direction of the target until collision.
        connection_achieved = True
        for nodes in [nodes_from_init, nodes_from_goal]:
            node = _get_closest_node(nodes, target)
            extension = RobotConfSegment(node.conf, target)
            for waypoint in iter_traj_with_max_distance(
                extension,
                hyperparameters.collision_check_max_distance,
                include_start=False,
            ):
                if mpp.has_collision(waypoint):
                    connection_achieved = False
                    break
                node = _RRTNode(waypoint, parent=node)
                nodes.append(node)
        if connection_achieved:
            return nodes_from_init, nodes_from_goal
    return nodes_from_init, nodes_from_goal
