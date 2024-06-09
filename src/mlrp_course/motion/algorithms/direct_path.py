"""Simplest possible motion planning: try a direct path from start to goal."""

from mlrp_course.motion.motion_planning_problem import MotionPlanningProblem, RobotConf
from mlrp_course.motion.utils import (
    MotionPlanningHyperparameters,
    RobotConfSegment,
    RobotConfTraj,
    iter_traj_with_max_distance,
)


def run_direct_path_motion_planning(
    mpp: MotionPlanningProblem[RobotConf],
    hyperparameters: MotionPlanningHyperparameters | None = None,
) -> RobotConfTraj[RobotConf] | None:
    """Attempt to construct a trajectory directly from start to goal.

    If none is found, returns None.
    """
    if hyperparameters is None:
        hyperparameters = MotionPlanningHyperparameters()
    traj = RobotConfSegment.from_max_velocity(
        mpp.initial_configuration, mpp.goal_configuration, hyperparameters.max_velocity
    )
    for waypoint in iter_traj_with_max_distance(
        traj,
        hyperparameters.collision_check_max_distance,
    ):
        if mpp.has_collision(waypoint):
            return None
    return traj
