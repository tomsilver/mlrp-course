"""Simplest possible motion planning: try a direct path from start to goal."""

from mlrp_course.motion.motion_planning_problem import MotionPlanningProblem, RobotConf
from mlrp_course.motion.utils import (
    MotionPlanningHyperparameters,
    try_direct_path_motion_plan,
)
from mlrp_course.utils import Trajectory


def run_direct_path_motion_planning(
    mpp: MotionPlanningProblem[RobotConf],
    hyperparameters: MotionPlanningHyperparameters | None = None,
) -> Trajectory[RobotConf] | None:
    """Attempt to construct a trajectory directly from start to goal.

    If none is found, returns None.
    """
    return try_direct_path_motion_plan(
        mpp.initial_configuration,
        mpp.goal_configuration,
        mpp.has_collision,
        hyperparameters,
    )
