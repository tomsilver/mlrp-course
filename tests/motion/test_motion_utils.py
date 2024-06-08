"""Tests for utils.py in motion planning."""

import numpy as np
from spatialmath import SE2

from mlrp_course.motion.utils import ConcatRobotConfTraj, RobotConfSegment


def test_motion_utils():
    """Tests for utils.py in motion planning."""
    waypoints = [
        SE2(0, 0, 0),
        SE2(1, 0, np.pi / 2),
        SE2(1, 1, np.pi),
        SE2(0, 1, -np.pi / 2),
        SE2(0, 0, 0),
    ]
    durations = [1.0, 1.0, 1.0, 5.0]
    segments = [
        RobotConfSegment(s, e, t)
        for s, e, t in zip(waypoints[:-1], waypoints[1:], durations, strict=True)
    ]
    traj = ConcatRobotConfTraj(segments)

    assert np.isclose(traj(0).x, 0)
    assert np.isclose(traj(0).y, 0)
    assert np.isclose(traj(0).theta(), 0)
    assert np.isclose(traj(1).x, 1)
    assert np.isclose(traj(1).y, 0)
    assert np.isclose(traj(1).theta(), np.pi / 2)
    assert np.isclose(traj(8).x, 0)
    assert np.isclose(traj(8).y, 0)
    assert np.isclose(traj(8).theta(), 0)

    # Uncomment to visualize.
    # from tomsgeoms2d.structs import Rectangle
    # import imageio.v2 as iio
    # import matplotlib.pyplot as plt
    # from mlrp_course.utils import fig2data
    # imgs = []
    # for t in np.linspace(0, traj.duration, 100):
    #     pose = traj(t)
    #     rect = Rectangle.from_center(pose.x, pose.y, 1.0, 1.0, pose.theta())
    #     fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=150)
    #     rect.plot(ax, fc="green", ec="black")
    #     ax.set_xlim(-0.5, 1.5)
    #     ax.set_ylim(-0.5, 1.5)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     plt.tight_layout()
    #     img = fig2data(fig)
    #     plt.close()
    #     imgs.append(img)
    # iio.mimsave("test_traj.mp4", imgs, fps=10)
