"""Classic pendulum swing-up trajopt problem.

Reference: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/pendulum.py  # pylint: disable=line-too-long
"""

from typing import ClassVar

import numpy as np

from mlrp_course.structs import Image
from mlrp_course.trajopt.trajopt_problem import (
    TrajOptAction,
    TrajOptState,
    TrajOptTraj,
    UnconstrainedTrajOptProblem,
)
from mlrp_course.utils import wrap_angle


class PendulumTrajOptProblem(UnconstrainedTrajOptProblem):
    """Classic pendulum swing-up trajopt problem."""

    _gravity: ClassVar[float] = 10
    _mass: ClassVar[float] = 1.0
    _length: ClassVar[float] = 1.0
    _dt: ClassVar[float] = 0.05
    _max_torque: ClassVar[float] = 2.0
    _max_speed: ClassVar[float] = 8.0
    _theta_cost_weight: ClassVar[float] = 1.0
    _theta_dot_cost_weight: ClassVar[float] = 0.1
    _torque_cost_weight: ClassVar[float] = 0.001

    @property
    def horizon(self) -> int:
        return 200

    @property
    def state_dim(self) -> int:
        return 2  # theta and theta_dot

    @property
    def action_dim(self) -> int:
        return 1  # torque

    @property
    def initial_state(self) -> TrajOptState:
        return np.array([np.pi, 1.0])  # down and slightly swinging

    def get_next_state(
        self, state: TrajOptState, action: TrajOptAction
    ) -> TrajOptState:
        theta, theta_dot = state

        g = self._gravity
        m = self._mass
        l = self._length
        dt = self._dt

        u = np.clip(action, -self._max_torque, self._max_torque)

        next_theta_dot = (
            theta_dot + (3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l**2) * u) * dt
        )
        next_theta_dot = np.clip(next_theta_dot, -self._max_speed, self._max_speed)
        next_theta = theta + next_theta_dot * dt

        return np.array([next_theta, next_theta_dot])

    def get_traj_cost(self, traj: TrajOptTraj) -> float:
        # Get states costs.
        thetas, theta_dots = traj.states.T
        norm_thetas = np.vectorize(wrap_angle)(thetas)
        theta_cost = sum(norm_thetas**2)
        theta_dot_cost = sum(theta_dots**2)
        # Get action costs.
        torque_cost = sum(traj.actions**2)
        # Combine.
        return (
            self._theta_cost_weight * theta_cost
            + self._theta_dot_cost_weight * theta_dot_cost
            + self._torque_cost_weight * torque_cost
        )

    def render_state(self, state: TrajOptState) -> Image:
        raise NotImplementedError
