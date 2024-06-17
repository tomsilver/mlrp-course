"""Classic pendulum swing-up trajopt problem.

Reference: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/pendulum.py  # pylint: disable=line-too-long
"""

from functools import cached_property
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces import Box
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tomsgeoms2d.structs import Circle, Rectangle

from mlrp_course.structs import Image
from mlrp_course.trajopt.trajopt_problem import (
    TrajOptAction,
    TrajOptState,
    TrajOptTraj,
    UnconstrainedTrajOptProblem,
)
from mlrp_course.utils import fig2data, wrap_angle


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

    def __init__(self, seed: int, horizon: int = 200) -> None:
        self._seed = seed
        self._horizon = horizon
        self.action_space.seed(seed)
        super().__init__()

    @property
    def horizon(self) -> int:
        return self._horizon

    @cached_property
    def state_space(self) -> Box:
        # theta and theta_dot.
        return Box(
            low=np.array([-np.pi, -self._max_speed]),
            high=np.array([np.pi, self._max_speed]),
        )

    @cached_property
    def action_space(self) -> Box:
        # torque.
        return Box(low=np.array([-self._max_torque]), high=np.array([self._max_torque]))

    @property
    def initial_state(self) -> TrajOptState:
        return jnp.array([np.pi, 1.0], dtype=jnp.float32)  # down and swinging

    def get_next_state(
        self, state: TrajOptState, action: TrajOptAction
    ) -> TrajOptState:
        return self._get_next_state(
            state,
            action,
            self._gravity,
            self._mass,
            self._length,
            self._dt,
            self._max_torque,
            self._max_speed,
        )

    @staticmethod
    @jax.jit
    def _get_next_state(
        state: TrajOptState,
        action: TrajOptAction,
        g: float,
        m: float,
        l: float,
        dt: float,
        max_torque: float,
        max_speed: float,
    ) -> TrajOptState:

        theta, theta_dot = state

        u = jnp.clip(action[0], -max_torque, max_torque)

        next_theta_dot = (
            theta_dot + (3 * g / (2 * l) * jnp.sin(theta) + 3.0 / (m * l**2) * u) * dt
        )
        next_theta_dot = jnp.clip(next_theta_dot, -max_speed, max_speed)
        next_theta = theta + next_theta_dot * dt
        next_theta = wrap_angle(next_theta)

        return jnp.array([next_theta, next_theta_dot], dtype=jnp.float32)

    def get_traj_cost(self, traj: TrajOptTraj) -> float:
        thetas, theta_dots = traj.states.T
        return self._get_traj_cost(
            thetas,
            theta_dots,
            traj.actions,
            self._theta_cost_weight,
            self._theta_dot_cost_weight,
            self._torque_cost_weight,
        )

    @staticmethod
    @jax.jit
    def _get_traj_cost(
        thetas: NDArray[jnp.float32],
        theta_dots: NDArray[jnp.float32],
        actions: NDArray[jnp.float32],
        theta_cost_weight: float,
        theta_dot_cost_weight: float,
        torque_cost_weight: float,
    ) -> float:
        # Get states costs.
        norm_thetas = jnp.vectorize(wrap_angle)(thetas)
        theta_cost = (norm_thetas**2).sum()
        theta_dot_cost = (theta_dots**2).sum()
        # Get action costs.
        torque_cost = (actions**2).sum()
        # Combine.
        cost = (
            theta_cost_weight * theta_cost
            + theta_dot_cost_weight * theta_dot_cost
            + torque_cost_weight * torque_cost
        )
        return cost

    def render_state(self, state: TrajOptState) -> Image:
        theta, _ = state
        rot = wrap_angle(theta + np.pi / 2)
        width = self._length / 8
        rect = Rectangle(0, 0, self._length, width, 0.0)
        rect = rect.rotate_about_point(0, width / 2, rot)
        circ = Circle(0, width / 2, width / 8)
        figsize = (5, 5)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        pad_scale = 1.25
        ax.set_xlim(-pad_scale * self._length, pad_scale * self._length)
        ax.set_ylim(-pad_scale * self._length, pad_scale * self._length)
        ax.set_xticks([])
        ax.set_yticks([])
        rect.plot(ax, fc="gray", ec="black")
        circ.plot(ax, fc="black", ec="black")
        plt.tight_layout()
        img = fig2data(fig)
        plt.close()
        return img
