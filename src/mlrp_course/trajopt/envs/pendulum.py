"""Classic pendulum swing-up trajopt problem.

Modified from
https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/pendulum.py
# pylint: disable=line-too-long
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
from mlrp_course.trajopt.algorithms.drake_solver import DrakeProblem
from mlrp_course.trajopt.trajopt_problem import (
    TrajOptAction,
    TrajOptState,
    TrajOptTraj,
    UnconstrainedTrajOptProblem,
)
from mlrp_course.utils import fig2data, wrap_angle


class UnconstrainedPendulumTrajOptProblem(UnconstrainedTrajOptProblem):
    """Classic pendulum swing-up trajopt problem."""

    _gravity: ClassVar[float] = 10
    _mass: ClassVar[float] = 1.0
    _length: ClassVar[float] = 1.0
    _dt: ClassVar[float] = 0.05
    _theta_cost_weight: ClassVar[float] = 1.0
    _theta_dot_cost_weight: ClassVar[float] = 0.0
    _torque_cost_weight: ClassVar[float] = 0.0

    def __init__(self, horizon: int = 200) -> None:
        self._horizon = horizon
        super().__init__()

    @property
    def horizon(self) -> int:
        return self._horizon

    @cached_property
    def state_space(self) -> Box:
        # theta and theta_dot.
        return Box(
            low=np.array([-np.pi, -np.inf]),
            high=np.array([np.pi, np.inf]),
        )

    @cached_property
    def action_space(self) -> Box:
        # torque.
        return Box(low=np.array([-np.inf]), high=np.array([np.inf]))

    @property
    def initial_state(self) -> TrajOptState:
        return np.array([np.pi, 1.0], dtype=np.float32)  # down and swinging

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
        )

    @staticmethod
    def _get_next_state(
        state: TrajOptState,
        action: TrajOptAction,
        g: float,
        m: float,
        l: float,
        dt: float,
    ) -> TrajOptState:

        theta, theta_dot = state

        u = action[0]

        next_theta_dot = (
            theta_dot + (3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l**2) * u) * dt
        )
        next_theta = theta + next_theta_dot * dt
        next_theta = wrap_angle(next_theta)

        return np.array([next_theta, next_theta_dot], dtype=state.dtype)

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
    def _get_traj_cost(
        thetas: NDArray[np.float32],
        theta_dots: NDArray[np.float32],
        actions: NDArray[np.float32],
        theta_cost_weight: float,
        theta_dot_cost_weight: float,
        torque_cost_weight: float,
    ) -> float:
        # Get states costs.
        theta_cost = (thetas**2).sum()
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


class JaxUnconstrainedPendulumTrajOptProblem(UnconstrainedPendulumTrajOptProblem):
    """Jax version of the unconstrained pendulum trajopt problem."""

    @property
    def initial_state(self) -> TrajOptState:
        return jnp.array(super().initial_state, dtype=jnp.float32)

    @staticmethod
    @jax.jit
    def _get_next_state(
        state: TrajOptState,
        action: TrajOptAction,
        g: float,
        m: float,
        l: float,
        dt: float,
    ) -> TrajOptState:

        theta, theta_dot = state

        u = action[0]

        next_theta_dot = (
            theta_dot + (3 * g / (2 * l) * jnp.sin(theta) + 3.0 / (m * l**2) * u) * dt
        )
        next_theta = theta + next_theta_dot * dt
        next_theta = wrap_angle(next_theta)

        return jnp.array([next_theta, next_theta_dot], dtype=jnp.float32)

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
        return UnconstrainedPendulumTrajOptProblem._get_traj_cost(
            thetas,
            theta_dots,
            actions,
            theta_cost_weight,
            theta_dot_cost_weight,
            torque_cost_weight,
        )


class DrakeUnconstrainedPendulumTrajOptProblem(
    UnconstrainedPendulumTrajOptProblem, DrakeProblem
):
    """Drake version of UnconstrainedPendulumTrajOptProblem."""
