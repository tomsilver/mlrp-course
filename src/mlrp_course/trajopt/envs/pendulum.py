"""Classic pendulum swing-up trajopt problem.

Modified from
https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/pendulum.py
# pylint: disable=line-too-long
"""

from dataclasses import dataclass
from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces import Box
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tomsgeoms2d.structs import Circle, Rectangle

from mlrp_course.structs import Hyperparameters, Image
from mlrp_course.trajopt.algorithms.drake_solver import DrakeProblem
from mlrp_course.trajopt.trajopt_problem import (
    TrajOptAction,
    TrajOptProblem,
    TrajOptState,
    TrajOptTraj,
)
from mlrp_course.utils import fig2data, wrap_angle


@dataclass(frozen=True)
class PendulumHyperparameters(Hyperparameters):
    """Hyperparameters for PendulumTrajOptProblem."""

    horizon: int = 200
    gravity: float = 10
    mass: float = 1.0
    length: float = 1.0
    dt: float = 0.05
    theta_cost_weight: float = 1.0
    theta_dot_cost_weight: float = 0.0
    torque_cost_weight: float = 0.0
    torque_lb: float = -np.inf
    torque_ub: float = np.inf


class PendulumTrajOptProblem(TrajOptProblem):
    """Classic pendulum swing-up trajopt problem."""

    def __init__(self, config: PendulumHyperparameters | None = None) -> None:
        self._config = config or PendulumHyperparameters()
        super().__init__()

    @property
    def horizon(self) -> int:
        return self._config.horizon

    @cached_property
    def state_space(self) -> Box:
        # theta and theta_dot.
        return Box(
            low=np.array([-np.inf, -np.inf]),
            high=np.array([np.inf, np.inf]),
            dtype=np.float64,
        )

    @cached_property
    def action_space(self) -> Box:
        # torque.
        return Box(
            low=np.array([self._config.torque_lb]),
            high=np.array([self._config.torque_ub]),
            dtype=np.float64,
        )

    @property
    def initial_state(self) -> TrajOptState:
        return np.array([np.pi, 1.0], dtype=np.float64)  # down and swinging

    def get_next_state(
        self, state: TrajOptState, action: TrajOptAction
    ) -> TrajOptState:
        return self._get_next_state(
            state,
            action,
            self._config.gravity,
            self._config.mass,
            self._config.length,
            self._config.dt,
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

        return np.array([next_theta, next_theta_dot], dtype=state.dtype)

    def get_traj_cost(self, traj: TrajOptTraj) -> float:
        thetas, theta_dots = traj.states.T
        return self._get_traj_cost(
            thetas,
            theta_dots,
            traj.actions,
            self._config.theta_cost_weight,
            self._config.theta_dot_cost_weight,
            self._config.torque_cost_weight,
        )

    @staticmethod
    def _get_traj_cost(
        thetas: NDArray[np.float64],
        theta_dots: NDArray[np.float64],
        actions: NDArray[np.float64],
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
        length = self._config.length
        width = length / 8
        rect = Rectangle(0, 0, length, width, 0.0)
        rect = rect.rotate_about_point(0, width / 2, rot)
        circ = Circle(0, width / 2, width / 8)
        figsize = (5, 5)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        pad_scale = 1.25
        ax.set_xlim(-pad_scale * length, pad_scale * length)
        ax.set_ylim(-pad_scale * length, pad_scale * length)
        ax.set_xticks([])
        ax.set_yticks([])
        rect.plot(ax, fc="gray", ec="black")
        circ.plot(ax, fc="black", ec="black")
        plt.tight_layout()
        img = fig2data(fig)
        plt.close()
        return img


class JaxPendulumTrajOptProblem(PendulumTrajOptProblem):
    """Jax version of the pendulum trajopt problem."""

    @property
    def initial_state(self) -> TrajOptState:
        return jnp.array(super().initial_state, dtype=jnp.float64)

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

        return jnp.array([next_theta, next_theta_dot], dtype=jnp.float64)

    @staticmethod
    @jax.jit
    def _get_traj_cost(
        thetas: NDArray[jnp.float64],
        theta_dots: NDArray[jnp.float64],
        actions: NDArray[jnp.float64],
        theta_cost_weight: float,
        theta_dot_cost_weight: float,
        torque_cost_weight: float,
    ) -> float:
        return PendulumTrajOptProblem._get_traj_cost(
            thetas,
            theta_dots,
            actions,
            theta_cost_weight,
            theta_dot_cost_weight,
            torque_cost_weight,
        )


class DrakePendulumTrajOptProblem(PendulumTrajOptProblem, DrakeProblem):
    """Drake version of PendulumTrajOptProblem."""
