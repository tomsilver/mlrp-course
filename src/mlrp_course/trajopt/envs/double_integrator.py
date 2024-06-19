"""Extremely simple testing environment."""

from functools import cached_property
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces import Box
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tomsgeoms2d.structs import Rectangle

from mlrp_course.structs import Image
from mlrp_course.trajopt.trajopt_problem import (
    TrajOptAction,
    TrajOptState,
    TrajOptTraj,
    UnconstrainedTrajOptProblem,
)
from mlrp_course.utils import fig2data


class UnconstrainedDoubleIntegratorProblem(UnconstrainedTrajOptProblem):
    """Extremely simple testing environment."""

    _dt: ClassVar[float] = 0.1
    _x_cost_weight: ClassVar[float] = 1.0
    _x_dot_cost_weight: ClassVar[float] = 0.1
    _torque_cost_weight: ClassVar[float] = 0.01

    def __init__(self, horizon: int = 25) -> None:
        self._horizon = horizon
        super().__init__()

    @property
    def horizon(self) -> int:
        return self._horizon

    @cached_property
    def state_space(self) -> Box:
        # x and x_dot.
        return Box(
            low=np.array([-np.inf, np.inf]),
            high=np.array([-np.inf, np.inf]),
        )

    @cached_property
    def action_space(self) -> Box:
        # torque.
        return Box(low=np.array([-np.inf]), high=np.array([np.inf]))

    @property
    def initial_state(self) -> TrajOptState:
        return np.array([-1.0, 0.0], dtype=np.float32)

    def get_next_state(
        self, state: TrajOptState, action: TrajOptAction
    ) -> TrajOptState:
        return self._get_next_state(
            state,
            action,
            self._dt,
        )

    @staticmethod
    def _get_next_state(
        state: TrajOptState,
        action: TrajOptAction,
        dt: float,
    ) -> TrajOptState:

        x, x_dot = state
        u = action[0]

        next_x_dot = x_dot + u * dt
        next_x = x + next_x_dot * dt

        return np.array([next_x, next_x_dot], dtype=np.float32)

    def get_traj_cost(self, traj: TrajOptTraj) -> float:
        xs, x_dots = traj.states.T
        return self._get_traj_cost(
            xs,
            x_dots,
            traj.actions,
            self._x_cost_weight,
            self._x_dot_cost_weight,
            self._torque_cost_weight,
        )

    @staticmethod
    def _get_traj_cost(
        xs: NDArray[np.float32],
        x_dots: NDArray[np.float32],
        actions: NDArray[np.float32],
        x_cost_weight: float,
        x_dot_cost_weight: float,
        torque_cost_weight: float,
    ) -> float:
        x_cost = (xs**2).sum()
        x_dot_cost = (x_dots**2).sum()
        action_cost = (actions**2).sum()
        return (
            x_cost_weight * x_cost
            + x_dot_cost_weight * x_dot_cost * torque_cost_weight * action_cost
        )

    def render_state(self, state: TrajOptState) -> Image:
        x, _ = state
        rect = Rectangle.from_center(x, 0, 1, 1, 0.0)
        figsize = (5, 5)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        rect.plot(ax, fc="gray", ec="black")
        plt.tight_layout()
        img = fig2data(fig)
        plt.close()
        return img


class JaxUnconstrainedDoubleIntegratorProblem(UnconstrainedDoubleIntegratorProblem):
    """Jax version of UnconstrainedDoubleIntegratorProblem."""

    @property
    def initial_state(self) -> TrajOptState:
        return jnp.array(super().initial_state, dtype=jnp.float32)

    @staticmethod
    @jax.jit
    def _get_next_state(
        state: TrajOptState,
        action: TrajOptAction,
        dt: float,
    ) -> TrajOptState:

        x, x_dot = state
        u = action[0]

        next_x_dot = x_dot + u * dt
        next_x = x + next_x_dot * dt

        return jnp.array([next_x, next_x_dot], dtype=jnp.float32)

    @staticmethod
    @jax.jit
    def _get_traj_cost(
        xs: NDArray[jnp.float32],
        x_dots: NDArray[jnp.float32],
        actions: NDArray[jnp.float32],
        x_cost_weight: float,
        x_dot_cost_weight: float,
        torque_cost_weight: float,
    ) -> float:
        return UnconstrainedDoubleIntegratorProblem._get_traj_cost(
            xs, x_dots, actions, x_cost_weight, x_dot_cost_weight, torque_cost_weight
        )
