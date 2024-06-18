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


class DoubleIntegratorProblem(UnconstrainedTrajOptProblem):
    """Extremely simple testing environment."""

    _max_torque: ClassVar[float] = 1.0
    _dt: ClassVar[float] = 0.1
    _x_cost_weight: ClassVar[float] = 1.0
    _x_dot_cost_weight: ClassVar[float] = 0.1
    _torque_cost_weight: ClassVar[float] = 0.001

    def __init__(self, seed: int, horizon: int = 25) -> None:
        self._seed = seed
        self._horizon = horizon
        self.action_space.seed(seed)
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
        return Box(low=np.array([-self._max_torque]), high=np.array([self._max_torque]))

    @property
    def initial_state(self) -> TrajOptState:
        return jnp.array([-1.0, 0.0], dtype=jnp.float32)

    def get_next_state(
        self, state: TrajOptState, action: TrajOptAction
    ) -> TrajOptState:
        return self._get_next_state(
            state,
            action,
            self._dt,
            self._max_torque,
        )

    @staticmethod
    @jax.jit
    def _get_next_state(
        state: TrajOptState,
        action: TrajOptAction,
        dt: float,
        max_torque: float,
    ) -> TrajOptState:

        x, x_dot = state
        u = jnp.clip(action[0], -max_torque, max_torque)

        next_x_dot = x_dot + u * dt
        next_x = x + next_x_dot * dt

        return jnp.array([next_x, next_x_dot], dtype=jnp.float32)

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
    @jax.jit
    def _get_traj_cost(
        xs: NDArray[jnp.float32],
        x_dots: NDArray[jnp.float32],
        actions: NDArray[jnp.float32],
        x_cost_weight: float,
        x_dot_cost_weight: float,
        torque_cost_weight: float,
    ) -> float:
        final_x = xs[-1]
        final_xdot = x_dots[-1]
        torque_cost = (actions**2).sum()
        return (
            x_cost_weight * final_x**2
            + x_dot_cost_weight * final_xdot**2
            + torque_cost_weight * torque_cost
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
