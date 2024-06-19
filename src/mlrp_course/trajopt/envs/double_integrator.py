"""Extremely simple testing environment."""

from dataclasses import dataclass
from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces import Box
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tomsgeoms2d.structs import Rectangle

from mlrp_course.structs import Hyperparameters, Image
from mlrp_course.trajopt.algorithms.drake_solver import DrakeProblem
from mlrp_course.trajopt.trajopt_problem import (
    TrajOptAction,
    TrajOptProblem,
    TrajOptState,
    TrajOptTraj,
)
from mlrp_course.utils import fig2data


@dataclass(frozen=True)
class DoubleIntegratorHyperparameters(Hyperparameters):
    """Hyperparameters for DoubleIntegratorProblem."""

    horizon: int = 25
    dt: float = 0.1
    x_cost_weight: float = 1.0
    x_dot_cost_weight: float = 0.1
    torque_cost_weight: float = 0.01
    torque_lb: float = -np.inf
    torque_ub: float = np.inf


class DoubleIntegratorProblem(TrajOptProblem):
    """Extremely simple testing environment."""

    def __init__(self, config: DoubleIntegratorHyperparameters | None = None) -> None:
        self._config = config or DoubleIntegratorHyperparameters()
        super().__init__()

    @property
    def horizon(self) -> int:
        return self._config.horizon

    @cached_property
    def state_space(self) -> Box:
        # x and x_dot.
        return Box(
            low=np.array([-np.inf, -np.inf]),
            high=np.array([np.inf, np.inf]),
        )

    @cached_property
    def action_space(self) -> Box:
        # torque.
        return Box(
            low=np.array([self._config.torque_lb]),
            high=np.array([self._config.torque_ub]),
        )

    @property
    def initial_state(self) -> TrajOptState:
        return np.array([-1.0, 0.0], dtype=np.float32)

    def get_next_state(
        self, state: TrajOptState, action: TrajOptAction
    ) -> TrajOptState:
        return self._get_next_state(
            state,
            action,
            self._config.dt,
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

        return np.array([next_x, next_x_dot], dtype=state.dtype)

    def get_traj_cost(self, traj: TrajOptTraj) -> float:
        xs, x_dots = traj.states.T
        return self._get_traj_cost(
            xs,
            x_dots,
            traj.actions,
            self._config.x_cost_weight,
            self._config.x_dot_cost_weight,
            self._config.torque_cost_weight,
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
        cost = (
            x_cost_weight * x_cost
            + x_dot_cost_weight * x_dot_cost
            + torque_cost_weight * action_cost
        )
        return cost

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


class JaxDoubleIntegratorProblem(DoubleIntegratorProblem):
    """Jax version of DoubleIntegratorProblem."""

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
        return DoubleIntegratorProblem._get_traj_cost(
            xs, x_dots, actions, x_cost_weight, x_dot_cost_weight, torque_cost_weight
        )


class DrakeDoubleIntegratorProblem(DoubleIntegratorProblem, DrakeProblem):
    """Drake version of DoubleIntegratorProblem."""
