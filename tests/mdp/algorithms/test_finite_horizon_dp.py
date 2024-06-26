"""Tests for finite_horizon_dp.py."""

import numpy as np

from mlrp_course.mdp.algorithms.finite_horizon_dp import (
    FiniteHorizonDPHyperparameters,
    finite_horizon_dp,
)
from mlrp_course.mdp.envs.marshmellow_mdp import MarshmallowMDP


def test_finite_horizon_dp():
    """Tests for finite_horizon_dp.py."""
    mdp = MarshmallowMDP()
    V = finite_horizon_dp(mdp, FiniteHorizonDPHyperparameters())
    assert all(np.isclose(v, 0.0) for v in V[4].values())
    assert V[0][(0, True)] > V[0][(1, True)]
    assert V[0][(0, True)] > V[0][(0, False)]
