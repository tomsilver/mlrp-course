"""Tests for utils.py."""

from mlrp_course.mdp.discrete_mdp import DiscreteState
from mlrp_course.mdp.envs.marshmellow_mdp import MarshmallowMDP
from mlrp_course.utils import DiscreteMDPGymEnv


def test_discrete_mdp_gym_env():
    """Tests for DiscreteMDPGymEnv()."""
    mdp = MarshmallowMDP()

    def _sample_initial_state(_: int | None) -> DiscreteState:
        return (0, True)

    env = DiscreteMDPGymEnv(mdp, _sample_initial_state)

    obs, _ = env.reset(seed=123)
    assert obs == (0, True)

    for _ in range(3):
        obs, _, terminated, truncated, _ = env.step("wait")
        assert obs in mdp.state_space
        assert not terminated
        assert not truncated
    _, _, _, truncated, _ = env.step("wait")
    assert truncated
