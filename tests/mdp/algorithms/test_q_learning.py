"""Tests for q_learning.py."""

from mlrp_course.mdp.algorithms.q_learning import (
    QLearningAgent,
)
from mlrp_course.mdp.envs.chase_mdp import ChaseState, StaticBunnyChaseMDP
from mlrp_course.mdp.utils import DiscreteMDPGymEnv
from mlrp_course.utils import run_episodes


def test_value_iteration():
    """Tests for value_iteration.py."""
    mdp = StaticBunnyChaseMDP()
    sample_initial_state = lambda _: ChaseState((0, 0), ((1, 2),))
    env = DiscreteMDPGymEnv(mdp, sample_initial_state)
    seed = 123
    env.reset(seed=seed)
    agent = QLearningAgent(mdp.action_space, mdp.temporal_discount_factor, seed)
    agent.train()
    run_episodes(agent, env, num_episodes=10000, max_episode_length=10)
    Q = agent._Q_dict  # pylint: disable=protected-access
    state = ChaseState((0, 0), ((1, 2),))
    assert Q[state]["right"] > Q[state]["left"]
    assert Q[state]["down"] > Q[state]["up"]
