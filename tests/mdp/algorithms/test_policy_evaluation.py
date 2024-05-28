"""Tests for policy_evaluation.py."""

from mlrp_course.mdp.algorithms.policy_evaluation import evaluate_policy_linear_system
from mlrp_course.mdp.envs.zits_mdp import ZitsMDP


def test_policy_evaluation():
    """Tests for policy_evaluation.py."""
    mdp = ZitsMDP()

    def _better_policy(s):
        if s == 4:
            return "apply"
        return "sleep"

    def _worse_policy(s):
        del s  # unused
        return "sleep"

    worse_V = evaluate_policy_linear_system(_worse_policy, mdp)
    better_V = evaluate_policy_linear_system(_better_policy, mdp)

    assert worse_V[4] < better_V[4]
