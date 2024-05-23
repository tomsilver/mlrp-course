"""Algorithms for policy evaluation."""

from typing import Callable, Dict

from sympy import Eq, Symbol, linsolve

from mlrp_course.mdp.discrete_mdp import DiscreteAction, DiscreteMDP, DiscreteState


def evaluate_policy_linear_system(
    pi: Callable[[DiscreteState], DiscreteAction], mdp: DiscreteMDP
) -> Dict[DiscreteState, float]:
    """Computes a value function by solving a system of linear equations."""
    # Get states, actions, Tr, and R
    states = mdp.state_space
    gamma = mdp.temporal_discount_factor

    def Tr(s: DiscreteState, a: DiscreteAction, ns: DiscreteState) -> float:
        """Shorthand for transition probability."""
        return mdp.get_transition_distribution(s, a).get(ns, 0.0)

    R = mdp.get_reward

    # Create symbolic variables for values.
    V = {s: Symbol(f"s{s}") for s in states}

    # Create equations.
    equations = []
    for s, v_s in V.items():
        # Constrain terminal states to be 0.
        if mdp.state_is_terminal(s):
            rhs = 0
        # Main equation.
        else:
            rhs = sum(
                Tr(s, pi(s), ns) * (R(s, pi(s), ns) + gamma * V[ns]) for ns in states
            )
        equation = Eq(v_s, rhs)
        equations.append(equation)

    # Solve equations.
    solutions = linsolve(equations, [V[s] for s in states])
    solutions = list(solutions)
    assert len(solutions) == 1
    values = solutions[0]

    # Construct value function.
    return dict(zip(states, values))
