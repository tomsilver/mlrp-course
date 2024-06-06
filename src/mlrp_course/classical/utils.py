"""Utilities for classical planning."""

from typing import Set

from relational_structs import LiftedOperator, PDDLDomain


def delete_relax_pddl_domain(pddl_domain: PDDLDomain) -> PDDLDomain:
    """Delete-relax a PDDL domain."""
    relaxed_operators: Set[LiftedOperator] = set()
    for operator in pddl_domain.operators:
        relaxed_operator = LiftedOperator(
            operator.name,
            operator.parameters,
            operator.preconditions,
            operator.add_effects,
            set(),
        )
        relaxed_operators.add(relaxed_operator)
    return PDDLDomain(
        pddl_domain.name, relaxed_operators, pddl_domain.predicates, pddl_domain.types
    )
