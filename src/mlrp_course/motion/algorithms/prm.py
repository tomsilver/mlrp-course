"""Motion planning with probabilistic roadmaps."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Generic, Iterator, List, Set, Tuple

from mlrp_course.classical.algorithms.search import (
    SearchFailure,
    run_uniform_cost_search,
)
from mlrp_course.classical.classical_problem import ClassicalPlanningProblem
from mlrp_course.motion.algorithms.direct_path import run_direct_path_motion_planning
from mlrp_course.motion.motion_planning_problem import MotionPlanningProblem, RobotConf
from mlrp_course.motion.utils import (
    MotionPlanningHyperparameters,
    RobotConfSegment,
    RobotConfTraj,
    get_robot_conf_distance,
    iter_traj_with_max_distance,
    robot_conf_sequence_to_trajectory,
)
from mlrp_course.structs import Image


@dataclass(frozen=True)
class PRMHyperparameters(MotionPlanningHyperparameters):
    """Hyperparameters for PRM."""

    num_iters: int = 100
    neighbor_distance_thresh: float = 10.0


# Give nodes unique IDs.
_NODE_ID_COUNT = itertools.count()


@dataclass
class _PRMNode(Generic[RobotConf]):
    """A node in a PRM."""

    conf: RobotConf
    neighbors: List[_PRMNode[RobotConf]]
    node_id: int = field(default_factory=lambda: next(_NODE_ID_COUNT))

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __lt__(self, other: Any) -> bool:
        assert isinstance(other, _PRMNode)
        return self.node_id < other.node_id


@dataclass
class _PRMGraph(Generic[RobotConf]):
    """A PRM."""

    start_node: _PRMNode[RobotConf]
    goal_node: _PRMNode[RobotConf]
    nodes: List[_PRMNode[RobotConf]]


def run_prm(
    mpp: MotionPlanningProblem[RobotConf],
    hyperparameters: PRMHyperparameters | None = None,
) -> RobotConfTraj[RobotConf] | None:
    """Create a PRM to find a collision-free path from start to goal.

    If none is found, returns None.
    """
    if hyperparameters is None:
        hyperparameters = PRMHyperparameters()

    # Check if the problem is obviously impossible.
    if mpp.has_collision(mpp.initial_configuration) or mpp.has_collision(
        mpp.goal_configuration
    ):
        return None

    # Try to just take a direct path from start to goal.
    direct_path = run_direct_path_motion_planning(mpp, hyperparameters)
    if direct_path is not None:
        return direct_path

    # Build the PRM.
    graph = _build_prm(mpp, hyperparameters)

    # Check if there is a path from start to goal.
    node_path = _find_node_path(graph)

    # No path found, fail.
    if node_path is None:
        return None

    # Convert the node path into a trajectory.
    conf_sequence = [node.conf for node in node_path]
    return robot_conf_sequence_to_trajectory(
        conf_sequence, hyperparameters.max_velocity
    )


def _build_prm(
    mpp: MotionPlanningProblem[RobotConf],
    hyperparameters: PRMHyperparameters,
) -> _PRMGraph[RobotConf]:
    start_node = _PRMNode(mpp.initial_configuration, [])
    goal_node = _PRMNode(mpp.goal_configuration, [])
    nodes = [start_node, goal_node]
    for _ in range(hyperparameters.num_iters):
        # Sample from the configuration space.
        new_conf = mpp.configuration_space.sample()
        # Check if feasible.
        if mpp.has_collision(new_conf):
            continue
        # Add to the graph.
        new_node = _PRMNode(new_conf, [])
        for node in nodes:
            dist = get_robot_conf_distance(node.conf, new_conf)
            if dist > hyperparameters.neighbor_distance_thresh:
                continue
            # Add edge if the path is clear.
            if not _path_has_collision(
                node.conf, new_conf, mpp, hyperparameters.collision_check_max_distance
            ):
                new_node.neighbors.append(node)
                node.neighbors.append(new_node)
        nodes.append(new_node)
    return _PRMGraph(start_node, goal_node, nodes)


def _path_has_collision(
    start: RobotConf,
    end: RobotConf,
    mpp: MotionPlanningProblem,
    collision_check_max_distance: float,
) -> bool:
    extension = RobotConfSegment(start, end)
    for waypoint in iter_traj_with_max_distance(
        extension,
        collision_check_max_distance,
        include_start=False,
        include_end=False,
    ):
        if mpp.has_collision(waypoint):
            return True
    return False


class _PRMGraphSearchProblem(
    ClassicalPlanningProblem[_PRMNode[RobotConf], _PRMNode[RobotConf]]
):

    def __init__(self, prm_graph: _PRMGraph[RobotConf]) -> None:
        self._prm_graph = prm_graph
        super().__init__()

    @property
    def state_space(self) -> Set[_PRMNode[RobotConf]]:
        return set(self._prm_graph.nodes)

    @property
    def action_space(self) -> Set[_PRMNode[RobotConf]]:
        # An action is just a target node.
        return set(self._prm_graph.nodes)

    @property
    def initial_state(self) -> _PRMNode[RobotConf]:
        return self._prm_graph.start_node

    def initiable(
        self, state: _PRMNode[RobotConf], action: _PRMNode[RobotConf]
    ) -> bool:
        return action in state.neighbors

    def get_cost(
        self,
        state: _PRMNode[RobotConf],
        action: _PRMNode[RobotConf],
        next_state: _PRMNode[RobotConf],
    ) -> float:
        assert action is next_state
        return get_robot_conf_distance(state.conf, next_state.conf)

    def get_next_state(
        self, state: _PRMNode[RobotConf], action: _PRMNode[RobotConf]
    ) -> _PRMNode[RobotConf]:
        return action

    def check_goal(self, state: _PRMNode[RobotConf]) -> bool:
        return state is self._prm_graph.goal_node

    def render_state(self, state: _PRMNode[RobotConf]) -> Image:
        raise NotImplementedError

    def get_successors(
        self, state: _PRMNode[RobotConf]
    ) -> Iterator[Tuple[_PRMNode[RobotConf], _PRMNode[RobotConf], float]]:
        for neighbor in state.neighbors:
            yield (neighbor, neighbor, self.get_cost(state, neighbor, neighbor))


def _find_node_path(graph: _PRMGraph[RobotConf]) -> List[_PRMNode[RobotConf]] | None:
    graph_search_problem = _PRMGraphSearchProblem(graph)
    try:
        states, _, _ = run_uniform_cost_search(graph_search_problem)
    except SearchFailure:
        return None
    return states
