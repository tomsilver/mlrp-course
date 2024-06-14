"""Motion planning with probabilistic roadmaps."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Generic, Iterator, List, Set, Tuple

import numpy as np

from mlrp_course.classical.algorithms.search import (
    SearchFailure,
    run_uniform_cost_search,
)
from mlrp_course.classical.classical_problem import ClassicalPlanningProblem
from mlrp_course.motion.algorithms.direct_path import run_direct_path_motion_planning
from mlrp_course.motion.motion_planning_problem import MotionPlanningProblem, RobotConf
from mlrp_course.motion.utils import (
    MotionPlanningHyperparameters,
    find_trajectory_shortcuts,
    iter_traj_with_max_distance,
)
from mlrp_course.structs import Image
from mlrp_course.utils import (
    Trajectory,
    TrajectorySegment,
    get_trajectory_point_distance,
    point_sequence_to_trajectory,
)


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

    nodes: List[_PRMNode[RobotConf]] = field(default_factory=list)


def run_prm(
    mpp: MotionPlanningProblem[RobotConf],
    rng: np.random.Generator,
    hyperparameters: PRMHyperparameters | None = None,
) -> Trajectory[RobotConf] | None:
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

    # Query the PRM.
    traj = _query_prm(
        mpp.initial_configuration, mpp.goal_configuration, graph, mpp, hyperparameters
    )
    if traj is None:
        return traj

    # Smooth the trajectory before returning.
    return find_trajectory_shortcuts(traj, rng, mpp, hyperparameters)


def _build_prm(
    mpp: MotionPlanningProblem[RobotConf],
    hyperparameters: PRMHyperparameters,
) -> _PRMGraph[RobotConf]:
    """Initialize a PRM, ignoring for now the initial state and goal."""
    graph: _PRMGraph[RobotConf] = _PRMGraph()
    for _ in range(hyperparameters.num_iters):
        # Sample from the configuration space.
        new_conf = mpp.configuration_space.sample()
        # Check if feasible.
        if mpp.has_collision(new_conf):
            continue
        # Add to the graph.
        _update_prm(new_conf, graph, mpp, hyperparameters)
    return graph


def _update_prm(
    conf: RobotConf,
    graph: _PRMGraph[RobotConf],
    mpp: MotionPlanningProblem[RobotConf],
    hyperparameters: PRMHyperparameters,
) -> _PRMNode[RobotConf]:
    new_node = _PRMNode(conf, [])
    for node in graph.nodes:
        dist = get_trajectory_point_distance(node.conf, conf)
        if dist > hyperparameters.neighbor_distance_thresh:
            continue
        # Add edge if the path is clear.
        if not _path_has_collision(
            node.conf, conf, mpp, hyperparameters.collision_check_max_distance
        ):
            new_node.neighbors.append(node)
            node.neighbors.append(new_node)
    graph.nodes.append(new_node)
    return new_node


def _query_prm(
    init_conf: RobotConf,
    goal_conf: RobotConf,
    graph: _PRMGraph[RobotConf],
    mpp: MotionPlanningProblem[RobotConf],
    hyperparameters: PRMHyperparameters,
) -> Trajectory[RobotConf] | None:
    init_node = _update_prm(init_conf, graph, mpp, hyperparameters)
    goal_node = _update_prm(goal_conf, graph, mpp, hyperparameters)
    # Check if there is a path from start to goal.
    node_path = _find_node_path(graph, init_node, goal_node)
    # No path found, fail.
    if node_path is None:
        return None
    # Convert the node path into a trajectory.
    conf_sequence = [node.conf for node in node_path]
    return point_sequence_to_trajectory(conf_sequence, hyperparameters.max_velocity)


def _path_has_collision(
    start: RobotConf,
    end: RobotConf,
    mpp: MotionPlanningProblem,
    collision_check_max_distance: float,
) -> bool:
    extension = TrajectorySegment(start, end)
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

    def __init__(
        self,
        prm_graph: _PRMGraph[RobotConf],
        init_node: _PRMNode[RobotConf],
        goal_node: _PRMNode[RobotConf],
    ) -> None:
        assert init_node in prm_graph.nodes
        assert goal_node in prm_graph.nodes
        self._prm_graph = prm_graph
        self._init_node = init_node
        self._goal_node = goal_node
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
        return self._init_node

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
        return get_trajectory_point_distance(state.conf, next_state.conf)

    def get_next_state(
        self, state: _PRMNode[RobotConf], action: _PRMNode[RobotConf]
    ) -> _PRMNode[RobotConf]:
        return action

    def check_goal(self, state: _PRMNode[RobotConf]) -> bool:
        return state is self._goal_node

    def render_state(self, state: _PRMNode[RobotConf]) -> Image:
        raise NotImplementedError

    def get_successors(
        self, state: _PRMNode[RobotConf]
    ) -> Iterator[Tuple[_PRMNode[RobotConf], _PRMNode[RobotConf], float]]:
        for neighbor in state.neighbors:
            yield (neighbor, neighbor, self.get_cost(state, neighbor, neighbor))


def _find_node_path(
    graph: _PRMGraph[RobotConf],
    init_node: _PRMNode[RobotConf],
    goal_node: _PRMNode[RobotConf],
) -> List[_PRMNode[RobotConf]] | None:
    graph_search_problem = _PRMGraphSearchProblem(graph, init_node, goal_node)
    try:
        states, _, _ = run_uniform_cost_search(graph_search_problem)
    except SearchFailure:
        return None
    return states
