"""Deterministic event-expanded min-cost-flow for backtracking routes.

Each litter event must send one unit of flow through exactly one route.  Route
nodes are event-specific, so actor keys are deliberately *not* capacity
constrained: one person may explain several litter events and one vehicle may
be linked to several people/events.

The implementation is dependency-free.  It uses a residual graph and
successive shortest augmenting paths (Bellman-Ford handles negative costs).
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple


# Milliscale quantization erased real sub-0.001 differences between actors and
# let deterministic route-id ordering decide the winner.  Microscale keeps the
# integer-flow objective while preserving those measured cost differences.
COST_SCALE = 1_000_000
SYNTHETIC_NULL_ROUTE_ID = "__null__"


@dataclass(frozen=True)
class RouteCandidate:
    """One complete attribution route for a litter event.

    ``person_key=None, vehicle_key=<key>`` represents a direct-vehicle route.
    Both keys being ``None`` represents the dustbin/NULL route.
    """

    route_id: str
    cost: float
    person_key: Optional[Hashable] = None
    vehicle_key: Optional[Hashable] = None
    route_type: str = "person_vehicle"
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @property
    def is_null(self) -> bool:
        return (
            self.route_type == "null"
            or (self.person_key is None and self.vehicle_key is None)
        )

    @property
    def is_direct_vehicle(self) -> bool:
        return self.person_key is None and self.vehicle_key is not None


@dataclass(frozen=True)
class Assignment:
    """Selected route and its confidence separation for one litter event."""

    litter_id: Hashable
    route: RouteCandidate
    total_cost: float
    scaled_cost: int
    margin_to_second: Optional[float]


@dataclass
class _Edge:
    to: int
    reverse_index: int
    capacity: int
    cost: int


class _ResidualGraph:
    def __init__(self, node_count: int) -> None:
        self.adjacency: List[List[_Edge]] = [[] for _ in range(node_count)]

    def add_edge(self, source: int, target: int, capacity: int, cost: int) -> _Edge:
        forward = _Edge(
            to=target,
            reverse_index=len(self.adjacency[target]),
            capacity=capacity,
            cost=cost,
        )
        backward = _Edge(
            to=source,
            reverse_index=len(self.adjacency[source]),
            capacity=0,
            cost=-cost,
        )
        self.adjacency[source].append(forward)
        self.adjacency[target].append(backward)
        return forward

    def shortest_augmenting_path(
        self, source: int, sink: int
    ) -> Tuple[List[Optional[Tuple[int, int]]], Optional[int]]:
        """Return a shortest residual path using deterministic Bellman-Ford."""

        node_count = len(self.adjacency)
        distances: List[Optional[int]] = [None] * node_count
        predecessor: List[Optional[Tuple[int, int]]] = [None] * node_count
        distances[source] = 0

        for _ in range(node_count - 1):
            changed = False
            for node in range(node_count):
                node_distance = distances[node]
                if node_distance is None:
                    continue
                for edge_index, edge in enumerate(self.adjacency[node]):
                    if edge.capacity <= 0:
                        continue
                    new_distance = node_distance + edge.cost
                    old_distance = distances[edge.to]
                    # Equal distances keep the first predecessor.  Nodes and
                    # edges are inserted in stable order, making ties stable.
                    if old_distance is None or new_distance < old_distance:
                        distances[edge.to] = new_distance
                        predecessor[edge.to] = (node, edge_index)
                        changed = True
            if not changed:
                break

        return predecessor, distances[sink]

    def send_one(
        self,
        source: int,
        sink: int,
        predecessor: Sequence[Optional[Tuple[int, int]]],
    ) -> None:
        node = sink
        while node != source:
            previous = predecessor[node]
            if previous is None:
                raise RuntimeError("incomplete residual path")
            previous_node, edge_index = previous
            edge = self.adjacency[previous_node][edge_index]
            edge.capacity -= 1
            reverse = self.adjacency[node][edge.reverse_index]
            reverse.capacity += 1
            node = previous_node


def _stable_value_key(value: Any) -> Tuple[str, str, str]:
    value_type = type(value)
    return (
        getattr(value_type, "__module__", ""),
        getattr(value_type, "__qualname__", value_type.__name__),
        repr(value),
    )


def _scaled_cost(cost: float) -> int:
    # Explicit half-away-from-zero rounding avoids Python's banker rounding at
    # exact .5 boundaries and keeps the integer network objective predictable.
    scaled = float(cost) * COST_SCALE
    if scaled >= 0:
        return int(math.floor(scaled + 0.5))
    return int(math.ceil(scaled - 0.5))


def _candidate_key(candidate: RouteCandidate) -> Tuple[Any, ...]:
    return (
        str(candidate.route_id),
        str(candidate.route_type),
        _stable_value_key(candidate.person_key),
        _stable_value_key(candidate.vehicle_key),
    )


def _prepare_candidates(
    candidates: Sequence[RouteCandidate],
    null_cost: float,
) -> List[RouteCandidate]:
    valid = [candidate for candidate in candidates if math.isfinite(candidate.cost)]
    if not any(candidate.is_null for candidate in valid):
        valid.append(
            RouteCandidate(
                route_id=SYNTHETIC_NULL_ROUTE_ID,
                cost=null_cost,
                route_type="null",
            )
        )
    # Cost is intentionally not part of this ordering.  The graph objective
    # chooses by cost; this stable order resolves equal integer costs.
    return sorted(valid, key=_candidate_key)


def solve_event_routes(
    routes_by_litter: Mapping[Hashable, Sequence[RouteCandidate]],
    *,
    null_cost: float = 10.0,
) -> Dict[Hashable, Assignment]:
    """Choose exactly one complete route for every litter event.

    Non-finite candidate costs are rejected.  A finite synthetic NULL route is
    added whenever an event has no valid explicit NULL candidate.

    Costs are optimized as integers at ``COST_SCALE=1_000_000``. Consequently,
    ``margin_to_second`` is also reported in the quantized objective units
    converted back to float.  A zero margin means an objective tie.
    """

    if not math.isfinite(null_cost):
        raise ValueError("null_cost must be finite")
    if not routes_by_litter:
        return {}

    litter_ids = sorted(routes_by_litter, key=_stable_value_key)
    prepared: Dict[Hashable, List[RouteCandidate]] = {
        litter_id: _prepare_candidates(routes_by_litter[litter_id], null_cost)
        for litter_id in litter_ids
    }

    # Layout: source, event nodes, event-specific route nodes, sink.
    source = 0
    event_nodes = {
        litter_id: index + 1 for index, litter_id in enumerate(litter_ids)
    }
    next_node = 1 + len(litter_ids)
    route_nodes: Dict[Hashable, List[int]] = {}
    for litter_id in litter_ids:
        nodes = list(range(next_node, next_node + len(prepared[litter_id])))
        route_nodes[litter_id] = nodes
        next_node += len(nodes)
    sink = next_node

    graph = _ResidualGraph(sink + 1)
    selected_edges: Dict[Hashable, List[Tuple[RouteCandidate, _Edge]]] = {}
    for litter_id in litter_ids:
        event_node = event_nodes[litter_id]
        graph.add_edge(source, event_node, capacity=1, cost=0)
        selected_edges[litter_id] = []
        for candidate, route_node in zip(
            prepared[litter_id], route_nodes[litter_id]
        ):
            selection_edge = graph.add_edge(
                event_node,
                route_node,
                capacity=1,
                cost=_scaled_cost(candidate.cost),
            )
            graph.add_edge(route_node, sink, capacity=1, cost=0)
            selected_edges[litter_id].append((candidate, selection_edge))

    required_flow = len(litter_ids)
    for _ in range(required_flow):
        predecessor, path_cost = graph.shortest_augmenting_path(source, sink)
        if path_cost is None:
            # This indicates an implementation error because each event has a
            # private finite NULL route.
            raise RuntimeError("min-cost flow became infeasible despite NULL routes")
        graph.send_one(source, sink, predecessor)

    assignments: Dict[Hashable, Assignment] = {}
    for litter_id in litter_ids:
        chosen = [
            candidate
            for candidate, edge in selected_edges[litter_id]
            if edge.capacity == 0
        ]
        if len(chosen) != 1:
            raise RuntimeError(
                "expected exactly one selected route for litter {!r}, got {}".format(
                    litter_id, len(chosen)
                )
            )
        route = chosen[0]
        scaled_costs = sorted(
            _scaled_cost(candidate.cost) for candidate in prepared[litter_id]
        )
        chosen_scaled = _scaled_cost(route.cost)
        remaining = list(scaled_costs)
        remaining.remove(chosen_scaled)
        margin = (
            (remaining[0] - chosen_scaled) / float(COST_SCALE)
            if remaining
            else None
        )
        assignments[litter_id] = Assignment(
            litter_id=litter_id,
            route=route,
            total_cost=float(route.cost),
            scaled_cost=chosen_scaled,
            margin_to_second=margin,
        )

    return assignments


__all__ = [
    "Assignment",
    "COST_SCALE",
    "RouteCandidate",
    "SYNTHETIC_NULL_ROUTE_ID",
    "solve_event_routes",
]
