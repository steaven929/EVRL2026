"""Single-period P1 charging assignment with heuristic fallback."""

from dataclasses import dataclass
from enum import Enum
from time import perf_counter
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover - exercised through fallback in deployments
    gp = None
    GRB = None

from specific.charging_physics import (
    ChargingConfig,
    charge_for_duration,
    distance_to_energy_kwh,
    energy_kwh_to_internal,
)


class FallbackReason(str, Enum):
    NONE = "none"
    DISABLED = "disabled"
    NO_ELIGIBLE_VEHICLES = "no_eligible_vehicles"
    NO_FEASIBLE_CHARGER = "no_feasible_charger"
    SOLVER_UNAVAILABLE = "solver_unavailable"
    INFEASIBLE = "infeasible"
    TIME_LIMIT_NO_INCUMBENT = "time_limit_without_incumbent"
    SOLVER_ERROR = "solver_error"
    OBJECTIVE_GUARD = "objective_guard"


@dataclass(frozen=True)
class ChargingCandidate:
    vehicle_id: int
    origin_zone: int
    soc_internal: float
    actor_preference: float


@dataclass(frozen=True)
class ChargingOptimizerConfig:
    time_limit_seconds: float = 5.0
    mip_gap: float = 0.01
    actor_weight: float = 10.0
    urgency_weight: float = 5.0
    availability_weight: float = 1.0
    travel_weight: float = 0.1
    energy_cost_weight: float = 0.01
    charging_minutes: float = 30.0
    objective_tolerance: float = 1e-7
    threads: int = 4


@dataclass(frozen=True)
class ChargingSolveResult:
    assignments: Dict[int, int]
    candidate_count: int
    feasible_pair_count: int
    used_fallback: bool
    fallback_reason: str
    status_code: Optional[int]
    status_name: str
    build_time_seconds: float
    solve_time_seconds: float
    total_time_seconds: float
    objective_value: float
    heuristic_objective: float
    best_bound: Optional[float]
    mip_gap: Optional[float]
    node_count: Optional[float]
    time_limit_reached: bool


def _candidate_map(candidates: Iterable[ChargingCandidate]) -> Dict[int, ChargingCandidate]:
    result = {}
    for candidate in candidates:
        if candidate.vehicle_id in result:
            raise ValueError(f"duplicate vehicle_id {candidate.vehicle_id}")
        result[candidate.vehicle_id] = candidate
    return result


def _pair_score(
    candidate: ChargingCandidate,
    destination: int,
    distance_matrix: Sequence[Sequence[float]],
    physics: ChargingConfig,
    config: ChargingOptimizerConfig,
) -> Optional[float]:
    if destination < 0 or destination >= len(distance_matrix):
        return None
    if candidate.origin_zone < 0 or candidate.origin_zone >= len(distance_matrix):
        return None
    distance_internal = float(distance_matrix[candidate.origin_zone][destination])
    travel_internal = energy_kwh_to_internal(
        distance_to_energy_kwh(distance_internal, physics), physics
    )
    arrival_soc = float(candidate.soc_internal) - travel_internal
    # A vehicle already at the charger may recover from a low-SOC state.
    # Cross-zone charging trips must still arrive with the safety reserve.
    if distance_internal > 1e-9 and arrival_soc < physics.minimum_soc_internal - 1e-9:
        return None
    if arrival_soc < -1e-9:
        return None
    if arrival_soc >= physics.target_soc_internal - 1e-9:
        return None

    update = charge_for_duration(arrival_soc, config.charging_minutes, physics)
    if update.battery_energy_kwh <= 0:
        return None
    soc_span = max(physics.target_soc_internal - physics.minimum_soc_internal, 1e-9)
    urgency = min(1.0, max(0.0, (physics.target_soc_internal - arrival_soc) / soc_span))
    availability = update.battery_energy_kwh / physics.battery_capacity_kwh
    travel_km = distance_internal * physics.distance_unit_km
    return (
        config.actor_weight * float(candidate.actor_preference)
        + config.urgency_weight * urgency
        + config.availability_weight * availability
        - config.travel_weight * travel_km
        - config.energy_cost_weight * update.charging_cost_rmb
    )


def _feasible_pairs(
    candidates: Sequence[ChargingCandidate],
    distance_matrix: Sequence[Sequence[float]],
    charger_capacity: Sequence[int],
    parking_capacity: Sequence[int],
    physics: ChargingConfig,
    config: ChargingOptimizerConfig,
) -> Dict[Tuple[int, int], float]:
    if len(charger_capacity) != len(parking_capacity):
        raise ValueError("charger and parking capacity lengths must match")
    scores = {}
    for candidate in candidates:
        for zone, charger_slots in enumerate(charger_capacity):
            zone_capacity = min(int(charger_slots), int(parking_capacity[zone]))
            if zone_capacity <= 0:
                continue
            score = _pair_score(candidate, zone, distance_matrix, physics, config)
            if score is not None:
                scores[(candidate.vehicle_id, zone)] = score
    return scores


def _repair_assignment(
    assignment: Mapping[int, int],
    candidate_by_id: Mapping[int, ChargingCandidate],
    pair_scores: Mapping[Tuple[int, int], float],
    charger_capacity: Sequence[int],
    parking_capacity: Sequence[int],
) -> Dict[int, int]:
    repaired = {}
    used_by_zone = [0] * len(charger_capacity)
    ordered = sorted(
        assignment.items(),
        key=lambda item: (-pair_scores.get((int(item[0]), int(item[1])), float("-inf")), int(item[0])),
    )
    for raw_vehicle, raw_zone in ordered:
        vehicle, zone = int(raw_vehicle), int(raw_zone)
        if vehicle not in candidate_by_id or (vehicle, zone) not in pair_scores:
            continue
        capacity = min(int(charger_capacity[zone]), int(parking_capacity[zone]))
        if used_by_zone[zone] >= capacity:
            continue
        repaired[vehicle] = zone
        used_by_zone[zone] += 1
    return repaired


def evaluate_assignment_objective(
    assignment: Mapping[int, int],
    candidates: Sequence[ChargingCandidate],
    distance_matrix: Sequence[Sequence[float]],
    physics: ChargingConfig,
    config: ChargingOptimizerConfig,
) -> float:
    candidate_by_id = _candidate_map(candidates)
    total = 0.0
    for vehicle, zone in assignment.items():
        candidate = candidate_by_id.get(int(vehicle))
        if candidate is None:
            raise ValueError(f"unknown vehicle {vehicle}")
        score = _pair_score(candidate, int(zone), distance_matrix, physics, config)
        if score is None:
            raise ValueError(f"infeasible assignment vehicle={vehicle}, zone={zone}")
        total += score
    return float(total)


def _status_name(status: Optional[int]) -> str:
    if GRB is None or status is None:
        return "NOT_RUN"
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
    }
    return names.get(status, f"STATUS_{status}")


def _fallback_result(
    assignment: Dict[int, int],
    reason: FallbackReason,
    heuristic_objective: float,
    started_at: float,
    status_code: Optional[int] = None,
    solve_time: float = 0.0,
    candidate_count: int = 0,
    feasible_pair_count: int = 0,
) -> ChargingSolveResult:
    total = perf_counter() - started_at
    return ChargingSolveResult(
        assignments=assignment,
        candidate_count=int(candidate_count),
        feasible_pair_count=int(feasible_pair_count),
        used_fallback=True,
        fallback_reason=reason.value,
        status_code=status_code,
        status_name=_status_name(status_code),
        build_time_seconds=max(0.0, total - solve_time),
        solve_time_seconds=solve_time,
        total_time_seconds=total,
        objective_value=heuristic_objective,
        heuristic_objective=heuristic_objective,
        best_bound=None,
        mip_gap=None,
        node_count=None,
        time_limit_reached=status_code == getattr(GRB, "TIME_LIMIT", object()),
    )


def solve_charging_assignment(
    candidates: Sequence[ChargingCandidate],
    distance_matrix: Sequence[Sequence[float]],
    charger_capacity: Sequence[int],
    parking_capacity: Sequence[int],
    physics: ChargingConfig,
    config: ChargingOptimizerConfig,
    heuristic_assignment: Optional[Mapping[int, int]] = None,
    solver_enabled: bool = True,
    max_assignments: Optional[int] = None,
) -> ChargingSolveResult:
    started_at = perf_counter()
    candidate_by_id = _candidate_map(candidates)
    pairs = _feasible_pairs(
        candidates,
        distance_matrix,
        charger_capacity,
        parking_capacity,
        physics,
        config,
    )
    heuristic = _repair_assignment(
        heuristic_assignment or {},
        candidate_by_id,
        pairs,
        charger_capacity,
        parking_capacity,
    )
    heuristic_objective = sum(pairs[(vehicle, zone)] for vehicle, zone in heuristic.items())

    if not solver_enabled:
        return _fallback_result(
            heuristic, FallbackReason.DISABLED, heuristic_objective, started_at,
            candidate_count=len(candidates), feasible_pair_count=len(pairs),
        )
    if not candidates:
        return _fallback_result(
            heuristic, FallbackReason.NO_ELIGIBLE_VEHICLES, heuristic_objective, started_at,
            candidate_count=len(candidates), feasible_pair_count=len(pairs),
        )
    if not pairs:
        return _fallback_result(
            heuristic, FallbackReason.NO_FEASIBLE_CHARGER, heuristic_objective, started_at,
            candidate_count=len(candidates), feasible_pair_count=len(pairs),
        )
    if gp is None:
        return _fallback_result(
            heuristic, FallbackReason.SOLVER_UNAVAILABLE, heuristic_objective, started_at,
            candidate_count=len(candidates), feasible_pair_count=len(pairs),
        )

    try:
        model = gp.Model("P1_Charging_Assignment")
        model.Params.OutputFlag = 0
        model.Params.Threads = max(1, int(config.threads))
        model.Params.TimeLimit = max(0.001, float(config.time_limit_seconds))
        model.Params.MIPGap = max(0.0, float(config.mip_gap))
        variables = {
            pair: model.addVar(vtype=GRB.BINARY, name=f"charge_{pair[0]}_{pair[1]}")
            for pair in pairs
        }
        model.update()
        for vehicle in candidate_by_id:
            model.addConstr(
                gp.quicksum(var for (v, _), var in variables.items() if v == vehicle) <= 1,
                name=f"one_charge_zone_{vehicle}",
            )
        for zone in range(len(charger_capacity)):
            capacity = min(int(charger_capacity[zone]), int(parking_capacity[zone]))
            model.addConstr(
                gp.quicksum(var for (_, j), var in variables.items() if j == zone) <= capacity,
                name=f"charger_capacity_{zone}",
            )
        if max_assignments is not None:
            model.addConstr(
                gp.quicksum(variables.values()) <= max(0, int(max_assignments)),
                name="charging_budget",
            )
        model.setObjective(
            gp.quicksum(pairs[pair] * variable for pair, variable in variables.items()),
            GRB.MAXIMIZE,
        )
        for pair, variable in variables.items():
            variable.Start = 1.0 if heuristic.get(pair[0]) == pair[1] else 0.0
        build_done = perf_counter()
        model.optimize()
        solve_done = perf_counter()
    except Exception as exc:
        reason = (
            FallbackReason.SOLVER_UNAVAILABLE
            if "license" in str(exc).lower()
            else FallbackReason.SOLVER_ERROR
        )
        return _fallback_result(
            heuristic, reason, heuristic_objective, started_at,
            candidate_count=len(candidates), feasible_pair_count=len(pairs),
        )

    status = int(model.Status)
    solve_time = solve_done - build_done
    has_incumbent = int(model.SolCount) > 0
    if not has_incumbent:
        reason = (
            FallbackReason.TIME_LIMIT_NO_INCUMBENT
            if status == GRB.TIME_LIMIT
            else FallbackReason.INFEASIBLE
        )
        return _fallback_result(
            heuristic,
            reason,
            heuristic_objective,
            started_at,
            status_code=status,
            solve_time=solve_time,
            candidate_count=len(candidates),
            feasible_pair_count=len(pairs),
        )

    assignment = {
        vehicle: zone
        for (vehicle, zone), variable in variables.items()
        if variable.X > 0.5
    }
    objective = float(model.ObjVal)
    if objective + config.objective_tolerance < heuristic_objective:
        return _fallback_result(
            heuristic,
            FallbackReason.OBJECTIVE_GUARD,
            heuristic_objective,
            started_at,
            status_code=status,
            solve_time=solve_time,
            candidate_count=len(candidates),
            feasible_pair_count=len(pairs),
        )

    return ChargingSolveResult(
        assignments=assignment,
        candidate_count=len(candidates),
        feasible_pair_count=len(pairs),
        used_fallback=False,
        fallback_reason=FallbackReason.NONE.value,
        status_code=status,
        status_name=_status_name(status),
        build_time_seconds=build_done - started_at,
        solve_time_seconds=solve_time,
        total_time_seconds=solve_done - started_at,
        objective_value=objective,
        heuristic_objective=float(heuristic_objective),
        best_bound=float(model.ObjBound),
        mip_gap=float(model.MIPGap),
        node_count=float(model.NodeCount),
        time_limit_reached=status == GRB.TIME_LIMIT,
    )
