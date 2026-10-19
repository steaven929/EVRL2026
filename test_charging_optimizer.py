import unittest

from specific.charging_optimizer import (
    ChargingCandidate,
    ChargingOptimizerConfig,
    FallbackReason,
    evaluate_assignment_objective,
    solve_charging_assignment,
)
from specific.charging_physics import ChargingConfig


def _physics():
    return ChargingConfig(
        distance_unit_km=0.1,
        battery_capacity_kwh=58.0,
        battery_capacity_internal=3000.0,
        energy_consumption_kwh_per_km=0.196,
        charging_power_kw=50.0,
        charging_efficiency=0.85,
        soc_min_ratio=0.20,
        soc_target_ratio=0.80,
        electricity_price_rmb_per_kwh=1.20,
    )


class ChargingOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.distance = [[0.0, 10.0], [10.0, 0.0]]
        self.config = ChargingOptimizerConfig(
            time_limit_seconds=2.0,
            mip_gap=0.0,
            actor_weight=10.0,
            urgency_weight=5.0,
            availability_weight=1.0,
            travel_weight=0.1,
            energy_cost_weight=0.01,
        )

    def test_capacity_repair_keeps_only_best_feasible_vehicle(self):
        candidates = [
            ChargingCandidate(0, 0, 700.0, 0.9),
            ChargingCandidate(1, 0, 1200.0, 0.2),
        ]
        result = solve_charging_assignment(
            candidates=candidates,
            distance_matrix=self.distance,
            charger_capacity=[0, 1],
            parking_capacity=[5, 5],
            physics=_physics(),
            config=self.config,
            heuristic_assignment={0: 1, 1: 1},
            solver_enabled=False,
        )
        self.assertEqual(result.assignments, {0: 1})
        self.assertTrue(result.used_fallback)

    def test_low_soc_vehicle_can_use_charger_in_its_current_zone(self):
        result = solve_charging_assignment(
            candidates=[ChargingCandidate(0, 0, 300.0, 1.0)],
            distance_matrix=self.distance,
            charger_capacity=[1, 0],
            parking_capacity=[5, 5],
            physics=_physics(),
            config=self.config,
            heuristic_assignment={0: 0},
            solver_enabled=False,
        )
        self.assertEqual(result.assignments, {0: 0})
        self.assertEqual(result.candidate_count, 1)
        self.assertEqual(result.feasible_pair_count, 1)

    def test_vehicle_that_cannot_reach_safe_soc_is_not_assigned(self):
        far_distance = [[0.0, 2000.0], [2000.0, 0.0]]
        result = solve_charging_assignment(
            candidates=[ChargingCandidate(0, 0, 610.0, 1.0)],
            distance_matrix=far_distance,
            charger_capacity=[0, 1],
            parking_capacity=[5, 5],
            physics=_physics(),
            config=self.config,
            heuristic_assignment={},
        )
        self.assertEqual(result.assignments, {})

    def test_disabled_solver_returns_feasible_heuristic(self):
        result = solve_charging_assignment(
            candidates=[ChargingCandidate(0, 0, 900.0, 0.5)],
            distance_matrix=self.distance,
            charger_capacity=[0, 1],
            parking_capacity=[5, 5],
            physics=_physics(),
            config=self.config,
            heuristic_assignment={0: 1},
            solver_enabled=False,
        )
        self.assertEqual(result.assignments, {0: 1})
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.fallback_reason, FallbackReason.DISABLED.value)

    def test_objective_uses_same_scoring_for_heuristic_and_milp(self):
        candidates = [ChargingCandidate(0, 0, 900.0, 0.5)]
        value = evaluate_assignment_objective(
            {0: 1}, candidates, self.distance, _physics(), self.config
        )
        self.assertGreater(value, 0.0)
        self.assertEqual(
            evaluate_assignment_objective({}, candidates, self.distance, _physics(), self.config),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
