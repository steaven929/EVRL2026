import unittest

from specific.charging_physics import (
    ChargingConfig,
    allocate_chargers_by_demand,
    charge_for_duration,
    charging_duration_minutes,
    distance_to_energy_kwh,
    energy_kwh_to_internal,
    rounded_charger_count,
)


class ChargingPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ChargingConfig(
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

    def test_distance_energy_conversion_preserves_internal_scale(self):
        energy = distance_to_energy_kwh(100.0, self.cfg)
        self.assertAlmostEqual(energy, 1.96)
        self.assertAlmostEqual(energy_kwh_to_internal(energy, self.cfg), 101.3793103448)

    def test_half_hour_charge_uses_power_and_efficiency(self):
        update = charge_for_duration(600.0, 30.0, self.cfg)
        self.assertAlmostEqual(update.grid_energy_kwh, 25.0)
        self.assertAlmostEqual(update.battery_energy_kwh, 21.25)
        self.assertAlmostEqual(update.soc_after_internal, 1699.1379310345)
        self.assertAlmostEqual(update.charging_cost_rmb, 30.0)
        self.assertFalse(update.reached_target)

    def test_charge_clips_at_target_and_uses_only_effective_minutes(self):
        update = charge_for_duration(2300.0, 30.0, self.cfg)
        self.assertAlmostEqual(update.soc_after_internal, 2400.0)
        self.assertAlmostEqual(update.battery_energy_kwh, 58.0 / 30.0)
        self.assertAlmostEqual(update.effective_charging_minutes, 2.7294117647)
        self.assertTrue(update.reached_target)

    def test_vehicle_above_target_keeps_its_existing_soc(self):
        update = charge_for_duration(2800.0, 30.0, self.cfg)
        self.assertEqual(update.soc_before_internal, 2800.0)
        self.assertEqual(update.soc_after_internal, 2800.0)
        self.assertEqual(update.grid_energy_kwh, 0.0)
        self.assertEqual(update.battery_energy_kwh, 0.0)
        self.assertEqual(update.effective_charging_minutes, 0.0)
        self.assertTrue(update.reached_target)

    def test_duration_from_twenty_to_eighty_percent(self):
        self.assertAlmostEqual(charging_duration_minutes(600.0, self.cfg), 49.1294117647)

    def test_round_half_up_charger_count(self):
        self.assertEqual(rounded_charger_count(250, 0.15), 38)
        self.assertEqual(rounded_charger_count(250, 0.30), 75)
        self.assertEqual(rounded_charger_count(250, 0.45), 113)

    def test_base_allocation_covers_every_zone_then_uses_demand_rank(self):
        demand = [float(i) for i in range(50)]
        allocation = allocate_chargers_by_demand(
            total_chargers=75,
            parking_capacity=[5] * 50,
            demand_score=demand,
            ensure_zone_coverage=True,
        )
        self.assertEqual(sum(allocation), 75)
        self.assertTrue(all(value >= 1 for value in allocation))
        self.assertEqual(allocation[49], 2)
        self.assertEqual(allocation[25], 2)
        self.assertEqual(allocation[24], 1)


if __name__ == "__main__":
    unittest.main()
