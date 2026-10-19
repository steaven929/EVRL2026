import unittest

import numpy as np

from specific.dclily_elec import dclily
from specific import optim


class EnvironmentP1IntegrationTests(unittest.TestCase):
    def test_vehicle_zone_field_remains_an_integer_index(self):
        env = dclily()
        env.write_ini_cap(True)
        self.assertIsInstance(env.car[0][0], (int, np.integer))

    def test_p1_reserves_capacity_without_marking_travelling_vehicle_occupied(self):
        env = dclily()
        env.write_ini_cap(True)
        env.charging_assignment_mode = "p1"
        env.car[:, 2] = 1
        env.car[:, 3] = -1
        env.car[0] = [0, 700, 0, -1]
        env.car[1] = [0, 800, 0, -1]

        actions = {}
        env.check_idle_vehicles(np.ones(env.ZONE_NUM), actions, epoch=100)

        assigned = [vehicle for vehicle in (0, 1) if env.car[vehicle][3] != -1]
        self.assertTrue(assigned)
        self.assertIn(env.last_p1_stats["status_name"], {"OPTIMAL", "NOT_RUN"})
        for vehicle in assigned:
            zone = int(env.car[vehicle][0])
            charger_id = int(env.car[vehicle][3])
            station = next(
                item for item in env.charging_stations[zone]
                if int(item["charger_id"]) == charger_id
            )
            self.assertEqual(station["reserved_by"], vehicle)
            self.assertIsNone(station["occupied_by"])

    def test_arrived_reserved_vehicle_charges_with_physical_update(self):
        env = dclily()
        env.write_ini_cap(True)
        vehicle = 0
        zone = 0
        station = env.charging_stations[zone][0]
        env.car[vehicle] = [zone, 600, 0, station["charger_id"]]
        station["reserved_by"] = vehicle

        update = env._charge_vehicle_for_step(vehicle)

        self.assertAlmostEqual(update.grid_energy_kwh, 25.0)
        self.assertAlmostEqual(env.car[vehicle][1], 1699.1379310345)
        self.assertEqual(station["occupied_by"], vehicle)
        self.assertIsNone(station["reserved_by"])

    def test_p1_mode_does_not_use_legacy_arrival_charger_assignment(self):
        env = dclily()
        env.write_ini_cap(True)
        env.charging_assignment_mode = "p1"
        vehicle = 0
        zone = next(zone for zone, stations in env.charging_stations.items() if stations)
        env.car[vehicle] = [zone, 2800, 0, -1]

        destination = env._handle_arrived_vehicle_charging(vehicle, zone)

        self.assertEqual(destination, zone)
        self.assertEqual(env.car[vehicle][3], -1)
        self.assertTrue(all(
            station["reserved_by"] is None and station["occupied_by"] is None
            for station in env.charging_stations[zone]
        ))

    def test_p1_stats_report_candidate_filter_counts(self):
        env = dclily()
        env.write_ini_cap(True)
        env.charging_assignment_mode = "p1"
        env.car[:, 2] = 1
        env.car[:, 3] = -1
        env.car[0] = [0, 700, 0, -1]
        env.car[1] = [0, 2800, 0, -1]

        env.check_idle_vehicles(np.ones(env.ZONE_NUM), {}, epoch=100)

        self.assertEqual(env.last_p1_stats["idle_unassigned_count"], 2)
        self.assertEqual(env.last_p1_stats["below_target_count"], 1)
        self.assertEqual(env.last_p1_stats["candidate_count"], 1)
        self.assertGreater(env.last_p1_stats["feasible_pair_count"], 0)

    def test_p1_mode_never_marks_an_ordinary_trip_as_charging(self):
        env = dclily()
        env.write_ini_cap(True)
        env.charging_assignment_mode = "p1"
        vehicle = 0
        zone = next(zone for zone, stations in env.charging_stations.items() if stations)
        env.car[vehicle] = [zone, 2800, 1, -1]
        env.available[1][zone] = 0

        env.assign_parking_slot(vehicle, is_charging_order=False)

        self.assertEqual(env.car[vehicle][3], -1)

    def test_p0_never_dispatches_a_vehicle_reserved_for_charging(self):
        distance = np.array([[0.0, 10.0], [10.0, 0.0]])
        available = np.array([[1, 0], [1, 1]], dtype=float)
        demand = np.array([[0, 1], [0, 0]], dtype=float)
        cars = np.array([
            [0, 1000.0, 0.0, 0],
            [0, 1000.0, 0.0, -1],
        ], dtype=object)
        car_memory = np.array([[-1, 0, 100.0], [-1, 0, 0.0]], dtype=float)

        cars, *_ = optim.solveproblem(
            distance,
            available,
            demand,
            cars,
            [[0.0, 0.0]],
            1.0,
            car_memory,
            0,
            np.zeros(2),
            0,
            0,
            np.zeros(2),
        )

        self.assertEqual(int(cars[0][0]), 0)
        self.assertEqual(int(cars[0][3]), 0)
        self.assertEqual(int(cars[1][0]), 1)


if __name__ == "__main__":
    unittest.main()
