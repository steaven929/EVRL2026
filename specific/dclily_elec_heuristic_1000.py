# -*- coding: utf-8 -*-
"""1000-car heuristic baseline built on top of the current SIP environment."""

import numpy as np

import specific.dclily_elec as _base_env


def build_output_path(train_eps, curr_time):
    return f"./outputs/heuristic_1000car_{train_eps}ep_{curr_time}/"


class dclily(_base_env.dclily):
    _heuristic_low_soc_threshold = 0.35

    def _get_idle_vehicle_counts(self):
        zone_num = int(getattr(self, "ZONE_NUM", _base_env.ZONE_NUM))
        available = getattr(self, "available", None)
        if isinstance(available, np.ndarray) and available.ndim >= 2 and available.shape[1] >= zone_num:
            return np.asarray(available[0][:zone_num], dtype=np.float32)

        idle_counts = np.zeros(zone_num, dtype=np.float32)
        for vehicle in np.asarray(getattr(self, "car", []), dtype=float):
            try:
                zone = int(vehicle[0])
                remain = float(vehicle[2])
            except Exception:
                continue
            if 0 <= zone < zone_num and remain <= 0:
                idle_counts[zone] += 1.0
        return idle_counts

    def _get_outgoing_demand_estimate(self):
        zone_num = int(getattr(self, "ZONE_NUM", _base_env.ZONE_NUM))
        demand_now = getattr(self, "od_demand_now", None)
        if isinstance(demand_now, np.ndarray) and demand_now.size > 0:
            demand_arr = np.asarray(demand_now[0], dtype=np.float32)
            if demand_arr.ndim == 2:
                return np.sum(demand_arr[:zone_num, :zone_num], axis=1, dtype=np.float32)
        return np.zeros(zone_num, dtype=np.float32)

    def _get_low_soc_ratio(self):
        zone_num = int(getattr(self, "ZONE_NUM", _base_env.ZONE_NUM))
        threshold = float(getattr(self, "_heuristic_low_soc_threshold", 0.35)) * float(_base_env.BATTERY_CAPACITY)
        idle_counts = np.zeros(zone_num, dtype=np.float32)
        low_soc_counts = np.zeros(zone_num, dtype=np.float32)

        for vehicle in np.asarray(getattr(self, "car", []), dtype=float):
            try:
                zone = int(vehicle[0])
                soc = float(vehicle[1])
                remain = float(vehicle[2])
            except Exception:
                continue
            if 0 <= zone < zone_num and remain <= 0:
                idle_counts[zone] += 1.0
                if soc <= threshold:
                    low_soc_counts[zone] += 1.0

        return low_soc_counts / np.maximum(idle_counts, 1.0)

    def _build_depart_preferences(self, idle_counts, outgoing_demand):
        idle_counts = np.asarray(idle_counts, dtype=np.float32)
        outgoing_demand = np.asarray(outgoing_demand, dtype=np.float32)
        surplus = np.maximum(idle_counts - outgoing_demand, 0.0)
        return np.clip(surplus / np.maximum(idle_counts, 1.0), 0.0, 1.0).astype(np.float32)

    def _build_destination_preferences(self, idle_counts, outgoing_demand):
        zone_num = int(getattr(self, "ZONE_NUM", _base_env.ZONE_NUM))
        idle_counts = np.asarray(idle_counts, dtype=np.float32)
        outgoing_demand = np.asarray(outgoing_demand, dtype=np.float32)
        shortage = np.maximum(outgoing_demand - idle_counts, 0.0)
        total_shortage = float(np.sum(shortage))
        if total_shortage <= 0:
            return np.zeros(zone_num, dtype=np.float32)
        return (shortage / total_shortage).astype(np.float32)

    def _nearest_charger_zone(self, zone):
        charging_stations = getattr(self, "charging_stations", {})
        dist = getattr(self, "dist", None)
        nearest_zone = None
        nearest_dist = None
        for candidate_zone, chargers in charging_stations.items():
            if not chargers:
                continue
            candidate_zone = int(candidate_zone)
            if dist is None:
                candidate_dist = 0.0 if candidate_zone == zone else 1.0
            else:
                candidate_dist = float(dist[zone][candidate_zone])
            if nearest_zone is None or candidate_dist < nearest_dist:
                nearest_zone = candidate_zone
                nearest_dist = candidate_dist
        return nearest_zone, nearest_dist

    def _build_charge_preferences(self, idle_counts, outgoing_demand, low_soc_ratio):
        zone_num = int(getattr(self, "ZONE_NUM", _base_env.ZONE_NUM))
        idle_counts = np.asarray(idle_counts, dtype=np.float32)
        outgoing_demand = np.asarray(outgoing_demand, dtype=np.float32)
        low_soc_ratio = np.asarray(low_soc_ratio, dtype=np.float32)

        demand_pressure = outgoing_demand / np.maximum(idle_counts, 1.0)
        local_charge_need = np.clip(low_soc_ratio * np.clip(0.5 + 0.5 * demand_pressure, 0.0, 1.0), 0.0, 1.0)
        charge_pref = np.zeros(zone_num, dtype=np.float32)

        charging_stations = getattr(self, "charging_stations", {})
        for zone in range(zone_num):
            if local_charge_need[zone] <= 0:
                continue
            if zone in charging_stations and charging_stations[zone]:
                charge_pref[zone] += float(local_charge_need[zone])
                continue

            nearest_zone, nearest_dist = self._nearest_charger_zone(zone)
            if nearest_zone is None:
                continue
            distance_weight = 1.0 / (1.0 + max(0.0, float(nearest_dist)))
            charge_pref[nearest_zone] += float(local_charge_need[zone] * distance_weight)

        return np.clip(charge_pref, 0.0, 1.0).astype(np.float32)

    def _build_rule_action(self):
        idle_counts = self._get_idle_vehicle_counts()
        outgoing_demand = self._get_outgoing_demand_estimate()
        low_soc_ratio = self._get_low_soc_ratio()

        depart_pref = self._build_depart_preferences(idle_counts, outgoing_demand)
        dest_pref = self._build_destination_preferences(idle_counts, outgoing_demand)
        charge_pref = self._build_charge_preferences(idle_counts, outgoing_demand, low_soc_ratio)

        return np.concatenate([depart_pref, dest_pref, charge_pref]).astype(np.float32)

    def step(self, action):
        return super().step(self._build_rule_action())
