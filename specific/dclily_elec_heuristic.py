# -*- coding: utf-8 -*-
"""Heuristic baseline environment built on the no-optimizer execution path."""

import numpy as np

from specific.dclily_elec import BATTERY_CAPACITY
from specific.dclily_elec_nooptim import dclily as _NoOptimDclily


class dclily(_NoOptimDclily):
    _heuristic_local_idle_reserve = 2
    _heuristic_low_soc_threshold = 0.45

    def _build_depart_preferences(self, dispatchable_counts, outgoing_demand):
        reserve = float(getattr(self, "_heuristic_local_idle_reserve", 2))
        surplus = np.maximum(dispatchable_counts - outgoing_demand - reserve, 0.0)
        denom = np.maximum(dispatchable_counts, 1.0)
        return np.clip(surplus / denom, 0.0, 1.0)

    def _build_destination_preferences(self, dispatchable_counts, outgoing_demand):
        shortage = np.maximum(outgoing_demand - dispatchable_counts, 0.0)
        if float(np.sum(shortage)) <= 0:
            if hasattr(self, "_get_future_out_demand"):
                future = np.asarray(self._get_future_out_demand(step_offset=1), dtype=float)
                future = np.maximum(future, 0.0)
                if float(np.sum(future)) > 0:
                    return future / np.sum(future)
            return np.zeros(self.ZONE_NUM, dtype=np.float32)
        return np.asarray(shortage / np.sum(shortage), dtype=np.float32)

    def _build_charge_preferences(self):
        charge_pref = np.zeros(self.ZONE_NUM, dtype=np.float32)
        low_soc_threshold = float(getattr(self, "_heuristic_low_soc_threshold", 0.45)) * float(BATTERY_CAPACITY)

        for zone in range(self.ZONE_NUM):
            idle_soc = []
            for vid in range(len(self.car)):
                if int(self.car[vid][2]) > 0:
                    continue
                if int(self.car[vid][0]) != zone:
                    continue
                idle_soc.append(float(self.car[vid][1]))

            if not idle_soc:
                continue

            low_soc_count = sum(1 for soc in idle_soc if soc <= low_soc_threshold)
            if low_soc_count <= 0:
                continue

            zone_pref = low_soc_count / len(idle_soc)
            has_local_charger = zone in self.charging_stations and len(self.charging_stations[zone]) > 0
            if has_local_charger:
                charge_pref[zone] += float(zone_pref)
                continue

            nearest_zone = None
            nearest_dist = None
            for candidate_zone, chargers in self.charging_stations.items():
                if len(chargers) == 0:
                    continue
                candidate_dist = float(self.dist[zone][candidate_zone])
                if nearest_zone is None or candidate_dist < nearest_dist:
                    nearest_zone = candidate_zone
                    nearest_dist = candidate_dist

            if nearest_zone is not None:
                distance_weight = 1.0 / (1.0 + max(0.0, nearest_dist))
                charge_pref[nearest_zone] += float(zone_pref * distance_weight)

        return np.clip(charge_pref, 0.0, 1.0)

    def _build_heuristic_action(self):
        dispatchable = np.asarray(self.available[0], dtype=np.float32)

        if getattr(self, "od_demand_now", None) is not None and np.size(self.od_demand_now) > 0:
            outgoing_demand = np.sum(np.asarray(self.od_demand_now[0], dtype=np.float32), axis=1)
        else:
            outgoing_demand = np.zeros(self.ZONE_NUM, dtype=np.float32)

        depart_pref = self._build_depart_preferences(dispatchable, outgoing_demand)
        dest_pref = self._build_destination_preferences(dispatchable, outgoing_demand)
        charge_pref = self._build_charge_preferences()

        return np.concatenate([depart_pref, dest_pref, charge_pref], axis=0).astype(np.float32)

    def step(self, action=None):
        heuristic_action = self._build_heuristic_action()
        return super().step(heuristic_action)
