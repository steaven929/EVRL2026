# -*- coding: utf-8 -*-
"""Rule-gap-plus environment built on the no-optimizer execution path."""

import numpy as np

from specific.dclily_elec import BATTERY_CAPACITY
from specific.dclily_elec_nooptim import dclily as _NoOptimDclily


class dclily(_NoOptimDclily):
    _rule_gap_plus_local_idle_reserve = 1
    _rule_gap_plus_dispatch_fraction_cap = 0.4
    _rule_gap_plus_service_fraction_cap = 0.9
    _rule_gap_plus_low_soc_threshold = 0.4
    _rule_gap_plus_dispatch_soc_factor = 1.1
    _rule_gap_plus_service_soc_factor = 1.0

    _local_idle_reserve = 1
    _dispatch_fraction_cap = 0.4
    _demand_fraction_cap = 0.9
    _dispatch_soc_factor = 1.1
    _demand_soc_factor = 1.0

    def _get_rule_gap_plus_local_idle_reserve(self):
        return int(getattr(self, "_rule_gap_plus_local_idle_reserve", 1))

    def _get_rule_gap_plus_fraction_cap(self, kind):
        if kind == "dispatch":
            return float(getattr(self, "_rule_gap_plus_dispatch_fraction_cap", 0.4))
        return float(getattr(self, "_rule_gap_plus_service_fraction_cap", 0.9))

    def _get_rule_gap_plus_soc_factor(self, kind):
        if kind == "dispatch":
            return float(getattr(self, "_rule_gap_plus_dispatch_soc_factor", 1.1))
        return float(getattr(self, "_rule_gap_plus_service_soc_factor", 1.0))

    def _build_depart_preferences(self, dispatchable_counts, outgoing_demand):
        dispatchable_counts = np.asarray(dispatchable_counts, dtype=float)
        outgoing_demand = np.asarray(outgoing_demand, dtype=float)
        reserve = float(self._get_rule_gap_plus_local_idle_reserve())
        surplus = np.maximum(dispatchable_counts - outgoing_demand - reserve, 0.0)
        cap = np.floor(dispatchable_counts * self._get_rule_gap_plus_fraction_cap("dispatch"))
        capped_surplus = np.minimum(surplus, cap)
        denom = np.maximum(dispatchable_counts, 1.0)
        return np.clip(capped_surplus / denom, 0.0, 1.0).astype(np.float32)

    def _build_destination_preferences(self, dispatchable_counts, outgoing_demand):
        dispatchable_counts = np.asarray(dispatchable_counts, dtype=float)
        outgoing_demand = np.asarray(outgoing_demand, dtype=float)
        reserve = float(self._get_rule_gap_plus_local_idle_reserve())
        shortage = np.maximum(outgoing_demand - dispatchable_counts, 0.0)
        shortage = np.minimum(shortage, np.floor(outgoing_demand * self._get_rule_gap_plus_fraction_cap("service")))
        if float(np.sum(shortage)) <= 0:
            return np.zeros(self.ZONE_NUM, dtype=np.float32)

        surplus = np.maximum(dispatchable_counts - outgoing_demand - reserve, 0.0)
        if float(np.sum(surplus)) <= 0:
            total_shortage = float(np.sum(shortage))
            if total_shortage <= 0:
                return np.zeros(self.ZONE_NUM, dtype=np.float32)
            return (shortage / total_shortage).astype(np.float32)

        demand_intensity = outgoing_demand / np.maximum(dispatchable_counts, 1.0)
        weighted_shortage = np.zeros(self.ZONE_NUM, dtype=float)
        surplus_origins = np.where(surplus > 0)[0]
        for dest in range(self.ZONE_NUM):
            if shortage[dest] <= 0:
                continue
            proximity_score = 0.0
            for origin in surplus_origins:
                proximity_score += surplus[origin] / (1.0 + float(self.dist[origin][dest]))
            weighted_shortage[dest] = shortage[dest] * (1.0 + demand_intensity[dest]) * proximity_score

        total_weighted_shortage = float(np.sum(weighted_shortage))
        if total_weighted_shortage <= 0:
            return np.zeros(self.ZONE_NUM, dtype=np.float32)
        return (weighted_shortage / total_weighted_shortage).astype(np.float32)

    def _build_charge_preferences(self):
        charge_pref = np.zeros(self.ZONE_NUM, dtype=np.float32)
        low_soc_threshold = float(getattr(self, "_rule_gap_plus_low_soc_threshold", 0.4)) * float(BATTERY_CAPACITY)

        if getattr(self, "available", None) is not None and np.size(self.available) > 0:
            dispatchable_counts = np.asarray(self.available[0], dtype=float)
        else:
            dispatchable_counts = np.zeros(self.ZONE_NUM, dtype=float)

        if getattr(self, "od_demand_now", None) is not None and np.size(self.od_demand_now) > 0:
            outgoing_demand = np.sum(np.asarray(self.od_demand_now[0], dtype=float), axis=1)
        else:
            outgoing_demand = np.zeros(self.ZONE_NUM, dtype=float)

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

            demand_pressure = float(outgoing_demand[zone]) / max(float(dispatchable_counts[zone]), 1.0)
            low_soc_ratio = low_soc_count / len(idle_soc)
            if demand_pressure >= 1.5:
                demand_discount = 0.15
            elif demand_pressure >= 1.0:
                demand_discount = 0.3
            elif demand_pressure >= 0.5:
                demand_discount = 0.6
            else:
                demand_discount = 1.0
            if low_soc_ratio >= 0.8:
                demand_discount = max(demand_discount, 0.85)
            elif low_soc_ratio >= 0.6:
                demand_discount = max(demand_discount, 0.55)
            zone_score = low_soc_ratio * demand_discount

            if zone in self.charging_stations and len(self.charging_stations[zone]) > 0:
                charge_pref[zone] += float(zone_score)
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
                distance_weight = 1.0 / (1.0 + max(0.0, float(nearest_dist)))
                charge_pref[nearest_zone] += float(zone_score * distance_weight)

        return np.clip(charge_pref, 0.0, 1.0).astype(np.float32)

    def _build_rule_gap_plus_action(self):
        dispatchable = np.asarray(self.available[0], dtype=np.float32)

        if getattr(self, "od_demand_now", None) is not None and np.size(self.od_demand_now) > 0:
            outgoing_demand = np.sum(np.asarray(self.od_demand_now[0], dtype=np.float32), axis=1)
        else:
            outgoing_demand = np.zeros(self.ZONE_NUM, dtype=np.float32)

        depart_pref = self._build_depart_preferences(dispatchable, outgoing_demand)
        dest_pref = self._build_destination_preferences(dispatchable, outgoing_demand)
        charge_pref = self._build_charge_preferences()
        return np.concatenate([depart_pref, dest_pref, charge_pref], axis=0).astype(np.float32)

    def _estimate_order_reward(self, origin, dest, trip_dist, re):
        mock_trip_reward = getattr(self, "_mock_trip_reward", None)
        if isinstance(mock_trip_reward, dict):
            key = (int(origin), int(dest))
            if key in mock_trip_reward:
                return float(mock_trip_reward[key])
        return float(re) * float(trip_dist)

    def _greedy_demand_solveproblem(self, dist, avail, demand, car, reward, re,
                                    car_mem, timestep, real_reward, detail,
                                    revise, pre_avail):
        """Higher-value, feasible-order-first execution for the stronger rule baseline."""
        car = np.asarray(car)
        avail = np.asarray(avail)
        demand = np.asarray(demand)
        dist = np.asarray(dist)
        rel_demand = np.zeros((self.ZONE_NUM, self.ZONE_NUM), dtype=int)
        real_reward_back = np.zeros(self.ZONE_NUM, dtype=float)

        for origin in range(min(self.ZONE_NUM, demand.shape[0])):
            incoming_pressure = np.sum(demand[:, origin]) if demand.ndim >= 2 else 0.0
            dest_order = sorted(
                range(min(self.ZONE_NUM, demand.shape[1])),
                key=lambda dest: (
                    -(
                        self._estimate_order_reward(origin, dest, dist[origin][dest], re)
                        + 0.5 * float(demand[origin][dest])
                        + 0.1 * float(incoming_pressure)
                        - 0.15 * float(dist[origin][dest])
                    ) if origin != dest and float(dist[origin][dest]) >= 0 and float(demand[origin][dest]) > 0
                    else float("inf"),
                    float(dist[origin][dest]) if origin != dest else float("inf"),
                    dest,
                ),
            )
            for dest in dest_order:
                if origin == dest:
                    continue

                trip_count = int(demand[origin][dest])
                if trip_count <= 0:
                    continue
                trip_count = _NoOptimDclily._get_max_executions(self, trip_count, "demand")

                trip_dist = float(dist[origin][dest])
                if trip_dist < 0:
                    continue

                for _ in range(trip_count):
                    if origin < len(avail[0]) and int(avail[0][origin]) <= _NoOptimDclily._get_local_idle_reserve(self):
                        break

                    chosen_vehicle = -1
                    for vid in range(len(car)):
                        if int(car[vid][0]) != origin:
                            continue
                        if int(car[vid][2]) > 0:
                            continue
                        if float(car[vid][1]) < trip_dist * _NoOptimDclily._get_soc_factor(self, "demand"):
                            continue
                        chosen_vehicle = vid
                        break

                    if chosen_vehicle < 0:
                        break

                    origin_reward = self._estimate_order_reward(origin, dest, trip_dist, re)

                    if origin < len(avail[0]):
                        avail[0][origin] -= 1
                        if avail[0][origin] < 0:
                            avail[0][origin] = 0
                    if origin < len(avail[1]):
                        avail[1][origin] += 1
                    if dest < len(avail[1]):
                        avail[1][dest] -= 1
                        if avail[1][dest] < 0:
                            avail[1][dest] = 0

                    rel_demand[origin][dest] += 1
                    car[chosen_vehicle][2] = trip_dist
                    real_reward_back[origin] += origin_reward

                    if chosen_vehicle < len(car_mem) and car_mem[chosen_vehicle][0] != -1:
                        reward[car_mem[chosen_vehicle][1]][car_mem[chosen_vehicle][0]] += origin_reward
                        car_mem[chosen_vehicle][0] = -1
                    elif revise >= 2:
                        reward[timestep][origin] += origin_reward

                    car[chosen_vehicle][1] = car[chosen_vehicle][1] - trip_dist
                    car[chosen_vehicle][0] = dest

        real_reward = np.asarray(real_reward)
        real_reward[:] = real_reward + real_reward_back
        if revise == 0:
            reward[timestep] = real_reward_back.copy()

        return car, avail, car_mem, reward, real_reward, rel_demand

    def step(self, action=None):
        rule_action = self._build_rule_gap_plus_action()
        return super().step(rule_action)
