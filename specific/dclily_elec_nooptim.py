# -*- coding: utf-8 -*-
"""No-optimizer variant of the existing dclily environment.

The base Gym action space stays flat for compatibility with existing MARL
callers, but this wrapper also accepts a `(ZONE_NUM, 3)` action matrix as a
convenience input form and flattens it before delegating to the base step.
"""

from specific.dclily_elec import *  # noqa: F401,F403
from specific.dclily_elec import dclily as _BaseDclily

import numpy as np
import threading


class dclily(_BaseDclily):
    _solveproblem_patch_lock = threading.Lock()
    _local_idle_reserve = 2
    _dispatch_soc_factor = 1.5
    _demand_soc_factor = 1.3
    _dispatch_fraction_cap = 0.35
    _demand_fraction_cap = 0.5

    def _get_local_idle_reserve(self):
        return int(getattr(self, "_local_idle_reserve", 2))

    def _get_soc_factor(self, kind):
        if kind == "dispatch":
            return float(getattr(self, "_dispatch_soc_factor", 1.5))
        return float(getattr(self, "_demand_soc_factor", 1.3))

    def _get_fraction_cap(self, kind):
        if kind == "dispatch":
            return float(getattr(self, "_dispatch_fraction_cap", 0.35))
        return float(getattr(self, "_demand_fraction_cap", 0.5))

    def _get_max_executions(self, trip_count, kind):
        if trip_count <= 0:
            return 0
        cap = self._get_fraction_cap(kind)
        return max(1, int(np.floor(float(trip_count) * cap)))

    def _greedy_dispatch_solveproblem(self, dist, avail, demand, car, reward, re,
                                      car_mem, timestep, real_reward, detail,
                                      revise, pre_avail):
        """Local, greedy replacement for the first dispatch solveproblem call."""
        car = np.asarray(car)
        avail = np.asarray(avail)
        demand = np.asarray(demand)
        dist = np.asarray(dist)
        rel_dispatch = np.zeros((self.ZONE_NUM, self.ZONE_NUM), dtype=int)
        real_reward_back = np.zeros(self.ZONE_NUM, dtype=float)

        for origin in range(min(self.ZONE_NUM, demand.shape[0])):
            for dest in range(min(self.ZONE_NUM, demand.shape[1])):
                if origin == dest:
                    continue

                trip_count = int(demand[origin][dest])
                if trip_count <= 0:
                    continue
                trip_count = self._get_max_executions(trip_count, "dispatch")

                trip_dist = float(dist[origin][dest])
                if trip_dist < 0:
                    continue

                for _ in range(trip_count):
                    if origin < len(avail[0]) and int(avail[0][origin]) <= self._get_local_idle_reserve():
                        break

                    chosen_vehicle = -1
                    for vid in range(len(car)):
                        if int(car[vid][0]) != origin:
                            continue
                        if int(car[vid][2]) > 0:
                            continue
                        if float(car[vid][1]) < trip_dist * self._get_soc_factor("dispatch"):
                            continue
                        chosen_vehicle = vid
                        break

                    if chosen_vehicle < 0:
                        break

                    car[chosen_vehicle][0] = dest
                    car[chosen_vehicle][1] = car[chosen_vehicle][1] - trip_dist
                    car[chosen_vehicle][2] = trip_dist

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

                    if chosen_vehicle < len(car_mem):
                        car_mem[chosen_vehicle][0] = origin
                        car_mem[chosen_vehicle][1] = timestep
                        car_mem[chosen_vehicle][2] = 0

                    reward[timestep][origin] += re * trip_dist
                    real_reward_back[origin] += re * trip_dist
                    rel_dispatch[origin][dest] += 1

        real_reward = np.asarray(real_reward)
        real_reward[:] = real_reward + real_reward_back
        if revise == 0:
            reward[timestep] = real_reward_back.copy()

        return car, avail, car_mem, reward, real_reward, rel_dispatch

    def _greedy_demand_solveproblem(self, dist, avail, demand, car, reward, re,
                                    car_mem, timestep, real_reward, detail,
                                    revise, pre_avail):
        """Local, greedy replacement for the second demand solveproblem call."""
        car = np.asarray(car)
        avail = np.asarray(avail)
        demand = np.asarray(demand)
        dist = np.asarray(dist)
        rel_demand = np.zeros((self.ZONE_NUM, self.ZONE_NUM), dtype=int)
        real_reward_back = np.zeros(self.ZONE_NUM, dtype=float)

        for origin in range(min(self.ZONE_NUM, demand.shape[0])):
            for dest in range(min(self.ZONE_NUM, demand.shape[1])):
                if origin == dest:
                    continue

                trip_count = int(demand[origin][dest])
                if trip_count <= 0:
                    continue
                trip_count = self._get_max_executions(trip_count, "demand")

                trip_dist = float(dist[origin][dest])
                if trip_dist < 0:
                    continue

                for _ in range(trip_count):
                    if origin < len(avail[0]) and int(avail[0][origin]) <= self._get_local_idle_reserve():
                        break

                    chosen_vehicle = -1
                    for vid in range(len(car)):
                        if int(car[vid][0]) != origin:
                            continue
                        if int(car[vid][2]) > 0:
                            continue
                        if float(car[vid][1]) < trip_dist * self._get_soc_factor("demand"):
                            continue
                        chosen_vehicle = vid
                        break

                    if chosen_vehicle < 0:
                        break

                    origin_reward = re * trip_dist

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

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        flat_size = 3 * self.ZONE_NUM

        if action.shape == (self.ZONE_NUM, 3):
            action = np.concatenate([action[:, 0], action[:, 1], action[:, 2]], axis=0)
        elif action.ndim == 1 and action.shape[0] == flat_size:
            pass
        else:
            raise ValueError(
                f"Invalid action shape {action.shape}; expected ({flat_size},) or "
                f"({self.ZONE_NUM}, 3)"
            )

        with self._solveproblem_patch_lock:
            original_solveproblem = gur.solveproblem
            call_state = {"count": 0}

            def _solveproblem_wrapper(*args, **kwargs):
                call_state["count"] += 1
                if call_state["count"] == 1:
                    return self._greedy_dispatch_solveproblem(*args, **kwargs)
                if call_state["count"] == 2:
                    return self._greedy_demand_solveproblem(*args, **kwargs)
                return original_solveproblem(*args, **kwargs)

            gur.solveproblem = _solveproblem_wrapper
            try:
                return super().step(action)
            finally:
                gur.solveproblem = original_solveproblem
