# -*- coding: utf-8 -*-
"""
MARL Ablation Entry (No Gurobi Optimizer)

通过 monkeypatch 替换 specific.optim.solveproblem，
保持 main_marl 其余训练/评估/输出流程不变。
"""

import copy
import numpy as np

import specific.optim as gur


# =========2026.1.30修改，最优解求解==========
def _safe_add_reward(reward, timestep, zone, delta):
    """兼容 list/ndarray 的 reward 累加。"""
    try:
        reward[timestep][zone] += delta
    except Exception:
        pass


def _collect_dispatchable_by_zone(car, zone_num):
    """收集每个区域可调度车辆（空闲且未占用充电桩）。"""
    by_zone = [[] for _ in range(zone_num)]
    for vid, c in enumerate(car):
        try:
            z = int(c[0])
            remain = float(c[2])
            charger_id = int(c[3]) if len(c) > 3 else -1
        except Exception:
            continue
        if 0 <= z < zone_num and remain <= 0 and charger_id == -1:
            by_zone[z].append(vid)
    return by_zone


def solveproblem_no_gurobi(dist, avail, demand, car, reward, re, car_mem, timestep, real_reward, detail, revise, pre_avail):
    """
    与 specific.optim.solveproblem 相同签名的启发式替代器。
    目标：去掉数学优化器，但保持返回契约、数据流与口径一致。
    """
    zone_num = len(avail[1])
    car_num = len(car)
    demand_arr = np.asarray(demand, dtype=float)
    dist_arr = np.asarray(dist, dtype=float)

    if demand_arr.ndim != 2:
        demand_arr = np.zeros((zone_num, zone_num), dtype=float)
    if dist_arr.ndim != 2:
        dist_arr = np.zeros((zone_num, zone_num), dtype=float)

    free_slots = np.asarray(copy.deepcopy(avail[1]), dtype=float)
    zone_assignment_np = np.zeros((zone_num, zone_num), dtype=int)
    real_reward_back = np.zeros(zone_num, dtype=float)

    dispatchable_by_zone = _collect_dispatchable_by_zone(car, zone_num)
    if len(car_mem) == car_num:
        for z in range(zone_num):
            dispatchable_by_zone[z].sort(
                key=lambda vid: float(car_mem[vid][2]) if len(car_mem[vid]) > 2 else 0.0,
                reverse=True,
            )

    for origin in range(zone_num):
        if origin >= demand_arr.shape[0]:
            continue

        for dest in range(zone_num):
            if origin == dest or dest >= demand_arr.shape[1]:
                continue

            need = int(max(0.0, demand_arr[origin][dest]))
            if need <= 0:
                continue

            pool = dispatchable_by_zone[origin]
            if not pool:
                continue

            while need > 0 and pool:
                vid = pool.pop(0)

                try:
                    current_zone = int(car[vid][0])
                    soc = float(car[vid][1])
                except Exception:
                    continue

                if current_zone != origin:
                    continue

                if origin >= dist_arr.shape[0] or dest >= dist_arr.shape[1]:
                    continue
                trip_dist = float(dist_arr[origin][dest])
                if trip_dist <= 0 or trip_dist > soc:
                    continue

                if dest < free_slots.shape[0] and free_slots[dest] <= 0:
                    break

                if origin < free_slots.shape[0]:
                    free_slots[origin] += 1.0
                if dest < free_slots.shape[0]:
                    free_slots[dest] -= 1.0

                if len(avail) > 0 and len(avail[0]) > origin:
                    avail[0][origin] -= 1

                zone_assignment_np[origin][dest] += 1

                real_reward_back[origin] += re * trip_dist
                if re < 0:
                    _safe_add_reward(reward, timestep, origin, re * trip_dist)
                    if len(car_mem[vid]) >= 3:
                        car_mem[vid][0] = origin
                        car_mem[vid][1] = timestep
                        car_mem[vid][2] = 0
                else:
                    if len(car_mem[vid]) > 0 and car_mem[vid][0] != -1:
                        src_zone = int(car_mem[vid][0])
                        src_t = int(car_mem[vid][1])
                        _safe_add_reward(reward, src_t, src_zone, re * trip_dist)
                        car_mem[vid][0] = -1
                    elif revise >= 2:
                        _safe_add_reward(reward, timestep, origin, re * trip_dist)

                car[vid][2] = trip_dist
                car[vid][1] -= trip_dist
                car[vid][0] = dest
                need -= 1

    avail[1] = np.asarray(np.maximum(free_slots, 0.0), dtype=int)
    real_reward = np.asarray(real_reward, dtype=float) + real_reward_back

    if revise == 0:
        try:
            reward[timestep] = real_reward_back
        except Exception:
            pass

    return car, avail, car_mem, reward, real_reward, zone_assignment_np


def main():
    gur.solveproblem = solveproblem_no_gurobi
    print("[Ablation] Gurobi solver disabled; using heuristic dispatcher.")

    import main_marl

    main_marl.main(output_prefix='marl_no_gurobi')


if __name__ == "__main__":
    main()
# ======修改结束
