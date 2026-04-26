# -*- coding: utf-8 -*-
"""
MARL评估指标模块
"""
import numpy as np


def _battery_capacity(env):
    return getattr(env, 'BATTERY_CAPACITY', None) or getattr(env.unwrapped, 'BATTERY_CAPACITY', 3000)


def _time_length(env):
    return float(getattr(env, 'TIME_LENGTH', None) or getattr(env.unwrapped, 'TIME_LENGTH', 10))


def compute_charging_efficiency(env):
    effective_charging = 0
    total_charging = 0
    battery_capacity = _battery_capacity(env)
    for i in range(env.CAR_NUM):
        if env.car[i][3] != -1 and env.car[i][2] == 0:
            total_charging += 1
            if env.car[i][1] < battery_capacity * 0.95:
                effective_charging += 1
    if total_charging == 0:
        return np.nan
    return effective_charging / total_charging


def compute_vehicle_utilization(env):
    active_vehicles = 0
    for i in range(env.CAR_NUM):
        if env.car[i][2] > 0:
            active_vehicles += 1
    return active_vehicles / env.CAR_NUM


def compute_fleet_health(env):
    total_soc = sum(env.car[i][1] for i in range(env.CAR_NUM))
    avg_soc = total_soc / env.CAR_NUM
    return avg_soc / _battery_capacity(env)


def compute_zone_coordination(env):
    vehicle_counts = [0] * env.ZONE_NUM
    for i in range(env.CAR_NUM):
        zone = env.car[i][0]
        if env.car[i][2] == 0 and env.car[i][3] == -1:
            vehicle_counts[zone] += 1
    if sum(vehicle_counts) == 0:
        return np.nan
    mean_count = np.mean(vehicle_counts)
    if mean_count == 0:
        return np.nan
    std_count = np.std(vehicle_counts)
    coordination = 1 - min(std_count / mean_count, 1.0)
    return max(coordination, 0.0)


def compute_response_readiness(env):
    ready_vehicles = 0
    battery_capacity = _battery_capacity(env)
    for i in range(env.CAR_NUM):
        if env.car[i][2] == 0 and env.car[i][3] == -1 and env.car[i][1] > battery_capacity * 0.6:
            ready_vehicles += 1
    return ready_vehicles / env.CAR_NUM


def compute_charger_occupancy_rate(env):
    stations = getattr(env, 'charging_stations', None)
    if stations is None and hasattr(env, 'unwrapped'):
        stations = getattr(env.unwrapped, 'charging_stations', None)
    if not stations:
        return np.nan

    total = 0
    occupied = 0
    for chargers in stations.values():
        total += len(chargers)
        occupied += sum(1 for ch in chargers if ch.get('occupied_by') is not None)
    if total <= 0:
        return np.nan
    return occupied / total


def compute_charger_queue_delay(env):
    battery_capacity = _battery_capacity(env)
    time_len = _time_length(env)

    low_soc_waiting = 0
    charging_now = 0
    for i in range(env.CAR_NUM):
        remain = env.car[i][2]
        charger_id = env.car[i][3]
        soc = env.car[i][1]
        if remain == 0 and charger_id != -1:
            charging_now += 1
        if remain == 0 and charger_id == -1 and soc < battery_capacity * 0.3:
            low_soc_waiting += 1

    if low_soc_waiting == 0:
        return 0.0
    return (low_soc_waiting / max(charging_now, 1)) * time_len


def compute_energy_not_served(info):
    total_demand = float(info.get('total_demand', np.nan))
    served_orders = float(info.get('served_orders', np.nan))
    if not np.isfinite(total_demand) or not np.isfinite(served_orders):
        return np.nan
    return max(0.0, total_demand - served_orders)


def compute_safety_soc_violation_count(env):
    battery_capacity = _battery_capacity(env)
    threshold = battery_capacity * 0.2
    violations = 0
    for i in range(env.CAR_NUM):
        if env.car[i][1] < threshold:
            violations += 1
    return float(violations)


def compute_charge_shift_success_rate(env):
    logs = getattr(env, 'current_step_logs', None)
    if logs is None and hasattr(env, 'unwrapped'):
        logs = getattr(env.unwrapped, 'current_step_logs', None)
    if not isinstance(logs, dict) or len(logs) == 0:
        return np.nan

    decisions = 0
    shifted = 0
    for entry in logs.values():
        if not isinstance(entry, dict):
            continue
        if entry.get('decision_charge', False):
            decisions += 1
            if bool(entry.get('shifted', False)):
                shifted += 1
    if decisions == 0:
        return np.nan
    return shifted / decisions


def compute_all_metrics(env, info):
    return {
        'charging_efficiency': compute_charging_efficiency(env),
        'vehicle_utilization': compute_vehicle_utilization(env),
        'fleet_health': compute_fleet_health(env),
        'zone_coordination': compute_zone_coordination(env),
        'response_readiness': compute_response_readiness(env),
        'charger_occupancy_rate': compute_charger_occupancy_rate(env),
        'charger_queue_delay': compute_charger_queue_delay(env),
        'energy_not_served': compute_energy_not_served(info),
        'safety_soc_violation_count': compute_safety_soc_violation_count(env),
        'charge_shift_success_rate': compute_charge_shift_success_rate(env),
    }
