# -*- coding: utf-8 -*-
"""
改进版传统指标模块
"""
import numpy as np


def _get_distance_matrix(env):
    if hasattr(env, 'dist'):
        return np.asarray(env.dist, dtype=float)
    if hasattr(env, 'distance'):
        return np.asarray(env.distance, dtype=float)
    inner = getattr(env, 'unwrapped', None)
    if inner is not None:
        if hasattr(inner, 'dist'):
            return np.asarray(inner.dist, dtype=float)
        if hasattr(inner, 'distance'):
            return np.asarray(inner.distance, dtype=float)
    return None


def compute_weighted_service_rate(env, info):
    demand_matrix = info.get('demand_matrix')
    served_matrix = info.get('demand_orders')
    if demand_matrix is None or served_matrix is None:
        return np.nan

    demand_matrix = np.asarray(demand_matrix, dtype=float)
    served_matrix = np.asarray(served_matrix, dtype=float)
    if demand_matrix.shape != served_matrix.shape:
        return np.nan

    total_demand = float(np.sum(demand_matrix))
    if total_demand <= 0:
        return np.nan

    dist_matrix = _get_distance_matrix(env)
    weighted_served = 0.0
    weighted_total = 0.0
    for i in range(demand_matrix.shape[0]):
        for j in range(demand_matrix.shape[1]):
            if i == j:
                continue
            distance = 1.0
            if dist_matrix is not None and i < dist_matrix.shape[0] and j < dist_matrix.shape[1]:
                distance = float(dist_matrix[i][j])
            weight = max(1.0, distance)
            weighted_served += float(served_matrix[i][j]) * weight
            weighted_total += float(demand_matrix[i][j]) * weight
    if weighted_total == 0:
        return np.nan
    return weighted_served / weighted_total


def compute_effective_service_rate(env, info):
    if 'demand_orders' not in info:
        return np.nan

    battery_capacity = getattr(env, 'BATTERY_CAPACITY', None) or getattr(env.unwrapped, 'BATTERY_CAPACITY', 3000)
    available_vehicles = 0
    for i in range(env.CAR_NUM):
        if env.car[i][2] == 0 and env.car[i][3] == -1 and env.car[i][1] > battery_capacity * 0.3:
            available_vehicles += 1

    if available_vehicles == 0:
        return 0.0

    served_orders = float(info.get('served_orders', 0.0))
    total_demand = float(info.get('total_demand', 0.0))
    if total_demand <= 0:
        return np.nan

    expected_capacity = min(float(available_vehicles), total_demand)
    if expected_capacity == 0:
        return 0.0
    return min(1.0, served_orders / expected_capacity)


def compute_service_quality_index(env, info):
    served_orders = float(info.get('served_orders', 0.0))
    total_demand = float(info.get('total_demand', 0.0))
    if total_demand <= 0:
        return np.nan
    base_service_rate = served_orders / max(total_demand, 1)

    total_soc = sum(env.car[i][1] for i in range(env.CAR_NUM))
    avg_soc = total_soc / env.CAR_NUM
    battery_capacity = getattr(env, 'BATTERY_CAPACITY', None) or getattr(env.unwrapped, 'BATTERY_CAPACITY', 3000)
    fleet_health = avg_soc / battery_capacity

    ready_vehicles = 0
    for i in range(env.CAR_NUM):
        if env.car[i][2] == 0 and env.car[i][3] == -1 and env.car[i][1] > battery_capacity * 0.6:
            ready_vehicles += 1
    response_readiness = ready_vehicles / env.CAR_NUM

    sqi = (base_service_rate ** 0.5) * (fleet_health ** 0.25) * (response_readiness ** 0.25)
    return min(1.0, sqi)


def compute_dispatch_efficiency(env, info):
    rld = float(info.get('rld', 0.0))
    served_orders = float(info.get('served_orders', 0.0))
    if rld == 0:
        return served_orders if served_orders > 0 else 0.0
    return served_orders / rld


def compute_effective_dispatch_rate(env, info):
    if 'dispatch_orders' not in info or info['dispatch_orders'] is None:
        return np.nan
    dispatch_orders = info['dispatch_orders']
    total_dispatches = sum(sum(row) for row in dispatch_orders)
    if total_dispatches == 0:
        return np.nan
    served_orders = float(info.get('served_orders', 0.0))
    return min(1.0, served_orders / total_dispatches)


def compute_dispatch_roi(env, info):
    rld = float(info.get('rld', 0.0))
    if rld == 0:
        return np.nan
    served_orders = float(info.get('served_orders', 0.0))
    service_revenue = served_orders * 100
    dispatch_cost = rld * 10
    if dispatch_cost == 0:
        return np.nan
    return (service_revenue - dispatch_cost) / dispatch_cost


def compute_all_improved_metrics(env, info):
    return {
        'weighted_service_rate': compute_weighted_service_rate(env, info),
        'effective_service_rate': compute_effective_service_rate(env, info),
        'service_quality_index': compute_service_quality_index(env, info),
        'dispatch_efficiency': compute_dispatch_efficiency(env, info),
        'effective_dispatch_rate': compute_effective_dispatch_rate(env, info),
        'dispatch_roi': compute_dispatch_roi(env, info),
    }
