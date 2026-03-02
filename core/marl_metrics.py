# -*- coding: utf-8 -*-
"""
MARL评估指标模块
Created on 2026-03-01

提供多智能体强化学习的专用评估指标
"""
import numpy as np


def _battery_capacity(env):
    return getattr(env, 'BATTERY_CAPACITY', None) or getattr(env.unwrapped, 'BATTERY_CAPACITY', 3000)


def _time_length(env):
    return float(getattr(env, 'TIME_LENGTH', None) or getattr(env.unwrapped, 'TIME_LENGTH', 10))


def compute_charging_efficiency(env):
    """
    充电效率指标：有效充电次数 / 总充电次数

    有效充电：SOC < 95%时充电
    无效充电：SOC >= 95%时充电（充电黑客）

    Returns:
        float: 充电效率 (0-1)，越高越好
    """
    effective_charging = 0
    total_charging = 0
    battery_capacity = _battery_capacity(env)

    for i in range(env.CAR_NUM):
        if env.car[i][3] != -1 and env.car[i][2] == 0:  # 正在充电
            total_charging += 1
            if env.car[i][1] < battery_capacity * 0.95:  # SOC < 95%
                effective_charging += 1

    if total_charging == 0:
        return 0.0
    return effective_charging / total_charging


def compute_vehicle_utilization(env):
    """
    车辆利用率：(服务中车辆 + 调度中车辆) / 总车辆数

    衡量车队的整体利用效率

    Returns:
        float: 车辆利用率 (0-1)，越高越好
    """
    active_vehicles = 0

    for i in range(env.CAR_NUM):
        # 服务中：car[i][2] > 0（剩余里程>0）
        # 调度中：car[i][2] > 0 且不在充电
        if env.car[i][2] > 0:
            active_vehicles += 1

    return active_vehicles / env.CAR_NUM


def compute_fleet_health(env):
    """
    车队健康度：平均SOC / 电池容量

    衡量车队的整体电量健康状况

    Returns:
        float: 车队健康度 (0-1)，越高越好
    """
    total_soc = sum(env.car[i][1] for i in range(env.CAR_NUM))
    avg_soc = total_soc / env.CAR_NUM
    battery_capacity = getattr(env, 'BATTERY_CAPACITY', None) or getattr(env.unwrapped, 'BATTERY_CAPACITY', 3000)
    return avg_soc / battery_capacity


def compute_zone_coordination(env):
    """
    区域协调度：车辆分布的均衡性（1 - 标准差/均值）

    衡量多智能体的协调能力

    Returns:
        float: 区域协调度 (0-1)，越高越好（分布越均衡）
    """
    vehicle_counts = [0] * env.ZONE_NUM

    for i in range(env.CAR_NUM):
        zone = env.car[i][0]
        # 只统计空闲车辆（不在服务中，不在充电）
        if env.car[i][2] == 0 and env.car[i][3] == -1:
            vehicle_counts[zone] += 1

    if sum(vehicle_counts) == 0:
        return 0.0

    mean_count = np.mean(vehicle_counts)
    if mean_count == 0:
        return 0.0

    std_count = np.std(vehicle_counts)
    coordination = 1 - min(std_count / mean_count, 1.0)
    return max(coordination, 0.0)


def compute_response_readiness(env):
    """
    响应就绪度：(高电量空闲车辆数 / 总车辆数)

    高电量：SOC > 60%
    空闲：不在服务中，不在充电

    衡量车队随时响应新订单的能力

    Returns:
        float: 响应就绪度 (0-1)，越高越好
    """
    ready_vehicles = 0
    battery_capacity = _battery_capacity(env)

    for i in range(env.CAR_NUM):
        # 空闲且高电量
        if (env.car[i][2] == 0 and  # 不在移动
            env.car[i][3] == -1 and  # 不在充电
            env.car[i][1] > battery_capacity * 0.6):  # SOC > 60%
            ready_vehicles += 1

    return ready_vehicles / env.CAR_NUM


def compute_energy_efficiency(env, service_count, charging_count):
    """
    能量效率：服务订单数 / 充电次数

    衡量单位充电能完成多少服务

    Args:
        service_count: 本步服务的订单数
        charging_count: 本步充电的车辆数

    Returns:
        float: 能量效率，越高越好
    """
    if charging_count == 0:
        return service_count  # 没有充电但有服务，效率最高
    return service_count / charging_count


def compute_charger_occupancy_rate(env):
    """
    充电桩占用率：被占用充电桩 / 总充电桩
    """
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
    """
    充电排队延迟（分钟，代理量）：
    低SOC且空闲但未占桩车辆数 / 当前充电车辆数 * 时间步长
    """
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
    """
    因能量/供给不足未满足需求量（订单数代理）：total_demand - served_orders
    """
    total_demand = float(info.get('total_demand', np.nan))
    served_orders = float(info.get('served_orders', np.nan))
    if not np.isfinite(total_demand) or not np.isfinite(served_orders):
        return np.nan
    return max(0.0, total_demand - served_orders)


def compute_safety_soc_violation_count(env):
    """
    安全SOC违规次数（每步）：SOC < 20%容量 的车辆数
    """
    battery_capacity = _battery_capacity(env)
    threshold = battery_capacity * 0.2
    violations = 0
    for i in range(env.CAR_NUM):
        if env.car[i][1] < threshold:
            violations += 1
    return float(violations)


def compute_charge_shift_success_rate(env):
    """
    充电跨区转移成功率：本步充电决策中 shifted=True 的占比
    """
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
    """
    计算所有MARL专用指标

    Args:
        env: 环境实例
        info: step返回的info字典

    Returns:
        dict: 包含所有指标的字典
    """
    metrics = {
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

    # 如果info中有服务和充电信息，计算能量效率
    if 'service_count' in info and 'charging_count' in info:
        metrics['energy_efficiency'] = compute_energy_efficiency(
            env, info['service_count'], info['charging_count']
        )

    return metrics
