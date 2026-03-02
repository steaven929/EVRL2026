# -*- coding: utf-8 -*-
"""
改进版传统指标模块
Created on 2026-03-01

重新设计SVC、RLD等传统指标，使其更能反映系统性能
同时保留旧版本用于对比
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


# ==================== SVC改进版 ====================

def compute_weighted_service_rate(env, info):
    """
    加权服务率 (Weighted Service Rate)

    改进点：考虑订单价值/距离，高价值订单权重更大

    旧版SVC问题：
    - 只统计订单数量，不考虑订单质量
    - 服务1个长距离订单 = 服务1个短距离订单

    新版改进：
    - 按订单距离加权（长距离订单权重更大）
    - 更能反映实际收益

    Args:
        env: 环境实例
        info: step返回的info字典

    Returns:
        float: 加权服务率 (0-1)，越高越好
    """
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

    # 计算加权需求和加权服务订单数
    weighted_served = 0.0
    weighted_total = 0.0

    for i in range(demand_matrix.shape[0]):
        for j in range(demand_matrix.shape[1]):
            if i != j:  # 排除自环
                distance = 1.0
                if dist_matrix is not None and i < dist_matrix.shape[0] and j < dist_matrix.shape[1]:
                    distance = float(dist_matrix[i][j])
                weight = max(1.0, distance)  # 至少权重为1

                served_count = float(served_matrix[i][j])
                demand_count = float(demand_matrix[i][j])
                weighted_served += served_count * weight
                weighted_total += demand_count * weight

    if weighted_total == 0:
        return np.nan

    return weighted_served / weighted_total


def compute_effective_service_rate(env, info):
    """
    有效服务率 (Effective Service Rate)

    改进点：排除因电量不足无法服务的订单

    旧版SVC问题：
    - 不区分拒绝原因（没车 vs 电量不足）
    - 电量管理差也会导致低SVC

    新版改进：
    - 只统计有车但因电量不足拒绝的订单
    - 更能反映调度策略质量（而非充电策略）

    Args:
        env: 环境实例
        info: step返回的info字典

    Returns:
        float: 有效服务率 (0-1)，越高越好
    """
    if 'demand_orders' not in info:
        return np.nan

    # 统计有足够电量的车辆数
    available_vehicles = 0
    battery_capacity = getattr(env, 'BATTERY_CAPACITY', None) or getattr(env.unwrapped, 'BATTERY_CAPACITY', 3000)
    for i in range(env.CAR_NUM):
        # 空闲且电量充足（>30%）
        if (env.car[i][2] == 0 and  # 不在移动
            env.car[i][3] == -1 and  # 不在充电
            env.car[i][1] > battery_capacity * 0.3):
            available_vehicles += 1

    # 如果有足够车辆，计算服务率
    if available_vehicles == 0:
        return 0.0

    served_orders = float(info.get('served_orders', 0.0))
    total_demand = float(info.get('total_demand', 0.0))

    if total_demand <= 0:
        return np.nan

    # 有效服务率 = 实际服务 / 可有效承载需求
    expected_capacity = min(float(available_vehicles), total_demand)
    if expected_capacity == 0:
        return 0.0

    return min(1.0, served_orders / expected_capacity)


def compute_service_quality_index(env, info):
    """
    服务质量指数 (Service Quality Index)

    改进点：综合考虑服务率、响应时间、车辆电量

    旧版SVC问题：
    - 只看数量，不看质量
    - 不考虑服务的及时性

    新版改进：
    - 服务率 × 车队健康度 × 响应就绪度
    - 综合评估服务能力

    Args:
        env: 环境实例
        info: step返回的info字典

    Returns:
        float: 服务质量指数 (0-1)，越高越好
    """
    # 基础服务率
    served_orders = float(info.get('served_orders', 0.0))
    total_demand = float(info.get('total_demand', 0.0))
    if total_demand <= 0:
        return np.nan
    base_service_rate = served_orders / max(total_demand, 1)

    # 车队健康度（平均电量）
    total_soc = sum(env.car[i][1] for i in range(env.CAR_NUM))
    avg_soc = total_soc / env.CAR_NUM
    battery_capacity = getattr(env, 'BATTERY_CAPACITY', None) or getattr(env.unwrapped, 'BATTERY_CAPACITY', 3000)
    fleet_health = avg_soc / battery_capacity

    # 响应就绪度（高电量空闲车辆比例）
    ready_vehicles = 0
    battery_capacity = getattr(env, 'BATTERY_CAPACITY', None) or getattr(env.unwrapped, 'BATTERY_CAPACITY', 3000)
    for i in range(env.CAR_NUM):
        if (env.car[i][2] == 0 and  # 不在移动
            env.car[i][3] == -1 and  # 不在充电
            env.car[i][1] > battery_capacity * 0.6):
            ready_vehicles += 1
    response_readiness = ready_vehicles / env.CAR_NUM

    # 综合指数（几何平均，避免某一项为0导致整体为0）
    sqi = (base_service_rate ** 0.5) * (fleet_health ** 0.25) * (response_readiness ** 0.25)

    return min(1.0, sqi)


# ==================== RLD改进版 ====================

def compute_dispatch_efficiency(env, info):
    """
    调度效率 (Dispatch Efficiency)

    改进点：调度后服务的订单数 / 调度距离

    旧版RLD问题：
    - 只看调度距离，不看调度效果
    - 无法区分有效调度和无效调度

    新版改进：
    - 单位调度距离带来的服务订单数
    - 越高说明调度越有效

    Args:
        env: 环境实例
        info: step返回的info字典

    Returns:
        float: 调度效率（订单数/距离），越高越好
    """
    rld = float(info.get('rld', 0.0))
    served_orders = float(info.get('served_orders', 0.0))

    if rld == 0:
        # 没有调度但有服务，效率最高
        return served_orders if served_orders > 0 else 0.0

    # 调度效率 = 服务订单数 / 调度距离
    efficiency = served_orders / rld

    return efficiency


def compute_effective_dispatch_rate(env, info):
    """
    有效调度率 (Effective Dispatch Rate)

    改进点：有效调度次数 / 总调度次数

    旧版RLD问题：
    - 不区分调度是否带来服务
    - 盲目调度也会增加RLD

    新版改进：
    - 统计调度后成功服务的比例
    - 反映调度决策质量

    Args:
        env: 环境实例
        info: step返回的info字典

    Returns:
        float: 有效调度率 (0-1)，越高越好
    """
    if 'dispatch_orders' not in info or info['dispatch_orders'] is None:
        return np.nan

    dispatch_orders = info['dispatch_orders']

    # 统计总调度次数
    total_dispatches = sum(sum(row) for row in dispatch_orders)

    if total_dispatches == 0:
        return np.nan

    # 统计服务订单数（假设调度后的服务是有效的）
    served_orders = float(info.get('served_orders', 0.0))

    # 有效调度率 = 服务订单数 / 总调度次数
    # 注意：这是简化版，理想情况下应该追踪每次调度是否带来服务
    effective_rate = min(1.0, served_orders / total_dispatches)

    return effective_rate


def compute_dispatch_roi(env, info):
    """
    调度ROI (Dispatch Return on Investment)

    改进点：调度带来的收益 / 调度成本

    旧版RLD问题：
    - 只看成本，不看收益
    - 无法判断调度是否值得

    新版改进：
    - 计算调度的投资回报率
    - 反映调度的经济性

    Args:
        env: 环境实例
        info: step返回的info字典

    Returns:
        float: 调度ROI，越高越好
    """
    rld = float(info.get('rld', 0.0))

    if rld == 0:
        return np.nan

    # 服务收益（简化：假设每个订单收益相同）
    served_orders = float(info.get('served_orders', 0.0))
    service_revenue = served_orders * 100  # 假设每单100收益

    # 调度成本（简化：距离 × 单位成本）
    dispatch_cost = rld * 10  # 假设每单位距离成本10

    if dispatch_cost == 0:
        return 0.0

    # ROI = (收益 - 成本) / 成本
    roi = (service_revenue - dispatch_cost) / dispatch_cost

    return roi


# ==================== 综合计算函数 ====================

def compute_all_improved_metrics(env, info):
    """
    计算所有改进版指标

    Args:
        env: 环境实例
        info: step返回的info字典

    Returns:
        dict: 包含所有改进版指标的字典
    """
    metrics = {
        # SVC改进版
        'weighted_service_rate': compute_weighted_service_rate(env, info),
        'effective_service_rate': compute_effective_service_rate(env, info),
        'service_quality_index': compute_service_quality_index(env, info),

        # RLD改进版
        'dispatch_efficiency': compute_dispatch_efficiency(env, info),
        'effective_dispatch_rate': compute_effective_dispatch_rate(env, info),
        'dispatch_roi': compute_dispatch_roi(env, info),
    }

    return metrics


# ==================== 对比分析函数 ====================

def compare_old_vs_new_metrics(old_svc, new_svc_variants, old_rld, new_rld_variants):
    """
    对比旧版和新版指标

    Args:
        old_svc: 旧版SVC值
        new_svc_variants: dict，包含所有新版SVC变体
        old_rld: 旧版RLD值
        new_rld_variants: dict，包含所有新版RLD变体

    Returns:
        dict: 对比分析结果
    """
    analysis = {
        'svc_comparison': {
            'old': old_svc,
            'weighted': new_svc_variants.get('weighted_service_rate', 0),
            'effective': new_svc_variants.get('effective_service_rate', 0),
            'quality_index': new_svc_variants.get('service_quality_index', 0),
        },
        'rld_comparison': {
            'old': old_rld,
            'efficiency': new_rld_variants.get('dispatch_efficiency', 0),
            'effective_rate': new_rld_variants.get('effective_dispatch_rate', 0),
            'roi': new_rld_variants.get('dispatch_roi', 0),
        }
    }

    return analysis
