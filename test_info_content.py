# -*- coding: utf-8 -*-
"""
测试脚本：验证环境step返回的info字典内容
Task 1: 验证dispatch_orders和demand_orders的数据结构
"""

import gym
import numpy as np
from gym.envs.registration import register
from core.util import get_args
from core.environment import env_agent_config

# 注册环境
register(id='dclily', entry_point='specific.dclily_elec:dclily')

print("=" * 80)
print("Task 1: 验证info字典内容")
print("=" * 80)

# 初始化配置和环境
cfg = get_args()
env = gym.make('dclily')
env.set_training_mode(0)

print("\n[步骤1] 环境初始化完成")
print(f"观测空间维度: {env.observation_space.shape}")
print(f"动作空间维度: {env.action_space.shape}")

# 重置环境
state = env.reset()
print("\n[步骤2] 环境重置完成")
print(f"初始状态形状: {state.shape}")

# 采样一个随机动作
action = env.action_space.sample()
print("\n[步骤3] 采样随机动作")
print(f"动作形状: {action.shape}")
print(f"动作值范围: [{action.min():.3f}, {action.max():.3f}]")

# 执行一步
next_state, reward, done, info = env.step(action)
print("\n[步骤4] 执行环境step")
print(f"Reward: {reward:.2f}")
print(f"Done: {done}")

# 验证info字典内容
print("\n" + "=" * 80)
print("Info字典内容分析")
print("=" * 80)

print("\n[Info Keys]")
print(f"包含的键: {list(info.keys())}")

print("\n[dispatch_orders 分析]")
dispatch_orders = info['dispatch_orders']
print(f"类型: {type(dispatch_orders)}")
print(f"形状: {np.array(dispatch_orders).shape}")
print(f"数据类型: {np.array(dispatch_orders).dtype}")
print(f"值范围: [{np.array(dispatch_orders).min()}, {np.array(dispatch_orders).max()}]")
print(f"总调度订单数: {np.sum(dispatch_orders)}")
print("\n完整矩阵:")
print(dispatch_orders)

print("\n[demand_orders 分析]")
demand_orders = info['demand_orders']
print(f"类型: {type(demand_orders)}")
print(f"形状: {np.array(demand_orders).shape}")
print(f"数据类型: {np.array(demand_orders).dtype}")
print(f"值范围: [{np.array(demand_orders).min()}, {np.array(demand_orders).max()}]")
print(f"总需求订单数: {np.sum(demand_orders)}")
print("\n完整矩阵:")
print(demand_orders)

print("\n[其他指标]")
print(f"rld (调度损失): {info['rld']:.2f}")
print(f"svc (服务率): {info['svc']:.4f}")
print(f"srd (服务距离): {info['srd']:.2f}")

# 验证矩阵维度是否符合预期 (ZONE_NUM x ZONE_NUM)
expected_shape = (4, 4)  # ZONE_NUM = 4
print("\n" + "=" * 80)
print("验证结果")
print("=" * 80)

dispatch_shape = np.array(dispatch_orders).shape
demand_shape = np.array(demand_orders).shape

print(f"\n预期形状: {expected_shape}")
dispatch_correct = "PASS" if dispatch_shape == expected_shape else "FAIL"
demand_correct = "PASS" if demand_shape == expected_shape else "FAIL"
print(f"dispatch_orders 形状: {dispatch_shape} - {dispatch_correct}")
print(f"demand_orders 形状: {demand_shape} - {demand_correct}")

# 验证矩阵含义
print("\n[矩阵含义解释]")
print("dispatch_orders[i][j]: 从区域i调度到区域j的车辆数")
print("demand_orders[i][j]: 从区域i到区域j的需求订单数")
print("\n示例:")
print(f"dispatch_orders[0][1] = {dispatch_orders[0][1]} (从区域0调度到区域1的车辆数)")
print(f"demand_orders[0][1] = {demand_orders[0][1]} (从区域0到区域1的需求订单数)")

print("\n" + "=" * 80)
print("Task 1 完成")
print("=" * 80)
