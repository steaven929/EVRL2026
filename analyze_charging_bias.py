"""
Analyze why RL learned wrong zone charging preferences

Research questions:
1. Why zone 0 has high charging preference (0.78-0.99) while others are very low (<0.01)?
2. Does this preference distribution match actual charging demand?
3. Is SOC information in state space properly utilized by RL?
"""

import numpy as np
import pandas as pd
import re
import json
from pathlib import Path
import sys
import io

# Set UTF-8 encoding for output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Read config
ZONE_NUM = 4  # 4 zones from code
BATTERY_CAPACITY = 100  # Assume battery capacity is 100

print("="*80)
print("RL Charging Preference Bias Root Cause Analysis")
print("="*80)

# 1. Analyze RL charging preference distribution
print("\n[1. RL Charging Preference Analysis]")
print("-"*80)

with open('action/RL-DDPG_actions.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# Extract all action vectors
pattern = r'\[([\d\., e\-]+)\]'
matches = re.findall(pattern, content)

charging_prefs_by_zone = {z: [] for z in range(ZONE_NUM)}
for match in matches:
    values = [float(x.strip()) for x in match.split(',')]
    charge_prefs = values[8:12]
    for z in range(ZONE_NUM):
        charging_prefs_by_zone[z].append(charge_prefs[z])

print(f"{'Zone':<10} {'Mean':<15} {'Median':<15} {'Min':<15} {'Max':<15}")
print("-"*80)
for z in range(ZONE_NUM):
    prefs = charging_prefs_by_zone[z]
    print(f"Zone {z:<7} {np.mean(prefs):<15.6f} {np.median(prefs):<15.6f} "
          f"{np.min(prefs):<15.6f} {np.max(prefs):<15.6f}")

# 2. 分析各区域的实际充电需求（从车辆SOC推断）
print("\n【2. 各区域实际充电需求分析】")
print("-"*80)
print("注：需要运行环境才能获取实时车辆SOC数据")
print("这里我们从动作文件中提取充电行为统计")

# 统计实际充电次数
charging_counts = {z: 0 for z in range(ZONE_NUM)}
pattern_charge = r'【充电动作】\s+(.*?)(?=【|$)'
charge_matches = re.findall(pattern_charge, content, re.DOTALL)

for match in charge_matches:
    if '无' not in match and match.strip():
        # 提取 "区域X: Y辆"
        zone_pattern = r'区域(\d+):\s*(\d+)辆'
        zone_matches = re.findall(zone_pattern, match)
        for zone_str, count_str in zone_matches:
            zone = int(zone_str)
            count = int(count_str)
            charging_counts[zone] += count

print(f"{'区域':<10} {'实际充电次数':<15} {'充电偏好均值':<15} {'偏好/次数比':<15}")
print("-"*80)
for z in range(ZONE_NUM):
    pref_mean = np.mean(charging_prefs_by_zone[z])
    count = charging_counts[z]
    ratio = pref_mean / count if count > 0 else float('inf')
    print(f"区域{z:<8} {count:<15} {pref_mean:<15.6f} {ratio:<15.6f}")

# 3. 分析充电偏好与充电次数的相关性
print("\n【3. 充电偏好与实际充电的相关性】")
print("-"*80)

prefs = [np.mean(charging_prefs_by_zone[z]) for z in range(ZONE_NUM)]
counts = [charging_counts[z] for z in range(ZONE_NUM)]

correlation = np.corrcoef(prefs, counts)[0, 1]
print(f"相关系数: {correlation:.4f}")

if correlation > 0.8:
    print("✅ 强正相关：RL学到的偏好与实际充电行为一致")
elif correlation > 0.5:
    print("⚠️ 中等正相关：RL偏好部分反映了充电需求")
else:
    print("❌ 弱相关或负相关：RL偏好与实际需求不匹配")

# 4. 分析问题根源
print("\n【4. 问题根源诊断】")
print("-"*80)

# 检查是否存在"区域0偏好过高"的问题
zone0_pref = np.mean(charging_prefs_by_zone[0])
other_zones_pref = np.mean([np.mean(charging_prefs_by_zone[z]) for z in range(1, ZONE_NUM)])

print(f"区域0平均偏好: {zone0_pref:.6f}")
print(f"其他区域平均偏好: {other_zones_pref:.6f}")
print(f"偏好比值 (区域0/其他): {zone0_pref / other_zones_pref:.2f}x")

if zone0_pref / other_zones_pref > 100:
    print("\n❌ 严重偏差：区域0偏好是其他区域的100倍以上")
    print("\n可能原因：")
    print("1. 初始化偏差：训练早期区域0获得了正向反馈，其他区域未充分探索")
    print("2. 状态表征问题：RL无法区分各区域的真实充电需求")
    print("3. 奖励信号稀疏：充电收益延迟，RL难以学习因果关系")
    print("4. 探索不足：ε-greedy探索期太短（30轮），未覆盖所有区域")

# 5. 分析充电时机分布
print("\n【5. 充电时机分布分析】")
print("-"*80)

# 提取每个时间步的充电行为
timestep_pattern = r'时间步 (\d+).*?【充电动作】\s+(.*?)(?=【|时间步|$)'
timestep_matches = re.findall(timestep_pattern, content, re.DOTALL)

charging_by_timestep = {}
for timestep_str, charge_action in timestep_matches:
    timestep = int(timestep_str)
    if '无' not in charge_action and charge_action.strip():
        zone_pattern = r'区域(\d+):\s*(\d+)辆'
        zone_matches = re.findall(zone_pattern, charge_action)
        if zone_matches:
            charging_by_timestep[timestep] = zone_matches

print(f"{'时间步':<10} {'充电区域':<20} {'充电数量':<10}")
print("-"*80)
for timestep in sorted(charging_by_timestep.keys()):
    zones_info = charging_by_timestep[timestep]
    for zone_str, count_str in zones_info:
        print(f"{timestep:<10} 区域{zone_str:<16} {count_str}辆")

if len(charging_by_timestep) > 0:
    early_steps = sum(1 for t in charging_by_timestep.keys() if t <= 6)
    late_steps = sum(1 for t in charging_by_timestep.keys() if t > 6)
    print(f"\n前期充电（步骤0-6）: {early_steps}次")
    print(f"后期充电（步骤7-11）: {late_steps}次")

    if late_steps == 0:
        print("❌ 充电集中在前期，后期完全不充电")
        print("   可能原因：RL学到了'早期充电避免后期低电量'的策略")
        print("   但未学会'持续充电维持电量'的长期策略")

# 6. 建议的优化方向
print("\n【6. 优化建议】")
print("="*80)

print("\n基于以上分析，建议按优先级尝试以下方案：")
print("\n⭐⭐⭐ 高优先级：")
print("1. 方案E2：增强探索强度")
print("   - 延长探索期：30轮 → 100轮（覆盖更多训练阶段）")
print("   - 提升探索概率：10% → 20%（增加随机充电频率）")
print("   - 扩大充电数量：1-3辆 → 2-5辆（让RL体验更多充电场景）")
print("   - 理由：当前探索不足，导致区域1/2/3未获得充分的充电经验")

print("\n⭐⭐ 中优先级：")
print("2. 方案E3：添加充电奖励梯度")
print("   - 修改奖励函数：低电量车辆充电获得更高奖励")
print("   - 公式：charging_bonus = 500 * (1 - soc/BATTERY_CAPACITY)")
print("   - 理由：当前奖励函数对所有充电一视同仁，未引导RL优先充电低电量车辆")

print("\n⭐ 低优先级：")
print("3. 方案E1：调整映射公式")
print("   - 降低阈值或使用平方根映射")
print("   - 理由：这只是执行层优化，无法解决RL学习到错误偏好的根本问题")

print("\n" + "="*80)
print("分析完成")
print("="*80)
