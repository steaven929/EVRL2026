# -*- coding: utf-8 -*-
"""
分析RL动作的合理性
"""
import re

# 读取RL action文件
with open('action/RL-DDPG_actions.txt', 'r', encoding='utf-8') as f:
    rl_content = f.read()

# 读取最优解action文件
with open('action/global_optimal_solution_actions.txt', 'r', encoding='utf-8') as f:
    opt_content = f.read()

print("="*80)
print("RL Action Quality Analysis")
print("="*80)

# 提取每个时间步的信息
timesteps = re.findall(r'时间步 (\d+).*?【服务订单】\s*\n(.*?)【调度动作】\s*\n(.*?)【充电动作】\s*\n(.*?)【动作向量】',
                       rl_content, re.DOTALL)

print(f"\nTotal timesteps: {len(timesteps)}")

# 统计
total_service = 0
total_dispatch = 0
total_charging = 0
mutual_swaps = []
no_action_steps = []
heavy_charging_steps = []

for step_num, service, dispatch, charging in timesteps:
    step = int(step_num)

    # 统计服务订单
    service_lines = [l.strip() for l in service.split('\n') if '区域' in l and '辆' in l]
    service_count = len(service_lines)
    total_service += service_count

    # 统计调度动作
    dispatch_lines = [l.strip() for l in dispatch.split('\n') if '区域' in l and '辆' in l]
    dispatch_count = len(dispatch_lines)
    total_dispatch += dispatch_count

    # 统计充电动作
    charging_lines = [l.strip() for l in charging.split('\n') if '区域' in l and '辆' in l]
    charging_counts = {}
    for line in charging_lines:
        match = re.search(r'区域(\d+):\s*(\d+)辆', line)
        if match:
            zone = int(match.group(1))
            count = int(match.group(2))
            charging_counts[zone] = count
            total_charging += count

    # 检查互相对调
    service_dict = {}
    for line in service_lines:
        match = re.search(r'区域(\d+)\s*→\s*区域(\d+)', line)
        if match:
            from_zone = int(match.group(1))
            to_zone = int(match.group(2))
            key = (from_zone, to_zone)
            service_dict[key] = service_dict.get(key, 0) + 1

    for (f, t), count in service_dict.items():
        if (t, f) in service_dict:
            mutual_swaps.append((step, f, t, count, service_dict[(t, f)]))

    # 检查无动作时间步
    if service_count == 0 and dispatch_count == 0:
        no_action_steps.append(step)

    # 检查重度充电（>10辆）
    for zone, count in charging_counts.items():
        if count >= 10:
            heavy_charging_steps.append((step, zone, count))

print(f"\n=== Statistics ===")
print(f"Total service orders: {total_service}")
print(f"Total dispatch actions: {total_dispatch}")
print(f"Total charging actions: {total_charging}")
print(f"Average charging per step: {total_charging / len(timesteps):.2f}")

print(f"\n=== Issues Found ===")

if mutual_swaps:
    print(f"\n1. Mutual Swaps: {len(mutual_swaps)} instances")
    for step, f, t, count1, count2 in mutual_swaps[:5]:  # Show first 5
        print(f"   Step {step}: Zone {f}<->{t} ({count1} vs {count2} vehicles)")

if no_action_steps:
    print(f"\n2. No Service/Dispatch Steps: {len(no_action_steps)} steps")
    print(f"   Steps: {no_action_steps}")
    print(f"   Note: These steps only have charging actions")

if heavy_charging_steps:
    print(f"\n3. Heavy Charging (>=10 vehicles): {len(heavy_charging_steps)} instances")
    for step, zone, count in heavy_charging_steps[:10]:
        print(f"   Step {step}, Zone {zone}: {count} vehicles")

# 分析充电分布
print(f"\n=== Charging Distribution ===")
zone_charging = {0: 0, 1: 0, 2: 0, 3: 0}
for step_num, service, dispatch, charging in timesteps:
    charging_lines = [l.strip() for l in charging.split('\n') if '区域' in l and '辆' in l]
    for line in charging_lines:
        match = re.search(r'区域(\d+):\s*(\d+)辆', line)
        if match:
            zone = int(match.group(1))
            count = int(match.group(2))
            zone_charging[zone] += count

print("Charging by zone:")
for zone in sorted(zone_charging.keys()):
    print(f"  Zone {zone}: {zone_charging[zone]} times ({zone_charging[zone]/total_charging*100:.1f}%)")

# 对比最优解
print(f"\n=== Comparison with Optimal Solution ===")
opt_timesteps = re.findall(r'时间步 (\d+).*?【服务订单】\s*\n(.*?)【调度动作】\s*\n(.*?)【充电动作】\s*\n(.*?)(?=时间步|\Z)',
                           opt_content, re.DOTALL)

opt_service = 0
opt_dispatch = 0
opt_charging = 0

for match in opt_timesteps:
    if len(match) >= 4:
        step_num, service, dispatch, charging = match[0], match[1], match[2], match[3]
        service_lines = [l.strip() for l in service.split('\n') if '区域' in l and '辆' in l]
        dispatch_lines = [l.strip() for l in dispatch.split('\n') if '区域' in l and '辆' in l]
        charging_lines = [l.strip() for l in charging.split('\n') if '区域' in l and '辆' in l]

        opt_service += len(service_lines)
        opt_dispatch += len(dispatch_lines)

        for line in charging_lines:
            match_count = re.search(r'区域(\d+):\s*(\d+)辆', line)
            if match_count:
                opt_charging += int(match_count.group(2))

print(f"{'Metric':<25} {'RL':<15} {'Optimal':<15} {'Difference':<15}")
print("-"*70)
print(f"{'Service orders':<25} {total_service:<15} {opt_service:<15} {total_service - opt_service:<15}")
print(f"{'Dispatch actions':<25} {total_dispatch:<15} {opt_dispatch:<15} {total_dispatch - opt_dispatch:<15}")
print(f"{'Charging actions':<25} {total_charging:<15} {opt_charging:<15} {total_charging - opt_charging:<15}")

print(f"\n=== Recommendations ===")
if len(mutual_swaps) > 0:
    print("1. Reduce mutual swaps - they waste resources")
if total_dispatch < opt_dispatch / 2:
    print("2. Increase proactive repositioning - RL dispatches much less than optimal")
if zone_charging[1] > total_charging * 0.7:
    print("3. Balance charging across zones - too concentrated in zone 1")
if len(no_action_steps) > 3:
    print("4. Add proactive dispatch during idle periods")
