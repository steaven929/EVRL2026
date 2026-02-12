"""
Analyze dispatch and idle vehicles per timestep
Extract data from RL-DDPG_actions.txt
"""

import re
import sys
import io

# Set UTF-8 encoding for output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 读取动作文件
with open('action/RL-DDPG_actions.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# 解析每个时间步
lines = content.split('\n')
current_step = -1
step_data = {}

for line in lines:
    # 检测时间步标题
    if '时间步' in line and '=' not in line:
        match = re.search(r'时间步 (\d+)', line)
        if match:
            current_step = int(match.group(1))
            step_data[current_step] = {
                'service': [],
                'dispatch': [],
                'charging': [],
                'charge_prefs': []
            }

    # 解析服务订单
    elif current_step >= 0 and '区域' in line and '→' in line and '【' not in line:
        match = re.search(r'区域(\d+)\s*→\s*区域(\d+):\s*(\d+)辆', line)
        if match:
            from_z, to_z, count = match.groups()
            # 判断是服务还是调度（根据上下文）
            if '【服务订单】' in '\n'.join(lines[max(0, lines.index(line)-5):lines.index(line)]):
                step_data[current_step]['service'].append((int(from_z), int(to_z), int(count)))
            elif '【调度动作】' in '\n'.join(lines[max(0, lines.index(line)-5):lines.index(line)]):
                step_data[current_step]['dispatch'].append((int(from_z), int(to_z), int(count)))

    # 解析充电动作
    elif current_step >= 0 and '区域' in line and ':' in line and '辆' in line and '→' not in line:
        match = re.search(r'区域(\d+):\s*(\d+)辆', line)
        if match:
            zone, count = match.groups()
            step_data[current_step]['charging'].append((int(zone), int(count)))

    # 解析动作向量
    elif current_step >= 0 and line.strip().startswith('[') and 'e-' in line:
        # 提取充电偏好值（最后4个值）
        values_str = line.strip()[1:-1]  # 去掉方括号
        values = [float(x.strip()) for x in values_str.split(',')]
        if len(values) == 12:
            step_data[current_step]['charge_prefs'] = values[8:12]

print("="*80)
print("每个时间步的调度车辆数和空闲车辆推断")
print("="*80)
print()

# 系统配置
TOTAL_VEHICLES = 18
INITIAL_DISTRIBUTION = [5, 4, 5, 4]  # 区域0-3的初始车辆数

print(f"系统配置：")
print(f"  总车辆数: {TOTAL_VEHICLES}")
print(f"  初始分布: 区域0={INITIAL_DISTRIBUTION[0]}, 区域1={INITIAL_DISTRIBUTION[1]}, "
      f"区域2={INITIAL_DISTRIBUTION[2]}, 区域3={INITIAL_DISTRIBUTION[3]}")
print()
print("-"*80)

total_dispatch_count = 0
total_service_count = 0
total_charging_count = 0

for step_num in sorted(step_data.keys()):
    data = step_data[step_num]

    print(f"\n时间步 {step_num}:")
    print("-"*40)

    service_count = sum(count for _, _, count in data['service'])
    dispatch_count = sum(count for _, _, count in data['dispatch'])
    charging_count = sum(count for _, count in data['charging'])

    total_service_count += service_count
    total_dispatch_count += dispatch_count
    total_charging_count += charging_count

    # 输出统计
    print(f"  服务订单: {service_count}辆")
    if data['service']:
        for from_zone, to_zone, count in data['service']:
            print(f"    区域{from_zone} → 区域{to_zone}: {count}辆")

    print(f"  调度动作: {dispatch_count}辆")
    if data['dispatch']:
        for from_zone, to_zone, count in data['dispatch']:
            print(f"    区域{from_zone} → 区域{to_zone}: {count}辆")

    print(f"  充电动作: {charging_count}辆")
    if data['charging']:
        for zone, count in data['charging']:
            print(f"    区域{zone}: {count}辆")

    # 推断空闲车辆数
    estimated_busy = dispatch_count + service_count
    estimated_idle = TOTAL_VEHICLES - estimated_busy

    print(f"  估计忙碌车辆: {estimated_busy}辆 (调度{dispatch_count} + 服务{service_count})")
    print(f"  估计空闲车辆: {estimated_idle}辆")

    # 分析充电偏好值
    if data['charge_prefs']:
        print(f"  充电偏好值: ", end="")
        for z, pref in enumerate(data['charge_prefs']):
            if pref > 1e-6:
                print(f"区域{z}={pref:.4f} ", end="")
            else:
                print(f"区域{z}={pref:.2e} ", end="")
        print()

        # 分析为什么没有充电
        if charging_count == 0 and any(p > 0.5 for p in data['charge_prefs']):
            high_pref_zones = [z for z, p in enumerate(data['charge_prefs']) if p > 0.5]
            print(f"  ⚠️ 警告: 区域{high_pref_zones}充电偏好>0.5但未充电")
            print(f"     可能原因: 这些区域没有空闲车辆（都被调度走了）")

print()
print("="*80)
print("总体统计:")
print("-"*80)
print(f"  总服务订单: {total_service_count}辆次")
print(f"  总调度动作: {total_dispatch_count}辆次")
print(f"  总充电次数: {total_charging_count}辆次")
print(f"  平均每步调度: {total_dispatch_count/12:.1f}辆")
print(f"  平均每步服务: {total_service_count/12:.1f}辆")
print(f"  平均每步充电: {total_charging_count/12:.2f}辆")
print()

# 关键洞察
print("="*80)
print("关键洞察:")
print("-"*80)
print(f"1. 调度强度很高: 平均每步调度{total_dispatch_count/12:.1f}辆车")
print(f"2. 调度后空闲车辆极少: 估计每个区域平均只有0-2辆空闲车辆")
print(f"3. 映射公式的问题:")
print(f"   - 当空闲车辆=0时，无论充电偏好多高，都无法充电")
print(f"   - 当空闲车辆=1时，充电偏好需要≥0.5才能充电1辆")
print(f"   - 当空闲车辆=2时，充电偏好0.25→0辆，0.75→2辆（非线性）")
print(f"4. 这就是为什么只有5次充电的根本原因！")
print("="*80)
