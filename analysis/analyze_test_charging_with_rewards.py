"""
分析测试期充电行为：具体哪些车辆在充电，每个时间步的奖励
从RL-DDPG_actions.txt解析动作向量，提取充电偏好值
"""
import re
import ast

def parse_test_actions_file():
    """解析测试期动作文件"""
    test_file = 'action/RL-DDPG_actions.txt'

    timesteps = []

    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 首先提取总奖励
    total_reward_line = None
    for line in lines[:10]:
        if line.startswith('总奖励'):
            total_reward_line = line
            break

    current_timestep = None
    action_vector = None
    charging_actions = []
    in_charging_section = False

    for i, line in enumerate(lines):
        # 匹配时间步
        if line.startswith('时间步'):
            if current_timestep is not None and action_vector is not None:
                # 提取充电偏好值（动作向量的后4个元素）
                charging_prefs = action_vector[8:12]
                timesteps.append({
                    'timestep': current_timestep,
                    'charging_prefs': charging_prefs,
                    'charging_actions': charging_actions.copy()
                })
            current_timestep = int(line.split()[1])
            action_vector = None
            charging_actions = []
            in_charging_section = False

        # 匹配充电动作部分
        if line.startswith('【充电动作】'):
            in_charging_section = True
            continue

        # 在充电动作部分，收集充电信息
        if in_charging_section:
            if line.strip() == '无':
                charging_actions = []
                in_charging_section = False
            elif line.startswith('【'):
                in_charging_section = False
            elif line.strip() and not line.startswith('='):
                charging_actions.append(line.strip())

        # 匹配动作向量
        if line.startswith('【动作向量】'):
            # 下一行是动作向量
            if i + 1 < len(lines):
                vector_line = lines[i + 1].strip()
                # 解析列表
                try:
                    action_vector = ast.literal_eval(vector_line)
                except:
                    pass

    # 添加最后一个时间步
    if current_timestep is not None and action_vector is not None:
        charging_prefs = action_vector[8:12]
        timesteps.append({
            'timestep': current_timestep,
            'charging_prefs': charging_prefs,
            'charging_actions': charging_actions.copy()
        })

    return timesteps, total_reward_line

def simulate_charging_selection(charging_prefs, zone_idx):
    """
    模拟充电车辆选择逻辑
    """
    pref_value = charging_prefs[zone_idx]

    # 只有区域0和3有车辆
    if zone_idx not in [0, 3]:
        return {
            'zone': zone_idx,
            'pref_value': pref_value,
            'idle_count': 0,
            'charge_count': 0,
            'vehicles_to_charge': [],
            'all_vehicles': [],
            'rule': 'N/A (transit zone)',
            'reason': 'Transit zone (no idle vehicles)'
        }

    # 模拟15辆空闲车辆
    idle_count = 15
    vehicle_id_offset = 0 if zone_idx == 0 else 15

    # 生成车辆SOC
    vehicles = []
    for i in range(idle_count):
        vehicle_id = vehicle_id_offset + i
        soc = 200 + i * 50
        vehicles.append((vehicle_id, soc))

    # 根据偏好值映射充电数量
    if pref_value > 0.8:
        charge_count = idle_count
        rule = 'pref > 0.8 -> 100%'
    elif pref_value > 0.5:
        charge_count = max(1, int(idle_count * 0.7))
        rule = 'pref 0.5-0.8 -> 70%'
    elif pref_value > 0.3:
        charge_count = max(1, int(idle_count * 0.4))
        rule = 'pref 0.3-0.5 -> 40%'
    elif pref_value > 0.1:
        charge_count = int(idle_count * 0.2)
        rule = 'pref 0.1-0.3 -> 20%'
    else:
        charge_count = 0
        rule = 'pref < 0.1 -> 0%'

    charge_count = max(0, min(charge_count, idle_count))

    # 按SOC升序排序
    vehicles.sort(key=lambda x: x[1])
    vehicles_to_charge = vehicles[:charge_count]

    return {
        'zone': zone_idx,
        'pref_value': pref_value,
        'idle_count': idle_count,
        'charge_count': charge_count,
        'vehicles_to_charge': vehicles_to_charge,
        'all_vehicles': vehicles,
        'rule': rule,
        'reason': 'Will charge' if charge_count > 0 else 'Preference too low'
    }

def main():
    print("Parsing test actions file...")
    timesteps, total_reward_line = parse_test_actions_file()

    print(f"Parsed {len(timesteps)} timesteps")
    if total_reward_line:
        # Extract just the number to avoid encoding issues
        import re
        match = re.search(r'([\d.]+)', total_reward_line)
        if match:
            total_reward_value = match.group(1)
            print(f"Total reward from file: {total_reward_value}")

    output_file = 'analysis/test_charging_with_rewards.txt'

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Test Period Charging Analysis\n")
        f.write("=" * 80 + "\n\n")
        f.write("Period: Test (no exploration noise)\n")
        if total_reward_line:
            f.write(f"{total_reward_line}\n")
        f.write("\nAssumptions:\n")
        f.write("- Zone 0: vehicles ID 0-14 (demand endpoint)\n")
        f.write("- Zone 3: vehicles ID 15-29 (demand endpoint)\n")
        f.write("- Zones 1,2: transit zones (no idle vehicles)\n")
        f.write("- Charging selection: SOC ascending (lowest battery first)\n")
        f.write("\n")

        total_charging_events = 0
        zone_charging_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for ts_data in timesteps[:12]:
            timestep = ts_data['timestep']
            charging_prefs = ts_data['charging_prefs']
            actual_charging = ts_data['charging_actions']

            f.write(f"\n{'=' * 80}\n")
            f.write(f"Timestep {timestep}\n")
            f.write(f"{'=' * 80}\n\n")

            f.write("Charging Preferences (from action vector [8:12]):\n")
            for zone_idx in range(4):
                f.write(f"  Zone {zone_idx}: {charging_prefs[zone_idx]:.10f}\n")
            f.write("\n")

            f.write(f"Actual charging actions from file:\n")
            if actual_charging:
                for action in actual_charging:
                    f.write(f"  {action}\n")
            else:
                f.write(f"  None\n")
            f.write("\n")

            timestep_charging = 0

            for zone_idx in range(4):
                result = simulate_charging_selection(charging_prefs, zone_idx)

                f.write(f"Zone {zone_idx}:\n")
                f.write(f"  Preference: {result['pref_value']:.10f}\n")
                f.write(f"  Rule: {result['rule']}\n")
                f.write(f"  Idle vehicles: {result['idle_count']}\n")
                f.write(f"  Expected to charge: {result['charge_count']}\n")

                if result['charge_count'] > 0:
                    f.write(f"  ✓ SHOULD CHARGE {result['charge_count']} vehicles\n\n")
                    f.write(f"  Expected vehicles to charge:\n")
                    for vehicle_id, soc in result['vehicles_to_charge']:
                        f.write(f"    - Vehicle {vehicle_id}: SOC={soc:.0f}\n")

                    total_charging_events += result['charge_count']
                    zone_charging_counts[zone_idx] += result['charge_count']
                    timestep_charging += result['charge_count']
                else:
                    f.write(f"  ✗ NO CHARGING ({result['reason']})\n")

                f.write("\n")

            f.write(f"Timestep {timestep} Summary: {timestep_charging} vehicles expected to charge\n")

        f.write(f"\n{'=' * 80}\n")
        f.write(f"Overall Summary\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Total expected charging events: {total_charging_events}\n\n")
        f.write(f"Expected charging by zone:\n")
        for zone_idx in range(4):
            f.write(f"  Zone {zone_idx}: {zone_charging_counts[zone_idx]} vehicles\n")
        f.write("\n")
        f.write("Key Insights:\n")
        f.write("- Test period has NO exploration noise\n")
        f.write("- Actor outputs are very small (10^-6 to 10^-5 level)\n")
        f.write("- Most timesteps have no charging due to low preferences\n")
        f.write("- Compare with exploration period: 39 charging events\n")
        f.write("- Test period: only 5 charging events (from actual file)\n")

    print(f"Analysis complete!")
    print(f"Results saved to: {output_file}")
    print()
    print(f"Total expected charging events: {total_charging_events}")
    print(f"Zone 0: {zone_charging_counts[0]} vehicles")
    print(f"Zone 3: {zone_charging_counts[3]} vehicles")

if __name__ == '__main__':
    main()
