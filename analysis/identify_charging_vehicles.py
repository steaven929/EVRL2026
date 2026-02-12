"""
识别具体哪些车辆ID在充电
基于探索期动作文件，结合环境状态，追踪充电车辆
"""
import re

def parse_exploration_file():
    """解析探索期动作文件"""
    exploration_file = 'action/RL-DDPG_actions_exploration.txt'

    timesteps = []

    with open(exploration_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_timestep = None
    charging_prefs = []

    for i, line in enumerate(lines):
        # 匹配时间步
        if line.startswith('时间步'):
            if current_timestep is not None and len(charging_prefs) == 4:
                timesteps.append({
                    'timestep': current_timestep,
                    'charging_prefs': charging_prefs
                })
            current_timestep = int(line.split()[1])
            charging_prefs = []

        # 匹配充电偏好值
        if line.strip().startswith('区域'):
            match = re.match(r'区域(\d+):\s+([\d.]+)', line.strip())
            if match:
                zone_idx = int(match.group(1))
                pref_value = float(match.group(2))
                charging_prefs.append(pref_value)

    # 添加最后一个时间步
    if current_timestep is not None and len(charging_prefs) == 4:
        timesteps.append({
            'timestep': current_timestep,
            'charging_prefs': charging_prefs
        })

    return timesteps

def simulate_charging_selection(charging_prefs, zone_idx, num_vehicles=30):
    """
    模拟充电车辆选择逻辑
    假设：
    - 每个区域有15辆车（区域0和3）
    - 车辆SOC从200-900均匀分布
    - 按SOC升序选择充电车辆
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

    # 模拟15辆空闲车辆，SOC从200-900
    idle_count = 15
    # 假设车辆ID：区域0是0-14，区域3是15-29
    vehicle_id_offset = 0 if zone_idx == 0 else 15

    # 生成车辆SOC（简化：均匀分布）
    vehicles = []
    for i in range(idle_count):
        vehicle_id = vehicle_id_offset + i
        soc = 200 + i * 50  # 200, 250, 300, ..., 900
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

    # 按SOC升序排序，选择电量最低的车辆
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
    print("Parsing exploration actions file...")
    timesteps = parse_exploration_file()

    print(f"Parsed {len(timesteps)} timesteps")

    output_file = 'analysis/charging_vehicles_identified.txt'

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Charging Vehicle Identification Analysis\n")
        f.write("=" * 80 + "\n\n")
        f.write("Assumptions:\n")
        f.write("- Zone 0 has vehicles ID 0-14 (idle at zone 0)\n")
        f.write("- Zone 3 has vehicles ID 15-29 (idle at zone 3)\n")
        f.write("- Zones 1,2 are transit zones (no idle vehicles)\n")
        f.write("- Vehicle SOC ranges from 200-900 (uniformly distributed)\n")
        f.write("- Charging selection: SOC ascending order (lowest battery first)\n")
        f.write("\n")

        total_charging_events = 0
        zone_charging_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for ts_data in timesteps[:12]:  # 只分析前12个时间步
            timestep = ts_data['timestep']
            charging_prefs = ts_data['charging_prefs']

            f.write(f"\n{'=' * 80}\n")
            f.write(f"Timestep {timestep}\n")
            f.write(f"{'=' * 80}\n\n")

            f.write("Charging Preferences:\n")
            for zone_idx in range(4):
                f.write(f"  Zone {zone_idx}: {charging_prefs[zone_idx]:.6f}\n")
            f.write("\n")

            for zone_idx in range(4):
                result = simulate_charging_selection(charging_prefs, zone_idx)

                f.write(f"Zone {zone_idx}:\n")
                f.write(f"  Preference value: {result['pref_value']:.6f}\n")
                f.write(f"  Mapping rule: {result['rule']}\n")
                f.write(f"  Idle vehicles: {result['idle_count']}\n")
                f.write(f"  Vehicles to charge: {result['charge_count']}\n")

                if result['charge_count'] > 0:
                    f.write(f"  ✓ WILL CHARGE {result['charge_count']} vehicles\n\n")
                    f.write(f"  Selected vehicles (by SOC ascending):\n")
                    for vehicle_id, soc in result['vehicles_to_charge']:
                        f.write(f"    - Vehicle ID {vehicle_id}: SOC = {soc:.0f} ← SELECTED FOR CHARGING\n")

                    f.write(f"\n  All idle vehicles in zone {zone_idx} (for reference):\n")
                    for vehicle_id, soc in result['all_vehicles']:
                        selected = " ← SELECTED" if (vehicle_id, soc) in result['vehicles_to_charge'] else ""
                        f.write(f"    - Vehicle ID {vehicle_id}: SOC = {soc:.0f}{selected}\n")

                    total_charging_events += result['charge_count']
                    zone_charging_counts[zone_idx] += result['charge_count']
                else:
                    f.write(f"  ✗ NO CHARGING ({result['reason']})\n")

                f.write("\n")

        f.write(f"\n{'=' * 80}\n")
        f.write(f"Summary\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Total charging events: {total_charging_events}\n")
        f.write(f"Charging events by zone:\n")
        for zone_idx in range(4):
            f.write(f"  Zone {zone_idx}: {zone_charging_counts[zone_idx]} vehicles charged\n")
        f.write("\n")
        f.write("Key Insights:\n")
        f.write("- Vehicles with lowest SOC are prioritized for charging\n")
        f.write("- Zone 0 and 3 have idle vehicles available for charging\n")
        f.write("- Zones 1 and 2 are transit zones with no idle vehicles\n")
        f.write("- Charging threshold 0.1 may be too high for some zones\n")

    print(f"Analysis complete!")
    print(f"Results saved to: {output_file}")
    print()
    print(f"Total charging events: {total_charging_events}")
    print(f"Zone 0: {zone_charging_counts[0]} vehicles charged")
    print(f"Zone 3: {zone_charging_counts[3]} vehicles charged")

if __name__ == '__main__':
    main()
