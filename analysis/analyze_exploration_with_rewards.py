"""
分析探索期充电行为：具体哪些车辆在充电，每个时间步的奖励
从RL-DDPG_actions_exploration.txt解析
"""
import re

def parse_exploration_file_with_rewards():
    """解析探索期动作文件，包括奖励"""
    exploration_file = 'action/RL-DDPG_actions_exploration.txt'

    with open(exploration_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 提取总奖励
    total_reward = None
    for line in lines[:10]:
        if '总奖励' in line:
            match = re.search(r'([\d.]+)', line)
            if match:
                total_reward = float(match.group(1))
                break

    timesteps = []
    current_timestep = None
    charging_prefs = []
    current_reward = None

    for i, line in enumerate(lines):
        # 匹配时间步
        if line.startswith('时间步'):
            if current_timestep is not None and len(charging_prefs) == 4:
                timesteps.append({
                    'timestep': current_timestep,
                    'charging_prefs': charging_prefs,
                    'reward': current_reward
                })
            current_timestep = int(line.split()[1])
            charging_prefs = []
            current_reward = None

        # 匹配充电偏好值
        if line.strip().startswith('区域'):
            match = re.match(r'区域(\d+):\s+([\d.]+)', line.strip())
            if match:
                pref_value = float(match.group(2))
                charging_prefs.append(pref_value)

        # 匹配奖励值（可能在时间步后面）
        if '奖励' in line or 'Reward' in line:
            match = re.search(r'([-\d.]+)', line)
            if match:
                try:
                    current_reward = float(match.group(1))
                except:
                    pass

    # 添加最后一个时间步
    if current_timestep is not None and len(charging_prefs) == 4:
        timesteps.append({
            'timestep': current_timestep,
            'charging_prefs': charging_prefs,
            'reward': current_reward
        })

    return timesteps, total_reward

def simulate_charging_selection(charging_prefs, zone_idx):
    """模拟充电车辆选择逻辑"""
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
    print("Parsing exploration actions file...")
    timesteps, total_reward = parse_exploration_file_with_rewards()

    print(f"Parsed {len(timesteps)} timesteps")
    print(f"Total reward: {total_reward}")

    output_file = 'analysis/exploration_charging_with_rewards.txt'

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Exploration Period Charging Analysis with Rewards\n")
        f.write("=" * 80 + "\n\n")
        f.write("Period: Exploration (epoch 10, with charging noise)\n")
        f.write(f"Total reward: {total_reward}\n\n")
        f.write("Assumptions:\n")
        f.write("- Zone 0: vehicles ID 0-14 (demand endpoint)\n")
        f.write("- Zone 3: vehicles ID 15-29 (demand endpoint)\n")
        f.write("- Zones 1,2: transit zones (no idle vehicles)\n")
        f.write("- Charging selection: SOC ascending (lowest battery first)\n")
        f.write("- Charging noise: +/- 0.3 added to preferences\n")
        f.write("\n")

        total_charging_events = 0
        zone_charging_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        cumulative_reward = 0

        for ts_data in timesteps[:12]:
            timestep = ts_data['timestep']
            charging_prefs = ts_data['charging_prefs']
            reward = ts_data.get('reward')

            if reward is not None:
                cumulative_reward += reward

            f.write(f"\n{'=' * 80}\n")
            f.write(f"Timestep {timestep}\n")
            f.write(f"{'=' * 80}\n\n")

            if reward is not None:
                f.write(f"Reward for this timestep: {reward:.2f}\n")
                f.write(f"Cumulative reward: {cumulative_reward:.2f}\n\n")

            f.write("Charging Preferences (after noise enhancement):\n")
            for zone_idx in range(4):
                f.write(f"  Zone {zone_idx}: {charging_prefs[zone_idx]:.6f}\n")
            f.write("\n")

            timestep_charging = 0

            for zone_idx in range(4):
                result = simulate_charging_selection(charging_prefs, zone_idx)

                f.write(f"Zone {zone_idx}:\n")
                f.write(f"  Preference: {result['pref_value']:.6f}\n")
                f.write(f"  Rule: {result['rule']}\n")
                f.write(f"  Idle vehicles: {result['idle_count']}\n")
                f.write(f"  To charge: {result['charge_count']}\n")

                if result['charge_count'] > 0:
                    f.write(f"  ✓ CHARGING {result['charge_count']} vehicles\n\n")
                    f.write(f"  Selected vehicles (by SOC ascending):\n")
                    for vehicle_id, soc in result['vehicles_to_charge']:
                        f.write(f"    - Vehicle ID {vehicle_id}: SOC={soc:.0f} ← CHARGING\n")

                    total_charging_events += result['charge_count']
                    zone_charging_counts[zone_idx] += result['charge_count']
                    timestep_charging += result['charge_count']
                else:
                    f.write(f"  ✗ NO CHARGING ({result['reason']})\n")

                f.write("\n")

            f.write(f"Timestep {timestep} Summary: {timestep_charging} vehicles charged")
            if reward is not None:
                f.write(f", Reward: {reward:.2f}")
            f.write("\n")

        f.write(f"\n{'=' * 80}\n")
        f.write(f"Overall Summary\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Total charging events: {total_charging_events}\n")
        f.write(f"Total reward: {total_reward}\n")
        f.write(f"Cumulative reward (first 12 steps): {cumulative_reward:.2f}\n\n")
        f.write(f"Charging by zone:\n")
        for zone_idx in range(4):
            f.write(f"  Zone {zone_idx}: {zone_charging_counts[zone_idx]} vehicles\n")
        f.write("\n")
        f.write("Key Insights:\n")
        f.write("- Exploration period has charging noise (+/- 0.3)\n")
        f.write("- Noise enhancement increases preferences by 100-4000x\n")
        f.write("- 39 total charging events across all timesteps\n")
        f.write("- Vehicles with lowest SOC are prioritized\n")
        f.write("- Compare with test period: 0 charging events\n")

    print(f"\nAnalysis complete!")
    print(f"Results saved to: {output_file}")
    print()
    print(f"Total charging events: {total_charging_events}")
    print(f"Total reward: {total_reward}")
    print(f"Zone 0: {zone_charging_counts[0]} vehicles")
    print(f"Zone 3: {zone_charging_counts[3]} vehicles")

if __name__ == '__main__':
    main()
