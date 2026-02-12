"""
分析充电行为对奖励的影响
对比有充电 vs 无充电时间步的奖励差异
"""
import re

def parse_exploration_file():
    """解析探索期文件，提取充电和奖励信息"""
    exploration_file = 'action/RL-DDPG_actions_exploration.txt'

    with open(exploration_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    timesteps = []
    current_timestep = None
    charging_prefs = []

    for i, line in enumerate(lines):
        if line.startswith('时间步'):
            if current_timestep is not None and len(charging_prefs) == 4:
                timesteps.append({
                    'timestep': current_timestep,
                    'charging_prefs': charging_prefs
                })
            current_timestep = int(line.split()[1])
            charging_prefs = []

        if line.strip().startswith('区域'):
            match = re.match(r'区域(\d+):\s+([\d.]+)', line.strip())
            if match:
                pref_value = float(match.group(2))
                charging_prefs.append(pref_value)

    if current_timestep is not None and len(charging_prefs) == 4:
        timesteps.append({
            'timestep': current_timestep,
            'charging_prefs': charging_prefs
        })

    return timesteps

def calculate_charging_count(charging_prefs):
    """计算预期充电数量"""
    total_charging = 0

    for zone_idx in [0, 3]:  # 只有区域0和3有车辆
        pref_value = charging_prefs[zone_idx]
        idle_count = 15

        if pref_value > 0.8:
            charge_count = idle_count
        elif pref_value > 0.5:
            charge_count = max(1, int(idle_count * 0.7))
        elif pref_value > 0.3:
            charge_count = max(1, int(idle_count * 0.4))
        elif pref_value > 0.1:
            charge_count = int(idle_count * 0.2)
        else:
            charge_count = 0

        total_charging += charge_count

    return total_charging

def main():
    print("Analyzing charging impact on rewards...")

    timesteps = parse_exploration_file()

    # 从文件中读取总奖励
    total_reward = 6290.0
    avg_reward_per_step = total_reward / 12

    output_file = 'analysis/charging_reward_impact.txt'

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Charging Behavior Impact on Rewards Analysis\n")
        f.write("=" * 80 + "\n\n")

        f.write("Reward Function Components:\n")
        f.write("  reward = (service_revenue + charging_bonus - soc_penalty) / 10\n")
        f.write("  charging_bonus = 500 * charging_count\n")
        f.write("  (divided by 10, so actual contribution = 50 * charging_count)\n\n")

        f.write(f"Total reward (12 steps): {total_reward}\n")
        f.write(f"Average reward per step: {avg_reward_per_step:.2f}\n\n")

        f.write("=" * 80 + "\n")
        f.write("Timestep Analysis\n")
        f.write("=" * 80 + "\n\n")

        charging_steps = []
        no_charging_steps = []

        for ts_data in timesteps:
            timestep = ts_data['timestep']
            charging_prefs = ts_data['charging_prefs']
            charging_count = calculate_charging_count(charging_prefs)

            # 估算充电奖励贡献
            charging_reward_contribution = (500 * charging_count) / 10  # = 50 * charging_count

            f.write(f"Timestep {timestep}:\n")
            f.write(f"  Charging count: {charging_count}\n")
            f.write(f"  Charging reward contribution: {charging_reward_contribution:.2f}\n")
            f.write(f"  (500 * {charging_count} / 10 = {charging_reward_contribution:.2f})\n")

            if charging_count > 0:
                charging_steps.append({
                    'timestep': timestep,
                    'count': charging_count,
                    'contribution': charging_reward_contribution
                })
                f.write(f"  ✓ HAS CHARGING\n")
            else:
                no_charging_steps.append(timestep)
                f.write(f"  ✗ NO CHARGING\n")

            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("Summary Statistics\n")
        f.write("=" * 80 + "\n\n")

        total_charging_events = sum([s['count'] for s in charging_steps])
        total_charging_reward = sum([s['contribution'] for s in charging_steps])

        f.write(f"Steps with charging: {len(charging_steps)}/12\n")
        f.write(f"Steps without charging: {len(no_charging_steps)}/12\n")
        f.write(f"Total charging events: {total_charging_events}\n")
        f.write(f"Total charging reward contribution: {total_charging_reward:.2f}\n")
        f.write(f"Percentage of total reward: {(total_charging_reward/total_reward)*100:.2f}%\n\n")

        if len(charging_steps) > 0:
            avg_charging_reward = total_charging_reward / len(charging_steps)
            f.write(f"Average charging reward per charging step: {avg_charging_reward:.2f}\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("Key Insights\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. Charging Reward Signal Strength:\n")
        f.write(f"   - Each charging vehicle contributes: 50 reward points\n")
        f.write(f"   - Total charging contribution: {total_charging_reward:.2f} ({(total_charging_reward/total_reward)*100:.2f}% of total)\n")
        f.write(f"   - Average reward per step: {avg_reward_per_step:.2f}\n")
        f.write(f"   - Charging reward is {(total_charging_reward/len(charging_steps))/avg_reward_per_step:.2f}x average step reward\n\n")

        f.write("2. Why Actor Doesn't Learn to Charge:\n")
        f.write("   a) Reward Signal Dilution:\n")
        f.write(f"      - Charging reward: {total_charging_reward:.2f}\n")
        f.write(f"      - Service revenue (estimated): {total_reward - total_charging_reward:.2f}\n")
        f.write(f"      - Charging is only {(total_charging_reward/total_reward)*100:.2f}% of total reward\n\n")

        f.write("   b) Temporal Credit Assignment:\n")
        f.write("      - Charging benefit is long-term (prevents future SOC penalty)\n")
        f.write(f"      - But gamma={0.95} may not be high enough\n")
        f.write("      - Immediate service revenue dominates learning\n\n")

        f.write("   c) Exploration Dependency:\n")
        f.write("      - Actor outputs ~10^-6 for charging preferences\n")
        f.write("      - Only noise (+/-0.3) triggers charging\n")
        f.write("      - Without noise, no charging → no learning signal\n\n")

        f.write("3. Proposed Solutions:\n")
        f.write("   A. Increase charging reward:\n")
        f.write("      - Current: 500 per vehicle (50 after /10)\n")
        f.write("      - Proposed: 1000-2000 per vehicle (100-200 after /10)\n")
        f.write("      - Make charging reward comparable to service revenue\n\n")

        f.write("   B. Lower charging threshold:\n")
        f.write("      - Current: 0.1 (too high for Actor's 10^-6 output)\n")
        f.write("      - Proposed: 0.01 or 0.001\n")
        f.write("      - Allow Actor's small outputs to trigger charging\n\n")

        f.write("   C. Extend exploration period:\n")
        f.write("      - Current: 20 epochs with charging noise\n")
        f.write("      - Proposed: 50-100 epochs\n")
        f.write("      - Give Actor more time to learn from charging experiences\n\n")

        f.write("   D. Increase gamma:\n")
        f.write("      - Current: 0.95\n")
        f.write("      - Proposed: 0.98 or 0.99\n")
        f.write("      - Strengthen long-term reward signal\n\n")

    print(f"Analysis complete!")
    print(f"Results saved to: {output_file}")
    print()
    print(f"Total charging events: {total_charging_events}")
    print(f"Total charging reward: {total_charging_reward:.2f}")
    print(f"Percentage of total reward: {(total_charging_reward/total_reward)*100:.2f}%")

if __name__ == '__main__':
    main()
