"""
追踪充电行为：具体哪些车辆ID在充电，它们的SOC状态
"""
import sys
import os
import re

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specific.dclily_elec import dclily
from core.algorithm import DDPG
import numpy as np
import torch

def load_trained_model(checkpoint_path):
    """加载训练好的模型"""
    env = dclily()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    cfg = {
        'actor_lr': 5e-6,
        'critic_lr': 5e-5,
        'gamma': 0.95,
        'tau': 5e-4,
        'memory_capacity': 16000,
        'batch_size': 128,
        'actor_hidden_dim': 256,
        'critic_hidden_dim': 256,
        'actor_embed_dim': 8,
        'critic_embed_dim': 8,
        'device': 'cpu'
    }

    agent = DDPG(state_dim, action_dim, cfg)
    agent.actor.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    agent.actor.eval()

    return agent, env

def get_vehicle_info(env):
    """获取所有车辆的详细信息"""
    vehicle_info = []
    for vehicle_id in range(len(env.car)):
        zone = env.car[vehicle_id][0]
        soc = env.car[vehicle_id][1]
        distance = env.car[vehicle_id][2]
        charger_id = env.car[vehicle_id][3]

        # 判断车辆状态
        is_idle = (distance == 0 and charger_id == -1)
        is_charging = (charger_id != -1)
        is_traveling = (distance > 0)

        vehicle_info.append({
            'id': vehicle_id,
            'zone': zone,
            'soc': soc,
            'distance': distance,
            'charger_id': charger_id,
            'is_idle': is_idle,
            'is_charging': is_charging,
            'is_traveling': is_traveling
        })

    return vehicle_info

def analyze_charging_decision(env, action, timestep):
    """分析充电决策：哪些车辆会被选中充电"""
    charging_prefs = action[8:12]  # 4个区域的充电偏好

    results = []

    for zone_idx in range(4):
        pref_value = charging_prefs[zone_idx]

        # 获取该区域的空闲车辆
        idle_vehicles = []
        for vehicle_id in range(len(env.car)):
            if env.car[vehicle_id][0] == zone_idx:  # 车辆在该区域
                distance = env.car[vehicle_id][2]
                charger_id = env.car[vehicle_id][3]
                if distance == 0 and charger_id == -1:  # 空闲车辆
                    soc = env.car[vehicle_id][1]
                    idle_vehicles.append((vehicle_id, soc))

        if len(idle_vehicles) == 0:
            results.append({
                'zone': zone_idx,
                'pref_value': pref_value,
                'idle_count': 0,
                'charge_count': 0,
                'vehicles_to_charge': [],
                'reason': 'No idle vehicles'
            })
            continue

        # 根据偏好值映射充电数量（复制dclily_elec.py的逻辑）
        max_chargeable = len(idle_vehicles)

        if pref_value > 0.8:
            charge_count = max_chargeable
            rule = 'pref > 0.8 -> 100%'
        elif pref_value > 0.5:
            charge_count = max(1, int(max_chargeable * 0.7))
            rule = 'pref 0.5-0.8 -> 70%'
        elif pref_value > 0.3:
            charge_count = max(1, int(max_chargeable * 0.4))
            rule = 'pref 0.3-0.5 -> 40%'
        elif pref_value > 0.1:
            charge_count = int(max_chargeable * 0.2)
            rule = 'pref 0.1-0.3 -> 20%'
        else:
            charge_count = 0
            rule = 'pref < 0.1 -> 0%'

        charge_count = max(0, min(charge_count, max_chargeable))

        # 按SOC升序排序，选择电量最低的车辆
        idle_vehicles.sort(key=lambda x: x[1])
        vehicles_to_charge = idle_vehicles[:charge_count]

        results.append({
            'zone': zone_idx,
            'pref_value': pref_value,
            'idle_count': len(idle_vehicles),
            'charge_count': charge_count,
            'vehicles_to_charge': vehicles_to_charge,
            'all_idle_vehicles': idle_vehicles,
            'rule': rule,
            'reason': 'Will charge' if charge_count > 0 else 'Preference too low'
        })

    return results

def run_analysis():
    """运行分析"""
    checkpoint_path = 'checkpoint/RL-DDPG_actor.pth'

    print("Loading trained model...")
    agent, env = load_trained_model(checkpoint_path)

    print("Running test episode with detailed vehicle tracking...")
    print()

    # 重置环境
    state = env.reset()
    done = False
    timestep = 0
    total_reward = 0

    output_file = 'analysis/charging_vehicles_trace.txt'

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Charging Vehicle Tracking Analysis\n")
        f.write("=" * 80 + "\n\n")

        while not done and timestep < 12:  # 只分析前12个时间步
            # 获取Actor动作（无噪声）
            action = agent.choose_action(state, train=False)

            # 在执行动作前分析充电决策
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Timestep {timestep}\n")
            f.write(f"{'=' * 80}\n\n")

            # 显示充电偏好值
            charging_prefs = action[8:12]
            f.write("Charging Preferences:\n")
            for zone_idx in range(4):
                f.write(f"  Zone {zone_idx}: {charging_prefs[zone_idx]:.6f}\n")
            f.write("\n")

            # 分析充电决策
            charging_decisions = analyze_charging_decision(env, action, timestep)

            for decision in charging_decisions:
                zone = decision['zone']
                f.write(f"Zone {zone}:\n")
                f.write(f"  Preference value: {decision['pref_value']:.6f}\n")
                f.write(f"  Mapping rule: {decision['rule']}\n")
                f.write(f"  Idle vehicles: {decision['idle_count']}\n")
                f.write(f"  Vehicles to charge: {decision['charge_count']}\n")

                if decision['charge_count'] > 0:
                    f.write(f"  ✓ WILL CHARGE {decision['charge_count']} vehicles\n")
                    f.write(f"  Selected vehicles (by SOC ascending):\n")
                    for vehicle_id, soc in decision['vehicles_to_charge']:
                        f.write(f"    - Vehicle ID {vehicle_id}: SOC = {soc:.2f}\n")

                    # 显示所有空闲车辆的SOC分布
                    f.write(f"  All idle vehicles in zone {zone}:\n")
                    for vehicle_id, soc in decision['all_idle_vehicles']:
                        selected = "← SELECTED" if (vehicle_id, soc) in decision['vehicles_to_charge'] else ""
                        f.write(f"    - Vehicle ID {vehicle_id}: SOC = {soc:.2f} {selected}\n")
                else:
                    f.write(f"  ✗ NO CHARGING ({decision['reason']})\n")

                f.write("\n")

            # 执行动作
            next_state, reward, done = env.step(action)
            total_reward += reward

            state = next_state
            timestep += 1

        f.write(f"\n{'=' * 80}\n")
        f.write(f"Summary\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Total timesteps analyzed: {timestep}\n")
        f.write(f"Total reward: {total_reward:.2f}\n")

    print(f"Analysis complete!")
    print(f"Results saved to: {output_file}")
    print()
    print("Key findings:")
    print("- Detailed vehicle IDs and SOC values for each charging decision")
    print("- Shows which vehicles are selected vs not selected")
    print("- Tracks SOC-based priority ordering")

if __name__ == '__main__':
    run_analysis()
