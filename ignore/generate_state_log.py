# -*- coding: utf-8 -*-
"""
生成每个时间步的状态详细日志
类似于RL-DDPG_actions.txt的格式
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import gym
from gym.envs.registration import register
import os

# 注册环境
register(id='dclily', entry_point='specific.dclily_elec:dclily')

from core.util import get_args
from core.environment import env_agent_config

def parse_state(state):
    """解析状态向量，返回各个组件"""
    parsed = {}

    # 1. OD流入需求 (8维)
    parsed['od_in'] = {
        'zone_0': state[0:2].tolist(),
        'zone_1': state[2:4].tolist(),
        'zone_2': state[4:6].tolist(),
        'zone_3': state[6:8].tolist()
    }

    # 2. OD流出需求 (8维)
    parsed['od_out'] = {
        'zone_0': state[8:10].tolist(),
        'zone_1': state[10:12].tolist(),
        'zone_2': state[12:14].tolist(),
        'zone_3': state[14:16].tolist()
    }

    # 3. 充电桩占用状态 (8维)
    parsed['charger_occupancy'] = {
        'zone_0': state[16:18].tolist(),
        'zone_1': state[18:20].tolist(),
        'zone_2': state[20:22].tolist(),
        'zone_3': state[22:24].tolist()
    }

    # 4. 充电桩数量 (8维)
    parsed['charger_count'] = {
        'zone_0': state[24:26].tolist(),
        'zone_1': state[26:28].tolist(),
        'zone_2': state[28:30].tolist(),
        'zone_3': state[30:32].tolist()
    }

    # 5. 区域平均SOC (4维)
    parsed['avg_soc'] = {
        'zone_0': int(state[32]),
        'zone_1': int(state[33]),
        'zone_2': int(state[34]),
        'zone_3': int(state[35])
    }

    # 6. 时间步信息 (2维)
    parsed['timestep_info'] = {
        'current': int(state[36]),
        'remaining': int(state[37])
    }

    return parsed

def format_state_log(timestep, state, env):
    """格式化状态日志"""
    parsed = parse_state(state)

    lines = []
    lines.append("=" * 80)
    lines.append(f"时间步 {timestep}")
    lines.append("=" * 80)
    lines.append("")

    # OD流入需求
    lines.append("【OD流入需求】(未来2个时间片)")
    for zone in range(4):
        demand = parsed['od_in'][f'zone_{zone}']
        lines.append(f"  区域{zone}: {demand} (下一步={demand[0]}, 下下步={demand[1]})")
    lines.append("")

    # OD流出需求
    lines.append("【OD流出需求】(未来2个时间片)")
    for zone in range(4):
        demand = parsed['od_out'][f'zone_{zone}']
        lines.append(f"  区域{zone}: {demand} (下一步={demand[0]}, 下下步={demand[1]})")
    lines.append("")

    # 充电桩占用状态
    lines.append("【充电桩占用状态】(档位0-3)")
    occupancy_map = {0: "0-25%", 1: "25-50%", 2: "50-75%", 3: "75-100%"}
    for zone in range(4):
        occ = parsed['charger_occupancy'][f'zone_{zone}']
        occ_str = f"{occupancy_map.get(occ[0], 'N/A')}"
        lines.append(f"  区域{zone}: {occ} → {occ_str}占用")
    lines.append("")

    # 充电桩数量
    lines.append("【充电桩数量】")
    for zone in range(4):
        count = parsed['charger_count'][f'zone_{zone}']
        lines.append(f"  区域{zone}: {count[0]}个充电桩")
    lines.append("")

    # 区域平均SOC
    lines.append("【区域平均SOC】(档位0-9, 代表0-100%)")
    for zone in range(4):
        soc_level = parsed['avg_soc'][f'zone_{zone}']
        soc_percent = f"{soc_level*10}-{(soc_level+1)*10}%"
        lines.append(f"  区域{zone}: {soc_level} → {soc_percent}电量")
    lines.append("")

    # 时间步信息
    lines.append("【时间步信息】")
    lines.append(f"  当前时间步: {parsed['timestep_info']['current']}/11")
    lines.append(f"  剩余时间步: {parsed['timestep_info']['remaining']}")
    lines.append("")

    # 车辆状态统计
    lines.append("【车辆状态统计】")
    zone_counts = [0, 0, 0, 0]
    charging_counts = [0, 0, 0, 0]
    moving_count = 0
    low_soc_count = 0
    soc_sum = [0, 0, 0, 0]
    soc_count = [0, 0, 0, 0]

    for i in range(len(env.car)):
        zone = int(env.car[i][0])
        soc = int(env.car[i][1])
        remain = int(env.car[i][2])
        charger_id = int(env.car[i][3])

        if remain > 0:
            moving_count += 1
        else:
            zone_counts[zone] += 1
            soc_sum[zone] += soc
            soc_count[zone] += 1
            if charger_id != -1:
                charging_counts[zone] += 1

        if soc < 600:  # 低于30%电量
            low_soc_count += 1

    lines.append(f"  各区域车辆数: 区域0={zone_counts[0]}, 区域1={zone_counts[1]}, 区域2={zone_counts[2]}, 区域3={zone_counts[3]}")
    lines.append(f"  各区域充电数: 区域0={charging_counts[0]}, 区域1={charging_counts[1]}, 区域2={charging_counts[2]}, 区域3={charging_counts[3]}")

    # 计算各区域实际平均SOC
    lines.append(f"  各区域实际平均SOC:")
    for zone in range(4):
        if soc_count[zone] > 0:
            avg_soc = soc_sum[zone] / soc_count[zone]
            lines.append(f"    区域{zone}: {avg_soc:.0f} ({avg_soc/2000*100:.1f}%)")
        else:
            lines.append(f"    区域{zone}: N/A (无车辆)")

    lines.append(f"  移动中车辆: {moving_count}辆")
    lines.append(f"  低电量车辆: {low_soc_count}辆 (SOC<600)")
    lines.append("")

    # 完整状态向量
    lines.append("【完整状态向量】(38维)")
    lines.append(f"  {state}")
    lines.append("")

    return "\n".join(lines)

def main():
    # 获取配置
    cfg = get_args()

    # 创建环境和智能体
    env, agent = env_agent_config(cfg)

    # 加载训练好的模型
    model_path = 'trained_model.pth'
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print(f"[INFO] 模型已从 {model_path} 加载")
    else:
        print(f"[WARNING] 未找到模型文件 {model_path}")
        return

    # 设置测试模式
    env.set_training_mode(0)

    # 运行一个episode
    state = env.reset()
    done = False
    timestep = 0

    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("RL-DDPG 策略 - 每个时间步的状态详细记录")
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append(f"总步数: 12")
    output_lines.append(f"总奖励: (将在episode结束后计算)")
    output_lines.append("")

    # 记录初始状态
    output_lines.append(format_state_log(timestep, state, env))

    total_reward = 0

    # 执行episode
    while not done:
        # 选择动作
        action = agent.predict_action(state)

        # 执行动作
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        timestep += 1

        # 记录状态
        if not done:
            output_lines.append(format_state_log(timestep, next_state, env))

        state = next_state

    # 更新总奖励
    output_lines[5] = f"总奖励: {total_reward:.2f}"

    # 保存到文件
    output_path = 'action/RL-DDPG_states.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"[INFO] 状态日志已保存到: {output_path}")
    print(f"[INFO] 总共记录了 {timestep+1} 个时间步的状态")
    print(f"[INFO] 总奖励: {total_reward:.2f}")

if __name__ == "__main__":
    main()
