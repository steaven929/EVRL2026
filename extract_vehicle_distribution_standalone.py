# -*- coding: utf-8 -*-
"""
独立运行的车辆分布提取脚本
统计每个时间步各区域的所有车辆数量（不仅是充电中的车辆）

使用方法:
python extract_vehicle_distribution_standalone.py

输出文件:
action/vehicle_distribution_per_timestep.txt
"""

import sys
import os
import gym
from gym.envs.registration import register
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 注册环境
register(
    id='dclily',
    entry_point='specific.dclily_elec:dclily',
)

from core.util import get_args
from core.environment import env_agent_config


def extract_and_save_vehicle_distribution():
    """
    运行RL策略并提取车辆分布信息
    """
    print("="*80)
    print("车辆分布提取工具")
    print("="*80)

    # 加载配置
    cfg = get_args()

    # 初始化环境和智能体
    print("\n[INFO] 初始化环境...")
    env = gym.make('dclily')
    env.set_training_mode(0)  # 测试模式

    print("[INFO] 加载训练好的模型...")
    _, agent = env_agent_config(cfg)
    agent.load_model('trained_model.pth')

    # 运行一个episode并记录车辆分布
    print("[INFO] 运行RL策略并记录车辆分布...")
    state = env.reset()
    done = False
    step = 0

    vehicle_distribution = []

    while not done:
        # 记录当前时间步的车辆分布
        zone_vehicle_counts = [0] * env.ZONE_NUM
        for car in env.car:
            zone = int(car[0])  # car[0]是车辆所在区域
            zone_vehicle_counts[zone] += 1

        vehicle_distribution.append({
            'step': step,
            'zone_counts': zone_vehicle_counts.copy()
        })

        # 执行动作
        action = agent.predict_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        step += 1

    print(f"[INFO] 完成！共记录 {len(vehicle_distribution)} 个时间步")

    # 保存到文件
    save_vehicle_distribution(vehicle_distribution)
    print("[INFO] 车辆分布已保存到: action/vehicle_distribution_per_timestep.txt")


def save_vehicle_distribution(vehicle_distribution):
    """
    保存车辆分布数据到文件（表格格式）
    """
    ZONE_NUM = 4

    output_file = 'action/vehicle_distribution_per_timestep.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("车辆分布 - 每个时间步各区域的车辆数量（所有车辆）\n")
        f.write("="*80 + "\n\n")

        # 表格标题
        f.write("时间步 | 区域0 | 区域1 | 区域2 | 区域3 | 总计\n")
        f.write("-------|-------|-------|-------|-------|------\n")

        # 表格内容
        for data in vehicle_distribution:
            timestep = data['step']
            zone_counts = data['zone_counts']
            total = sum(zone_counts)

            f.write(f"  {timestep:2d}   |  {zone_counts[0]:3d}  |  {zone_counts[1]:3d}  |  {zone_counts[2]:3d}  |  {zone_counts[3]:3d}  | {total:4d}\n")

        # 添加统计信息
        f.write("\n" + "="*80 + "\n")
        f.write("统计信息\n")
        f.write("="*80 + "\n\n")

        # 计算每个区域的平均车辆数
        zone_totals = [0] * ZONE_NUM
        for data in vehicle_distribution:
            for zone in range(ZONE_NUM):
                zone_totals[zone] += data['zone_counts'][zone]

        f.write("各区域平均车辆数:\n")
        for zone in range(ZONE_NUM):
            avg = zone_totals[zone] / len(vehicle_distribution) if vehicle_distribution else 0
            f.write(f"  区域{zone}: {avg:.2f}辆\n")

        # 添加总体统计
        total_avg = sum(zone_totals) / len(vehicle_distribution) if vehicle_distribution else 0
        f.write(f"\n平均车辆总数: {total_avg:.2f}辆\n")


if __name__ == "__main__":
    extract_and_save_vehicle_distribution()
