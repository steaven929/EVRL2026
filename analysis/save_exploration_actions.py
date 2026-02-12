# -*- coding: utf-8 -*-
"""
在训练的第10轮（探索期）保存详细动作
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
register(
    id='dclily',
    entry_point='specific.dclily_elec:dclily',
)

from core.util import get_args
from core.environment import env_agent_config
from core.training import OUNoise

def save_exploration_episode(cfg, env, agent, epoch=10):
    """
    在指定epoch保存一个完整episode的动作
    """
    print(f"\n开始记录第{epoch}轮探索期动作...")

    env.set_training_mode(0)  # 设置为测试模式，会保存详细动作
    env.set_epoch(epoch)
    ou_noise = OUNoise(env.action_space)

    state = env.reset()
    ou_noise.reset()

    actions_log = []
    ep_reward = 0

    for i_step in range(cfg['max_steps']):
        # 获取actor输出（无噪声）
        action_no_noise = agent.sample_action(state)

        # 添加OU噪声
        action_with_ou = ou_noise.get_action(action_no_noise, i_step+1)

        # 添加充电噪声（模拟探索期）
        charging_noise = np.random.uniform(-0.3, 0.3, size=4)
        action_final = action_with_ou.copy()
        action_final[8:12] = np.clip(action_with_ou[8:12] + charging_noise, 0, 1)

        # 记录动作
        actions_log.append({
            'step': i_step,
            'action_no_noise': action_no_noise.tolist(),
            'action_with_ou': action_with_ou.tolist(),
            'action_final': action_final.tolist(),
            'charging_noise': charging_noise.tolist()
        })

        # 执行动作
        next_state, reward, done, info = env.step(action_final)
        ep_reward += reward

        state = next_state
        if done:
            break

    # 保存到文件
    output_file = 'action/RL-DDPG_actions_exploration.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"RL-DDPG 策略 - 详细动作记录（探索期第{epoch}轮）\n")
        f.write("="*80 + "\n\n")
        f.write(f"总步数: {len(actions_log)}\n")
        f.write(f"总奖励: {ep_reward:.2f}\n\n")

        for action_data in actions_log:
            f.write("="*80 + "\n")
            f.write(f"时间步 {action_data['step']}\n")
            f.write("="*80 + "\n\n")

            f.write("【Actor原始输出（无噪声）】\n")
            f.write(f"  {action_data['action_no_noise']}\n\n")

            f.write("【添加OU噪声后】\n")
            f.write(f"  {action_data['action_with_ou']}\n\n")

            f.write("【充电噪声】\n")
            f.write(f"  {action_data['charging_noise']}\n\n")

            f.write("【最终动作（添加充电噪声后）】\n")
            f.write(f"  {action_data['action_final']}\n\n")

            # 提取充电偏好值
            charging_prefs = action_data['action_final'][8:12]
            f.write("【充电偏好值】\n")
            for zone_id, pref in enumerate(charging_prefs):
                f.write(f"  区域{zone_id}: {pref:.6f}\n")
            f.write("\n")

    print(f"探索期动作已保存到: {output_file}")
    print(f"总奖励: {ep_reward:.2f}")

if __name__ == '__main__':
    import sys

    # 切换到项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # 加载配置
    cfg = get_args()

    # 初始化环境和agent
    env, agent = env_agent_config(cfg)

    # 加载训练好的模型
    model_path = 'trained_model.pth'
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print(f"已加载模型: {model_path}")
    else:
        print(f"警告: 未找到模型文件 {model_path}")
        print("将使用未训练的模型（随机初始化）")

    # 保存探索期动作
    save_exploration_episode(cfg, env, agent, epoch=10)
