# -*- coding: utf-8 -*-
"""
MARL Phase-1: Main Entry Point
Created on 2026-02-19

多智能体强化学习主程序
"""
import sys
import os
import torch
import datetime

# 添加项目路径
sys.path.append(os.getcwd())

from core.util import get_args, plot_rewards
from core.environment import env_agent_config
from core.marl_training import marl_train, marl_test

# =========2026.2.19修改，MARL Phase-1：主程序==========
def main():
    # 获取配置参数
    cfg = get_args()

    # 强制设置为multi模式
    cfg['marl_mode'] = 'multi'
    cfg['n_agents'] = 50  # SIP数据集：50个区域

    print("\n" + "="*80)
    print("MARL Phase-1: 区域级多智能体强化学习")
    print("="*80)
    print(f"模式: {cfg['marl_mode']}")
    print(f"智能体数量: {cfg['n_agents']}")
    print(f"参数共享: {cfg['share_policy']}")
    print(f"训练轮数: {cfg['train_eps']}")
    print(f"测试轮数: {cfg['test_eps']}")
    print("="*80 + "\n")

    # 创建环境和智能体
    env, agent = env_agent_config(cfg)

    # 设置保存路径
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = f"./outputs/marl_{curr_time}/"
    os.makedirs(path, exist_ok=True)

    # 训练
    print("\n" + "="*80)
    print("开始训练")
    print("="*80)
    rewards, ma_rewards = marl_train(env, agent, cfg, path=path)

    # 绘制训练曲线
    print("\n绘制训练曲线...")
    plot_rewards(rewards, cfg, path=path + 'train_rewards.png', tag='train')

    # 测试
    print("\n" + "="*80)
    print("开始测试")
    print("="*80)
    test_rewards = marl_test(env, agent, cfg, path=path)

    print("\n" + "="*80)
    print("训练完成！")
    print(f"结果保存在: {path}")
    print("="*80)

if __name__ == "__main__":
    main()
# ======修改结束
