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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(os.getcwd())

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from core.util import get_args, plot_rewards, plot_critic_losses
from core.environment import env_agent_config
from core.marl_training import marl_train, marl_test

from gym.envs.registration import register
register(id='dclily', entry_point='specific.dclily_elec:dclily')

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
    print(f"参数共享: {cfg.get('share_policy', True)}")
    print(f"实验分支: {cfg.get('exp_variant', 'a1')}")
    print(f"奖励模式: {cfg.get('reward_mode', 'base')}")
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
    res_dic = marl_train(env, agent, cfg, path=path)

    # 保存所有数据（与main.py一致）
    np.save(path + 'critic_losses.npy', res_dic['critic_losses'])
    np.save(path + 'actor_losses.npy', res_dic['actor_losses'])
    np.save(path + 'rewards.npy', res_dic['rewards'])
    if 'raw_rewards' in res_dic:
        np.save(path + 'raw_rewards.npy', res_dic['raw_rewards'])
    np.save(path + 'ma_rewards.npy', res_dic['ma_rewards'])
    if 'rld_history' in res_dic:
        np.save(path + 'rld_history.npy', res_dic['rld_history'])
    if 'svc_history' in res_dic:
        np.save(path + 'svc_history.npy', res_dic['svc_history'])
    if 'srd_history' in res_dic:
        np.save(path + 'srd_history.npy', res_dic['srd_history'])
    print(f"数据已保存到: {path}")
    print(f"Info字段缺失计数: {res_dic.get('info_missing_counts', {})}")

    # 保存模型
    agent.save_model(path + 'marl_model_final.pth')
    print(f"模型已保存: {path}marl_model_final.pth")

    # 绘制训练曲线
    print("\n绘制训练曲线...")
    plot_rewards(res_dic['rewards'], cfg, path=path + 'train_rewards.png', tag='train')

    # 绘制Critic Loss曲线
    plot_critic_losses(res_dic['critic_losses'], None, tag="train", save_path=path + "critic_loss_curve.png")

    # 生成综合指标图（与main.py一致的8个子图）
    print("生成综合指标图...")
    fig, axes = plt.subplots(8, 1, figsize=(10, 20))

    # 子图1: Critic Loss
    axes[0].plot(res_dic['critic_losses'], label='Critic Loss', color='blue', linewidth=1.5)
    axes[0].set_xlabel('Update Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Critic Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 子图2: Actor Loss
    axes[1].plot(res_dic['actor_losses'], label='Actor Loss (-Q)', color='red', linewidth=1.5)
    axes[1].set_xlabel('Update Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Actor Loss (-Q Value, should decrease)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 子图3-4: 预留（MARL没有单独的Q值记录）
    axes[2].text(0.5, 0.5, 'Q Value tracking not available in MARL',
                ha='center', va='center', transform=axes[2].transAxes)
    axes[2].set_title('Q Value (N/A for MARL)')

    axes[3].text(0.5, 0.5, 'Target Q tracking not available in MARL',
                ha='center', va='center', transform=axes[3].transAxes)
    axes[3].set_title('Target Q Value (N/A for MARL)')

    # 子图5: RLD
    if 'rld_history' in res_dic and len(res_dic['rld_history']) > 0:
        axes[4].plot(res_dic['rld_history'], label='RLD', color='green', linewidth=1.5)
        axes[4].set_xlabel('Step')
        axes[4].set_ylabel('RLD')
        axes[4].set_title('Repositioning Loss Distance (RLD)')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)

    # 子图6: SVC
    if 'svc_history' in res_dic and len(res_dic['svc_history']) > 0:
        axes[5].plot(res_dic['svc_history'], label='SVC', color='purple', linewidth=1.5)
        axes[5].set_xlabel('Step')
        axes[5].set_ylabel('SVC')
        axes[5].set_title('Service Rate (SVC)')
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)

    # 子图7: SRD
    if 'srd_history' in res_dic and len(res_dic['srd_history']) > 0:
        axes[6].plot(res_dic['srd_history'], label='SRD', color='orange', linewidth=1.5)
        axes[6].set_xlabel('Step')
        axes[6].set_ylabel('SRD')
        axes[6].set_title('Service Reward Ratio (SRD)')
        axes[6].legend()
        axes[6].grid(True, alpha=0.3)

    # 子图8: Episode Reward
    axes[7].plot(res_dic['rewards'], label='Reward', color='gray', alpha=0.3, linewidth=1.0)
    from core.util import smooth
    axes[7].plot(smooth(res_dic['rewards']), label='Smoothed', color='blue', linewidth=2.0)
    axes[7].set_xlabel('Episode')
    axes[7].set_ylabel('Reward')
    axes[7].set_title('Episode Reward')
    axes[7].legend()
    axes[7].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path + 'final_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"综合指标图已保存: {path}final_metrics.png")

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
