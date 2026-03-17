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

from core.util import get_args, plot_rewards, plot_critic_losses, cp
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

    # 与 main.py 对齐：按 data_folder 同步配置到 specific/config.json
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(current_dir, 'specific')
    os.chdir(target_dir)
    cp("..\\" + cfg['data_folder'])
    os.chdir(current_dir)

    # =========2026.3.3修改，注释掉启动信息==========
    # print("\n" + "="*80)
    # print("MARL Phase-1: 区域级多智能体强化学习")
    # print("="*80)
    # print(f"模式: {cfg['marl_mode']}")
    # print(f"智能体数量: {cfg['n_agents']}")
    # print(f"参数共享: {cfg.get('share_policy', True)}")
    # print(f"实验分支: {cfg.get('exp_variant', 'a1')}")
    # print(f"奖励模式: {cfg.get('reward_mode', 'base')}")
    # print(f"训练轮数: {cfg['train_eps']}")
    # print(f"测试轮数: {cfg['test_eps']}")
    # print("="*80 + "\n")
    # ======修改结束

    # 创建环境和智能体
    env, agent = env_agent_config(cfg)

    # 与 main.py 对齐：统一时间步配置
    env_time_num = int(getattr(env, 'TIME_NUM', 26)) if hasattr(env, 'TIME_NUM') else 26
    if env_time_num <= 0:
        env_time_num = 26
    cfg['log_time_num'] = env_time_num
    if int(cfg.get('max_steps', env_time_num)) > env_time_num:
        cfg['max_steps'] = env_time_num

    # 设置保存路径（包含训练轮数）
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_eps = cfg.get('train_eps', 0)
    path = f"./outputs/marl_{train_eps}ep_{curr_time}/"
    os.makedirs(path, exist_ok=True)

    # =========2026.3.3修改，注释掉训练开始标题==========
    # print("\n" + "="*80)
    # print("开始训练")
    # print("="*80)
    # ======修改结束
    res_dic = marl_train(env, agent, cfg, path=path)

    # 保存所有数据（与main.py一致）
    np.save(path + 'critic_losses.npy', res_dic['critic_losses'])
    np.save(path + 'actor_losses.npy', res_dic['actor_losses'])
    np.save(path + 'rewards.npy', res_dic['rewards'])
    if 'eval_rewards' in res_dic:
        np.save(path + 'eval_rewards.npy', res_dic['eval_rewards'])
    if 'eval_episodes' in res_dic:
        np.save(path + 'eval_episodes.npy', res_dic['eval_episodes'])
    if 'raw_rewards' in res_dic:
        np.save(path + 'raw_rewards.npy', res_dic['raw_rewards'])
    np.save(path + 'ma_rewards.npy', res_dic['ma_rewards'])
    if 'rld_history' in res_dic:
        np.save(path + 'rld_history.npy', res_dic['rld_history'])
    if 'svc_history' in res_dic:
        np.save(path + 'svc_history.npy', res_dic['svc_history'])
    if 'srd_history' in res_dic:
        np.save(path + 'srd_history.npy', res_dic['srd_history'])
    # =========2026.3.3修改，恢复数据保存日志==========
    print(f"数据已保存到: {path}")
    # ======修改结束
    # =========2026.3.3修改，注释掉Info字段缺失计数==========
    # print(f"Info字段缺失计数: {res_dic.get('info_missing_counts', {})}")
    # ======修改结束

    # 保存模型
    agent.save_model(path + 'marl_model_final.pth')
    # =========2026.3.3修改，恢复模型保存日志==========
    print(f"模型已保存: {path}marl_model_final.pth")
    # ======修改结束

    # 绘制训练曲线
    # =========2026.3.3修改，恢复绘图日志==========
    print("\n绘制训练曲线...")
    # ======修改结束
    plot_rewards(res_dic['rewards'], cfg, path=path + 'train_rewards.png', tag='train')

    # 绘制Critic Loss曲线
    plot_critic_losses(res_dic['critic_losses'], None, tag="train", save_path=path + "critic_loss_curve.png")

    # 生成综合指标图（与main.py一致的8个子图）
    # =========2026.3.3修改，恢复绘图日志==========
    print("生成综合指标图...")
    # ======修改结束
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
    # =========2026.3.3修改，恢复保存日志==========
    print(f"综合指标图已保存: {path}final_metrics.png")
    # ======修改结束

    # =========2026.3.2修改，添加MARL专用指标图和对比图==========
    # 保存MARL专用指标数据
    if 'charging_efficiency_history' in res_dic:
        np.save(path + 'charging_efficiency_history.npy', res_dic['charging_efficiency_history'])
    if 'vehicle_utilization_history' in res_dic:
        np.save(path + 'vehicle_utilization_history.npy', res_dic['vehicle_utilization_history'])
    if 'fleet_health_history' in res_dic:
        np.save(path + 'fleet_health_history.npy', res_dic['fleet_health_history'])
    if 'zone_coordination_history' in res_dic:
        np.save(path + 'zone_coordination_history.npy', res_dic['zone_coordination_history'])
    if 'response_readiness_history' in res_dic:
        np.save(path + 'response_readiness_history.npy', res_dic['response_readiness_history'])
    if 'charger_occupancy_rate_history' in res_dic:
        np.save(path + 'charger_occupancy_rate_history.npy', res_dic['charger_occupancy_rate_history'])
    if 'charger_queue_delay_history' in res_dic:
        np.save(path + 'charger_queue_delay_history.npy', res_dic['charger_queue_delay_history'])
    if 'energy_not_served_history' in res_dic:
        np.save(path + 'energy_not_served_history.npy', res_dic['energy_not_served_history'])
    if 'safety_soc_violation_count_history' in res_dic:
        np.save(path + 'safety_soc_violation_count_history.npy', res_dic['safety_soc_violation_count_history'])
    if 'charge_shift_success_rate_history' in res_dic:
        np.save(path + 'charge_shift_success_rate_history.npy', res_dic['charge_shift_success_rate_history'])

    # 保存改进指标数据
    if 'weighted_service_rate_history' in res_dic:
        np.save(path + 'weighted_service_rate_history.npy', res_dic['weighted_service_rate_history'])
    if 'effective_service_rate_history' in res_dic:
        np.save(path + 'effective_service_rate_history.npy', res_dic['effective_service_rate_history'])
    if 'service_quality_index_history' in res_dic:
        np.save(path + 'service_quality_index_history.npy', res_dic['service_quality_index_history'])
    if 'dispatch_efficiency_history' in res_dic:
        np.save(path + 'dispatch_efficiency_history.npy', res_dic['dispatch_efficiency_history'])
    if 'effective_dispatch_rate_history' in res_dic:
        np.save(path + 'effective_dispatch_rate_history.npy', res_dic['effective_dispatch_rate_history'])
    if 'dispatch_roi_history' in res_dic:
        np.save(path + 'dispatch_roi_history.npy', res_dic['dispatch_roi_history'])

    # =========2026.3.12新增，保存26时间片级序列与均值=========
    timestep_keys = [
        'zone_ids',
        'timestep_ids',
        'svc_by_timestep',
        'rld_by_timestep',
        'srd_by_timestep',
        'reward_by_timestep',
        'total_demand_by_timestep',
        'served_orders_by_timestep',
        'charging_efficiency_by_timestep',
        'charger_queue_delay_by_timestep',
        'energy_not_served_by_timestep',
        'soc_violation_by_timestep',
        'svc_timestep_mean',
        'rld_timestep_mean',
        'srd_timestep_mean',
        'reward_timestep_mean',
        'total_demand_timestep_mean',
        'served_orders_timestep_mean',
        'charging_efficiency_timestep_mean',
        'charger_queue_delay_timestep_mean',
        'energy_not_served_timestep_mean',
        'soc_violation_timestep_mean',
        'demand_by_timestep_zone',
        'served_by_timestep_zone',
        'service_rate_by_timestep_zone',
        'charging_actions_by_timestep_zone',
        'dispatch_by_timestep_zone',
        'demand_timestep_zone_mean',
        'served_timestep_zone_mean',
        'service_rate_timestep_zone_mean',
        'charging_actions_timestep_zone_mean',
        'dispatch_timestep_zone_mean',
        'parking_available_by_timestep_zone',
        'parking_used_by_timestep_zone',
        'charger_available_by_timestep_zone',
        'charger_used_by_timestep_zone',
        'parking_available_timestep_zone_mean',
        'parking_used_timestep_zone_mean',
        'charger_available_timestep_zone_mean',
        'charger_used_timestep_zone_mean',
    ]
    for key in timestep_keys:
        if key in res_dic:
            np.save(path + f'{key}.npy', res_dic[key])
    # ======新增结束

    # =========2026.3.6新增，保存训练元数据用于对比分析==========
    import json
    import time
    metadata = {
        'algo_name': 'MARL',
        'seed': cfg['seed'],
        'train_eps': cfg['train_eps'],
        'n_agents': cfg.get('n_agents', 50),
        'batch_size': cfg['batch_size'],
        'actor_lr': cfg.get('actor_lr', 'N/A'),
        'critic_lr': cfg.get('critic_lr', 'N/A'),
        'gamma': cfg['gamma'],
        'tau': cfg.get('tau', 'N/A'),
        'memory_capacity': cfg['memory_capacity'],
        'obs_dim': cfg.get('obs_dim', 6),
        'action_dim': cfg.get('action_dim', 3),
        'global_state_dim': cfg.get('global_state_dim', 251),
        'timestamp': time.strftime('%Y%m%d-%H%M%S'),
        'total_episodes': len(res_dic['rewards']),
        'mean_reward': float(np.mean(res_dic['rewards'])),
        'max_reward': float(np.max(res_dic['rewards'])),
        'final_reward': float(res_dic['rewards'][-1]),
        'std_reward': float(np.std(res_dic['rewards'])),
        'mean_zone_coordination': float(np.mean(res_dic.get('zone_coordination_history', [0]))),
        'mean_response_readiness': float(np.mean(res_dic.get('response_readiness_history', [0]))),
        'mean_fleet_health': float(np.mean(res_dic.get('fleet_health_history', [0]))),
    }
    with open(path + 'training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"训练元数据已保存到: {path}training_metadata.json")
    # ======修改结束

    # 生成MARL专用综合指标图（12个子图）
    # =========2026.3.3修改，恢复绘图日志==========
    print("生成MARL综合指标图...")
    # ======修改结束
    fig, axes = plt.subplots(6, 2, figsize=(16, 24))
    axes = axes.flatten()

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

    # 子图3: Episode Reward (保持原版，不修改)
    axes[2].plot(res_dic['rewards'], label='Episode Reward', color='green', linewidth=1.5, alpha=0.5)
    axes[2].plot(res_dic['ma_rewards'], label='MA Reward', color='darkgreen', linewidth=2.0)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Reward')
    axes[2].set_title('Episode Reward (Scaled)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # 子图4: Service Rate (SVC)
    axes[3].plot(res_dic['svc_history'], label='SVC (Mean)', color='purple', linewidth=1.5)
    axes[3].set_xlabel('Episode')
    axes[3].set_ylabel('Service Rate (%)')
    axes[3].set_title('Service Rate (SVC) - Traditional Metric')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    # ===== 新增MARL专用指标 =====

    # 子图5: Charging Efficiency
    if 'charging_efficiency_history' in res_dic:
        axes[4].plot(res_dic['charging_efficiency_history'], label='Charging Efficiency',
                    color='#FF6B6B', linewidth=2.0)
        axes[4].set_xlabel('Episode')
        axes[4].set_ylabel('Efficiency (0-1)')
        axes[4].set_title('Charging Efficiency (Effective/Total) - Higher is Better')
        axes[4].axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Target: 0.9')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)
        axes[4].set_ylim([0, 1.05])

    # 子图6: Vehicle Utilization
    if 'vehicle_utilization_history' in res_dic:
        axes[5].plot(res_dic['vehicle_utilization_history'], label='Vehicle Utilization',
                    color='#4ECDC4', linewidth=2.0)
        axes[5].set_xlabel('Episode')
        axes[5].set_ylabel('Utilization (0-1)')
        axes[5].set_title('Vehicle Utilization (Active/Total) - Higher is Better')
        axes[5].axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Target: 0.5')
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)
        axes[5].set_ylim([0, 1.05])

    # 子图7: Fleet Health
    if 'fleet_health_history' in res_dic:
        axes[6].plot(res_dic['fleet_health_history'], label='Fleet Health',
                    color='#95E1D3', linewidth=2.0)
        axes[6].set_xlabel('Episode')
        axes[6].set_ylabel('Health (0-1)')
        axes[6].set_title('Fleet Health (Avg SOC/Capacity) - Higher is Better')
        axes[6].axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='Target: 0.6')
        axes[6].legend()
        axes[6].grid(True, alpha=0.3)
        axes[6].set_ylim([0, 1.05])

    # 子图8: Zone Coordination
    if 'zone_coordination_history' in res_dic:
        axes[7].plot(res_dic['zone_coordination_history'], label='Zone Coordination',
                    color='#F38181', linewidth=2.0)
        axes[7].set_xlabel('Episode')
        axes[7].set_ylabel('Coordination (0-1)')
        axes[7].set_title('Zone Coordination (Vehicle Distribution Balance) - Higher is Better')
        axes[7].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Target: 0.7')
        axes[7].legend()
        axes[7].grid(True, alpha=0.3)
        axes[7].set_ylim([0, 1.05])

    # 子图9: Response Readiness
    if 'response_readiness_history' in res_dic:
        axes[8].plot(res_dic['response_readiness_history'], label='Response Readiness',
                    color='#AA96DA', linewidth=2.0)
        axes[8].set_xlabel('Episode')
        axes[8].set_ylabel('Readiness (0-1)')
        axes[8].set_title('Response Readiness (High-SOC Idle Vehicles) - Higher is Better')
        axes[8].axhline(y=0.4, color='green', linestyle='--', alpha=0.5, label='Target: 0.4')
        axes[8].legend()
        axes[8].grid(True, alpha=0.3)
        axes[8].set_ylim([0, 1.05])

    # 子图10: RLD (Relocation Distance)
    axes[9].plot(res_dic['rld_history'], label='RLD', color='orange', linewidth=1.5)
    axes[9].set_xlabel('Episode')
    axes[9].set_ylabel('Distance')
    axes[9].set_title('Relocation Distance (RLD)')
    axes[9].legend()
    axes[9].grid(True, alpha=0.3)

    # 子图11: MARL Overall Performance
    if all(k in res_dic for k in ['charging_efficiency_history', 'vehicle_utilization_history',
                                    'fleet_health_history', 'zone_coordination_history',
                                    'response_readiness_history']):
        last_n = 10
        metrics_avg = {
            'Charging\\nEfficiency': np.mean(res_dic['charging_efficiency_history'][-last_n:]),
            'Vehicle\\nUtilization': np.mean(res_dic['vehicle_utilization_history'][-last_n:]),
            'Fleet\\nHealth': np.mean(res_dic['fleet_health_history'][-last_n:]),
            'Zone\\nCoordination': np.mean(res_dic['zone_coordination_history'][-last_n:]),
            'Response\\nReadiness': np.mean(res_dic['response_readiness_history'][-last_n:]),
        }

        axes[10].bar(range(len(metrics_avg)), list(metrics_avg.values()),
                    color=['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181', '#AA96DA'])
        axes[10].set_xticks(range(len(metrics_avg)))
        axes[10].set_xticklabels(list(metrics_avg.keys()), rotation=15, ha='right')
        axes[10].set_ylabel('Score (0-1)')
        axes[10].set_title(f'MARL Overall Performance (Last {last_n} Episodes Avg)')
        axes[10].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Baseline: 0.5')
        axes[10].legend()
        axes[10].grid(True, alpha=0.3, axis='y')
        axes[10].set_ylim([0, 1.05])

        for i, (k, v) in enumerate(metrics_avg.items()):
            axes[10].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    # 子图12: Training Summary
    axes[11].axis('off')
    summary_text = f"""
    MARL Training Summary
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Episodes: {len(res_dic['rewards'])}
    Final Reward: {res_dic['rewards'][-1]:.2f}
    Best Reward: {max(res_dic['rewards']):.2f}

    Traditional Metrics (Last 10 Eps Avg):
    • Service Rate: {np.mean(res_dic['svc_history'][-10:]):.2f}%
    • RLD: {np.mean(res_dic['rld_history'][-10:]):.2f}

    MARL-Specific Metrics (Last 10 Eps Avg):
    • Charging Efficiency: {np.mean(res_dic['charging_efficiency_history'][-10:]):.3f}
    • Vehicle Utilization: {np.mean(res_dic['vehicle_utilization_history'][-10:]):.3f}
    • Fleet Health: {np.mean(res_dic['fleet_health_history'][-10:]):.3f}
    • Zone Coordination: {np.mean(res_dic['zone_coordination_history'][-10:]):.3f}
    • Response Readiness: {np.mean(res_dic['response_readiness_history'][-10:]):.3f}

    ✅ MARL Advantage: Multi-dimensional System Evaluation
    """
    axes[11].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                 verticalalignment='center')

    plt.tight_layout()
    plt.savefig(path + 'marl_comprehensive_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    # =========2026.3.3修改，恢复保存日志==========
    print(f"MARL综合指标图已保存: {path}marl_comprehensive_metrics.png")
    # ======修改结束

    # 生成旧版vs新版指标对比图
    # =========2026.3.3修改，恢复绘图日志==========
    print("生成旧版vs新版指标对比图...")
    # ======修改结束
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.flatten()

    # 子图1: SVC Comparison
    axes[0].plot(res_dic['svc_history'], label='SVC (Old)', color='gray', linewidth=2.0, alpha=0.6)
    if 'weighted_service_rate_history' in res_dic:
        axes[0].plot(res_dic['weighted_service_rate_history'], label='Weighted Service Rate (New)',
                    color='#FF6B6B', linewidth=2.0)
    if 'effective_service_rate_history' in res_dic:
        axes[0].plot(res_dic['effective_service_rate_history'], label='Effective Service Rate (New)',
                    color='#4ECDC4', linewidth=2.0)
    if 'service_quality_index_history' in res_dic:
        axes[0].plot(res_dic['service_quality_index_history'], label='Service Quality Index (New)',
                    color='#95E1D3', linewidth=2.0)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Rate (0-1)')
    axes[0].set_title('SVC Metrics Comparison: Old vs New')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    # 子图2: RLD Comparison
    axes[1].plot(res_dic['rld_history'], label='RLD (Old)', color='gray', linewidth=2.0, alpha=0.6)
    ax1_twin = axes[1].twinx()
    if 'dispatch_efficiency_history' in res_dic:
        ax1_twin.plot(res_dic['dispatch_efficiency_history'], label='Dispatch Efficiency (New)',
                     color='#F38181', linewidth=2.0)
    if 'effective_dispatch_rate_history' in res_dic:
        ax1_twin.plot(res_dic['effective_dispatch_rate_history'], label='Effective Dispatch Rate (New)',
                     color='#AA96DA', linewidth=2.0)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('RLD Distance (Old)', color='gray')
    ax1_twin.set_ylabel('Efficiency/Rate (New)', color='#F38181')
    axes[1].set_title('RLD Metrics Comparison: Old vs New')
    axes[1].legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # 子图3: Weighted Service Rate
    if 'weighted_service_rate_history' in res_dic:
        axes[2].plot(res_dic['weighted_service_rate_history'], label='Weighted Service Rate',
                    color='#FF6B6B', linewidth=2.0)
        axes[2].fill_between(range(len(res_dic['weighted_service_rate_history'])),
                            res_dic['weighted_service_rate_history'],
                            alpha=0.3, color='#FF6B6B')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Weighted Service Rate')
        axes[2].set_title('Weighted Service Rate (Distance/Value Weighted)')
        axes[2].axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Target: 0.5')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 1.05])

    # 子图4: Service Quality Index
    if 'service_quality_index_history' in res_dic:
        axes[3].plot(res_dic['service_quality_index_history'], label='Service Quality Index',
                    color='#95E1D3', linewidth=2.0)
        axes[3].fill_between(range(len(res_dic['service_quality_index_history'])),
                            res_dic['service_quality_index_history'],
                            alpha=0.3, color='#95E1D3')
        axes[3].set_xlabel('Episode')
        axes[3].set_ylabel('Service Quality Index')
        axes[3].set_title('Service Quality Index (Rate+SOC+Readiness)')
        axes[3].axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='Target: 0.6')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylim([0, 1.05])

    # 子图5: Dispatch Efficiency
    if 'dispatch_efficiency_history' in res_dic:
        axes[4].plot(res_dic['dispatch_efficiency_history'], label='Dispatch Efficiency',
                    color='#F38181', linewidth=2.0)
        axes[4].fill_between(range(len(res_dic['dispatch_efficiency_history'])),
                            res_dic['dispatch_efficiency_history'],
                            alpha=0.3, color='#F38181')
        axes[4].set_xlabel('Episode')
        axes[4].set_ylabel('Dispatch Efficiency (orders/distance)')
        axes[4].set_title('Dispatch Efficiency (Orders/Distance)')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)

    # 子图6: Dispatch ROI
    if 'dispatch_roi_history' in res_dic:
        axes[5].plot(res_dic['dispatch_roi_history'], label='Dispatch ROI',
                    color='#AA96DA', linewidth=2.0)
        axes[5].fill_between(range(len(res_dic['dispatch_roi_history'])),
                            res_dic['dispatch_roi_history'],
                            alpha=0.3, color='#AA96DA')
        axes[5].axhline(y=0, color='red', linestyle='-', alpha=0.5, label='Break-even')
        axes[5].set_xlabel('Episode')
        axes[5].set_ylabel('Dispatch ROI')
        axes[5].set_title('Dispatch ROI (Return on Investment)')
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path + 'metrics_comparison_old_vs_new.png', dpi=150, bbox_inches='tight')
    plt.close()
    # =========2026.3.3修改，恢复保存日志==========
    print(f"旧版vs新版指标对比图已保存: {path}metrics_comparison_old_vs_new.png")
    # ======修改结束

    # 充电调度关键指标图（论文指标）
    # =========2026.3.3修改，恢复绘图日志==========
    print("生成充电调度关键指标图...")
    # ======修改结束
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    axes = axes.flatten()

    if 'charger_occupancy_rate_history' in res_dic:
        axes[0].plot(res_dic['charger_occupancy_rate_history'], color='#1f77b4', linewidth=2.0, label='Charger Occupancy Rate')
        axes[0].set_title('Charger Occupancy Rate')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Rate (0-1)')
        axes[0].set_ylim([0, 1.05])
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

    if 'charger_queue_delay_history' in res_dic:
        axes[1].plot(res_dic['charger_queue_delay_history'], color='#ff7f0e', linewidth=2.0, label='Charger Queue Delay')
        axes[1].set_title('Charger Queue Delay (minutes, proxy)')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Minutes')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

    if 'energy_not_served_history' in res_dic:
        axes[2].plot(res_dic['energy_not_served_history'], color='#d62728', linewidth=2.0, label='Energy Not Served')
        axes[2].set_title('Energy Not Served (orders, proxy)')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Orders')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

    if 'safety_soc_violation_count_history' in res_dic:
        axes[3].plot(res_dic['safety_soc_violation_count_history'], color='#9467bd', linewidth=2.0,
                     label='Safety SOC Violation Count')
        axes[3].set_title('Safety SOC Violation Count (SOC < 20%)')
        axes[3].set_xlabel('Episode')
        axes[3].set_ylabel('Vehicle Count')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()

    if 'charge_shift_success_rate_history' in res_dic:
        axes[4].plot(res_dic['charge_shift_success_rate_history'], color='#2ca02c', linewidth=2.0,
                     label='Charge Shift Success Rate')
        axes[4].set_title('Charge Shift Success Rate')
        axes[4].set_xlabel('Episode')
        axes[4].set_ylabel('Rate (0-1)')
        axes[4].set_ylim([0, 1.05])
        axes[4].grid(True, alpha=0.3)
        axes[4].legend()

    axes[5].axis('off')
    summary = f"""
    Charging Scheduling Metrics Summary (Last 10 Eps Avg)
    -----------------------------------------------------
    Charger Occupancy Rate: {np.nanmean(res_dic.get('charger_occupancy_rate_history', [np.nan])[-10:]):.3f}
    Charger Queue Delay: {np.nanmean(res_dic.get('charger_queue_delay_history', [np.nan])[-10:]):.3f}
    Energy Not Served: {np.nanmean(res_dic.get('energy_not_served_history', [np.nan])[-10:]):.3f}
    Safety SOC Violations: {np.nanmean(res_dic.get('safety_soc_violation_count_history', [np.nan])[-10:]):.3f}
    Charge Shift Success Rate: {np.nanmean(res_dic.get('charge_shift_success_rate_history', [np.nan])[-10:]):.3f}
    """
    axes[5].text(0.05, 0.5, summary, fontsize=10, family='monospace', va='center')

    plt.tight_layout()
    plt.savefig(path + 'charging_scheduling_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    # =========2026.3.3修改，恢复保存日志==========
    print(f"充电调度关键指标图已保存: {path}charging_scheduling_metrics.png")
    # ======修改结束
    # ======修改结束

    # 测试
    # =========2026.3.3修改，注释掉测试标题==========
    # print("\n" + "="*80)
    # print("开始测试")
    # print("="*80)
    # ======修改结束
    test_rewards = marl_test(env, agent, cfg, path=path)

    # =========2026.3.3修改，恢复完成标题==========
    print("\n" + "="*80)
    print("训练完成！")
    print(f"结果保存在: {path}")
    print("="*80)
    # ======修改结束

if __name__ == "__main__":
    main()
# ======修改结束
