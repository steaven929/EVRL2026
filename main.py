# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 19:29:48 2024

@author: TLab
"""
# =========2026.2.5修改，修复Windows中文编码问题==========
import sys
import io
# 设置标准输出为 UTF-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# ======修改结束

from matplotlib import pyplot as plt

from core.util import get_args, plot_rewards,plot_vehicle_battery_heatmap,plot_vehicle_battery, cp, plot_critic_losses, smooth
import core.environment as dclilyenv
import core.training as dclilytrain
import numpy as np
import os
from datetime import datetime

from gym.envs.registration import register

# 建立环境

register(
    id='dclily',  # 自定义环境的唯一 ID
    entry_point='specific.dclily_elec:dclily',  # 环境类的路径：模块.类名
)

# 获取参数
cfg = get_args()

# 复制参数h

# 获取 main.py 所在的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接出 specific 文件夹的绝对路径
target_dir = os.path.join(current_dir, 'specific')

# 切换工作目录
os.chdir(target_dir)
#os.chdir('.\\specific')
cp("..\\" + cfg['data_folder'])
os.chdir('..')

# 为每次运行创建独立输出目录：时间戳 + 算法名 + 训练轮数
output_root = os.path.join(current_dir, 'output')
os.makedirs(output_root, exist_ok=True)
run_stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
run_dir = os.path.join(output_root, f"{run_stamp}_{str(cfg['algo_name']).upper()}_{int(cfg['train_eps'])}ep")
os.makedirs(run_dir, exist_ok=True)
cfg['run_output_dir'] = run_dir
print(f"运行输出目录: {run_dir}")


def out_file(filename):
    return os.path.join(run_dir, filename)

# 训练
env, agent = dclilyenv.env_agent_config(cfg)


# ✅ 添加：给环境绑定 ZONE_NUM 属性（用于切片）
#env.ZONE_NUM = env.action_space.shape[0] //


# ✅ 添加：自定义动作测试
#from core.environment import manual_action_test  # 该函数加在 environment.py 里
#manual_action_test(env)

#训练
res_dic = dclilytrain.train(cfg, env, agent)

# 保存loss数据和指标数据  # 修改
np.save(out_file('critic1_losses.npy'), res_dic['critic1_losses'])  # 修改
if res_dic.get('critic2_losses'):  # 修改
    np.save(out_file('critic2_losses.npy'), res_dic['critic2_losses'])  # 修改
if res_dic.get('actor_losses'):  # 修改
    np.save(out_file('actor_losses.npy'), res_dic['actor_losses'])  # 修改
if res_dic.get('q_values'):  # 修改
    np.save(out_file('q_values.npy'), res_dic['q_values'])  # 修改
if res_dic.get('target_q_values'):  # 修改
    np.save(out_file('target_q_values.npy'), res_dic['target_q_values'])  # 修改
np.save(out_file('rewards.npy'), res_dic['rewards'])  # 修改：保存 episode rewards
if res_dic.get('raw_rewards'):
    np.save(out_file('raw_rewards.npy'), res_dic['raw_rewards'])
np.save(out_file('eval_rewards.npy'), res_dic.get('eval_rewards', []))
np.save(out_file('eval_episodes.npy'), res_dic.get('eval_episodes', []))
np.save(out_file('rld_history.npy'), res_dic['rld_history'])  # 修改
np.save(out_file('svc_history.npy'), res_dic['svc_history'])  # 修改
np.save(out_file('srd_history.npy'), res_dic['srd_history'])  # 修改

optional_episode_metric_keys = [
    'charging_efficiency_history',
    'vehicle_utilization_history',
    'fleet_health_history',
    'zone_coordination_history',
    'response_readiness_history',
    'charger_occupancy_rate_history',
    'charger_queue_delay_history',
    'energy_not_served_history',
    'safety_soc_violation_count_history',
    'charge_shift_success_rate_history',
    'weighted_service_rate_history',
    'effective_service_rate_history',
    'service_quality_index_history',
    'dispatch_efficiency_history',
    'effective_dispatch_rate_history',
    'dispatch_roi_history',
]
for key in optional_episode_metric_keys:
    if key in res_dic:
        np.save(out_file(f'{key}.npy'), res_dic[key])

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
        np.save(out_file(f'{key}.npy'), res_dic[key])

print(f"Loss数据和指标已保存")  # 修改
print(f"Info字段缺失计数: {res_dic.get('info_missing_counts', {})}")

# =========2026.3.6新增，保存训练元数据用于对比分析==========
import json
import time
metadata = {
    'algo_name': cfg['algo_name'],
    'seed': cfg['seed'],
    'train_eps': cfg['train_eps'],
    'batch_size': cfg['batch_size'],
    'actor_lr': cfg.get('actor_lr', cfg.get('lr', 'N/A')),
    'critic_lr': cfg.get('critic_lr', cfg.get('lr', 'N/A')),
    'gamma': cfg['gamma'],
    'tau': cfg.get('tau', 'N/A'),
    'memory_capacity': cfg['memory_capacity'],
    'n_states': cfg['n_states'],
    'n_actions': cfg['n_actions'],
    'timestamp': time.strftime('%Y%m%d-%H%M%S'),
    'total_episodes': len(res_dic['rewards']),
    'mean_reward': float(np.mean(res_dic['rewards'])),
    'max_reward': float(np.max(res_dic['rewards'])),
    'final_reward': float(res_dic['rewards'][-1]),
    'std_reward': float(np.std(res_dic['rewards'])),
}
with open(out_file('training_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"训练元数据已保存到: {out_file('training_metadata.json')}")
# ======修改结束

# =========2026.1.31修改，保存训练好的模型==========
agent.save_model(out_file('trained_model.pth'))
print(f"训练完成，模型已保存到: {out_file('trained_model.pth')}")
# ======修改结束

plot_rewards(res_dic['rewards'], cfg, tag="train")

plot_critic_losses(
    res_dic['critic1_losses'],
    res_dic.get('critic2_losses', None),
    tag="train",
    save_path=out_file("critic_loss_curve.png")
)

# 生成包含7个子图的综合指标图  # 修改
import matplotlib  # 修改
matplotlib.use('Agg')  # 修改
fig, axes = plt.subplots(8, 1, figsize=(10, 20))  # 修改

# 子图1: Critic Loss  # 修改
axes[0].plot(res_dic['critic1_losses'], label='Critic Loss', color='blue', linewidth=1.5)  # 修改
axes[0].set_xlabel('Update Step')  # 修改
axes[0].set_ylabel('Loss')  # 修改
axes[0].set_title('Critic Loss')  # 修改
axes[0].legend()  # 修改
axes[0].grid(True, alpha=0.3)  # 修改

# 子图2: Actor Loss (-Q值，应该下降)  # 修改
if res_dic.get('actor_losses') and len(res_dic['actor_losses']) > 0:  # 修改
    axes[1].plot(res_dic['actor_losses'], label='Actor Loss (-Q)', color='red', linewidth=1.5)  # 修改
axes[1].set_xlabel('Update Step')  # 修改
axes[1].set_ylabel('Loss')  # 修改
axes[1].set_title('Actor Loss (-Q Value, should decrease)')  # 修改
axes[1].legend()  # 修改
axes[1].grid(True, alpha=0.3)  # 修改

# 子图3: Q Value (应该上升，变得不那么负)  # 修改
if res_dic.get('q_values') and len(res_dic['q_values']) > 0:  # 修改
    axes[2].plot(res_dic['q_values'], label='Q Value', color='cyan', linewidth=1.5)  # 修改
axes[2].set_xlabel('Update Step')  # 修改
axes[2].set_ylabel('Q Value')  # 修改
axes[2].set_title('Q Value (should increase towards 0)')  # 修改
axes[2].legend()  # 修改
axes[2].grid(True, alpha=0.3)  # 修改

# 子图4: Target Q Value  # 修改
if res_dic.get('target_q_values') and len(res_dic['target_q_values']) > 0:  # 修改
    axes[3].plot(res_dic['target_q_values'], label='Target Q', color='magenta', linewidth=1.5)  # 修改
axes[3].set_xlabel('Update Step')  # 修改
axes[3].set_ylabel('Target Q Value')  # 修改
axes[3].set_title('Target Q Value')  # 修改
axes[3].legend()  # 修改
axes[3].grid(True, alpha=0.3)  # 修改

# 子图5: RLD  # 修改
axes[4].plot(res_dic['rld_history'], label='RLD', color='green', linewidth=1.5)  # 修改
axes[4].set_xlabel('Step')  # 修改
axes[4].set_ylabel('RLD')  # 修改
axes[4].set_title('Repositioning Loss Distance (RLD)')  # 修改
axes[4].legend()  # 修改
axes[4].grid(True, alpha=0.3)  # 修改

# 子图6: SVC  # 修改
axes[5].plot(res_dic['svc_history'], label='SVC', color='purple', linewidth=1.5)  # 修改
axes[5].set_xlabel('Step')  # 修改
axes[5].set_ylabel('SVC')  # 修改
axes[5].set_title('Service Rate (SVC)')  # 修改
axes[5].legend()  # 修改
axes[5].grid(True, alpha=0.3)  # 修改

# 子图7: SRD  # 修改
axes[6].plot(res_dic['srd_history'], label='SRD', color='orange', linewidth=1.5)  # 修改
axes[6].set_xlabel('Step')  # 修改
axes[6].set_ylabel('SRD')  # 修改
axes[6].set_title('Service Reward Ratio (SRD)')  # 修改
axes[6].legend()  # 修改
axes[6].grid(True, alpha=0.3)  # 修改

# 子图8: Reward（按 episode）  # 修改
rewards = np.asarray(res_dic.get('rewards', []), dtype=float)
if rewards.size > 0:
    # 原始数据（灰色背景）
    axes[7].plot(rewards, label='Reward', linewidth=2.0, alpha=0.15, color='gray')  # 修改

    # 平滑数据（深红色），直接调用 util.py 里的 smooth
    if len(rewards) > 1:
        try:
            smoothed_rewards = smooth(rewards.tolist(), weight=0.9)
            axes[7].plot(smooth(rewards), label='smoothed', linewidth=2.0)  # 修改
        except Exception as e:
            print(f"平滑计算出错: {e}")

    eval_rewards = np.asarray(res_dic.get('eval_rewards', []), dtype=float)
    eval_episodes = np.asarray(res_dic.get('eval_episodes', []), dtype=float)
    if eval_rewards.size > 0 and eval_episodes.size == eval_rewards.size:
        axes[7].plot(eval_episodes - 1, eval_rewards, 'o-', label='eval_reward', linewidth=1.5, markersize=4)

axes[7].set_xlabel('Episode')  # 修改
axes[7].set_ylabel('Reward')  # 修改
axes[7].set_title('Episode Reward')  # 修改
axes[7].legend(loc='upper right')  # 修改
axes[7].grid(True, alpha=0.3)  # 修改

plt.tight_layout()  # 修改
plt.savefig(out_file('final_metrics.png'), dpi=300, bbox_inches='tight')  # 修改
print(f"综合指标图已保存: {out_file('final_metrics.png')}")  # 修改
plt.close()  # 修改

#plot_vehicle_battery(np.array(res_dic['battery_log'][0]), cfg)



#这一部分被注释的是热力图图像
# plot_vehicle_battery_heatmap(res_dic['battery_log'], title="Battery Heatmap", save_path=None)
# print(res_dic['battery_log'])
# print(np.array(res_dic['battery_log']))
# plt.show()
# ==========================================
# ✅ 修改：绘制热力图并保存
# ==========================================
# 由于上面用了 matplotlib.use('Agg')，plt.show() 可能不会弹窗
# 所以强制传入 save_path 来保存图片
plot_vehicle_battery_heatmap(
    res_dic['battery_log'],
    title="Battery Heatmap",
    save_path=out_file("battery_heatmap.png") # 明确指定保存路径
)
print(f"电池热力图已保存: {out_file('battery_heatmap.png')}")


# 测试
res_dic = dclilytrain.test(cfg, env, agent)
plot_rewards(res_dic['rewards'], cfg, tag="test")  # 画出结果
np.save(out_file('test_rewards.npy'), res_dic.get('rewards', []))
#plot_vehicle_battery(np.array(res_dic['battery_log'][0]), cfg)

# (这一趴貌似是我不小心多加进去的，重复了）
#res_dic = dclilytrain.test(cfg, env, agent)
#plot_rewards(res_dic['rewards'], cfg, tag="test")



