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
np.save('critic1_losses.npy', res_dic['critic1_losses'])  # 修改
if res_dic.get('critic2_losses'):  # 修改
    np.save('critic2_losses.npy', res_dic['critic2_losses'])  # 修改
if res_dic.get('actor_losses'):  # 修改
    np.save('actor_losses.npy', res_dic['actor_losses'])  # 修改
if res_dic.get('q_values'):  # 修改
    np.save('q_values.npy', res_dic['q_values'])  # 修改
if res_dic.get('target_q_values'):  # 修改
    np.save('target_q_values.npy', res_dic['target_q_values'])  # 修改
np.save('rewards.npy', res_dic['rewards'])  # 修改：保存 episode rewards
np.save('eval_rewards.npy', res_dic.get('eval_rewards', []))
np.save('eval_episodes.npy', res_dic.get('eval_episodes', []))
np.save('rld_history.npy', res_dic['rld_history'])  # 修改
np.save('svc_history.npy', res_dic['svc_history'])  # 修改
np.save('srd_history.npy', res_dic['srd_history'])  # 修改
print(f"Loss数据和指标已保存")  # 修改

# =========2026.1.31修改，保存训练好的模型==========
agent.save_model('trained_model.pth')
print(f"训练完成，模型已保存到: trained_model.pth")
# ======修改结束

plot_rewards(res_dic['rewards'], cfg, tag="train")

plot_critic_losses(res_dic['critic1_losses'], res_dic.get('critic2_losses', None), tag="train", save_path="critic_loss_curve.png")

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
plt.savefig('final_metrics.png', dpi=300, bbox_inches='tight')  # 修改
print("综合指标图已保存: final_metrics.png")  # 修改
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
    save_path="battery_heatmap.png" # 明确指定保存路径
)
print("电池热力图已保存: battery_heatmap.png")


# 测试
res_dic = dclilytrain.test(cfg, env, agent)
plot_rewards(res_dic['rewards'], cfg, tag="test")  # 画出结果
#plot_vehicle_battery(np.array(res_dic['battery_log'][0]), cfg)

# (这一趴貌似是我不小心多加进去的，重复了）
#res_dic = dclilytrain.test(cfg, env, agent)
#plot_rewards(res_dic['rewards'], cfg, tag="test")


