# -*- coding: utf-8 -*-
"""
绘制训练过程中的loss曲线（从保存的数据文件）
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# 检查文件是否存在
if not os.path.exists('critic1_losses.npy'):
    print("错误：找不到critic1_losses.npy文件")
    print("请先运行main.py进行训练")
    exit()

# 加载loss数据
critic1_losses = np.load('critic1_losses.npy')
print(f"加载critic1 loss数据: {len(critic1_losses)}个数据点")

# 检查是否有critic2和actor loss
critic2_losses = None
actor_losses = None

if os.path.exists('critic2_losses.npy'):
    critic2_losses = np.load('critic2_losses.npy')
    print(f"加载critic2 loss数据: {len(critic2_losses)}个数据点")

if os.path.exists('actor_losses.npy'):
    actor_losses = np.load('actor_losses.npy')
    print(f"加载actor loss数据: {len(actor_losses)}个数据点")

# 创建图像
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# 绘制Critic Loss
axes[0].plot(critic1_losses, label='Critic Loss', color='blue', linewidth=1.5)
if critic2_losses is not None:
    axes[0].plot(critic2_losses, label='Critic2 Loss', color='cyan', linewidth=1.5)
axes[0].set_xlabel('Update Step')
axes[0].set_ylabel('Loss')
axes[0].set_title('Critic Loss Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 绘制Actor Loss
if actor_losses is not None and len(actor_losses) > 0:
    axes[1].plot(actor_losses, label='Actor Loss', color='red', linewidth=1.5)
    axes[1].set_xlabel('Update Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Actor Loss Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
else:
    axes[1].text(0.5, 0.5, 'No Actor Loss Data', ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_title('Actor Loss Curve (No Data)')

plt.tight_layout()

# 保存图像
plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
print("\n图像已保存为: loss_curves.png")

# 显示图像
plt.show()

print("\n统计信息:")
print(f"Critic Loss - 最小值: {np.min(critic1_losses):.4f}, 最大值: {np.max(critic1_losses):.4f}, 平均值: {np.mean(critic1_losses):.4f}")
if actor_losses is not None and len(actor_losses) > 0:
    print(f"Actor Loss - 最小值: {np.min(actor_losses):.4f}, 最大值: {np.max(actor_losses):.4f}, 平均值: {np.mean(actor_losses):.4f}")
