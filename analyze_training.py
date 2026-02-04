import numpy as np
import matplotlib.pyplot as plt

# 加载训练数据
rewards = np.load('rewards.npy')
actor_losses = np.load('actor_losses.npy')
critic1_losses = np.load('critic1_losses.npy')

print("=" * 80)
print("700轮训练结果分析")
print("=" * 80)

# 基本统计
print(f"\n【训练轮数】{len(rewards)}")
print(f"【总体统计】")
print(f"  平均奖励: {rewards.mean():.2f}")
print(f"  最大奖励: {rewards.max():.2f}")
print(f"  最小奖励: {rewards.min():.2f}")
print(f"  标准差: {rewards.std():.2f}")

# 最后70轮（10%）统计
last_70 = rewards[-70:]
print(f"\n【最后70轮统计】")
print(f"  平均奖励: {last_70.mean():.2f}")
print(f"  最大奖励: {last_70.max():.2f}")
print(f"  最小奖励: {last_70.min():.2f}")
print(f"  标准差: {last_70.std():.2f}")

# 分段统计
segments = 7
segment_size = len(rewards) // segments
print(f"\n【分段统计（每100轮）】")
for i in range(segments):
    start = i * segment_size
    end = (i + 1) * segment_size if i < segments - 1 else len(rewards)
    segment_rewards = rewards[start:end]
    print(f"  轮次 {start:3d}-{end:3d}: 平均 {segment_rewards.mean():8.2f}, "
          f"最大 {segment_rewards.max():8.2f}, 最小 {segment_rewards.min():8.2f}")

# 与基线对比
print(f"\n【与基线对比】")
print(f"  最优解: 469.80")
print(f"  Greedy: -4295.00")
print(f"  RL-DDPG (最后70轮平均): {last_70.mean():.2f}")
print(f"  与最优解差距: {469.80 - last_70.mean():.2f}")
print(f"  与Greedy差距: {last_70.mean() - (-4295.00):.2f}")

# 收敛性分析
print(f"\n【收敛性分析】")
first_100 = rewards[:100].mean()
last_100 = rewards[-100:].mean()
improvement = last_100 - first_100
print(f"  前100轮平均: {first_100:.2f}")
print(f"  后100轮平均: {last_100:.2f}")
print(f"  改进幅度: {improvement:.2f} ({improvement/abs(first_100)*100:.2f}%)")

# 绘制训练曲线
plt.figure(figsize=(15, 10))

# 奖励曲线
plt.subplot(3, 1, 1)
plt.plot(rewards, alpha=0.3, label='原始奖励')
# 移动平均
window = 20
moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
plt.plot(range(window-1, len(rewards)), moving_avg, label=f'{window}轮移动平均', linewidth=2)
plt.axhline(y=469.80, color='g', linestyle='--', label='最优解 (469.80)')
plt.axhline(y=-4295.00, color='r', linestyle='--', label='Greedy (-4295.00)')
plt.xlabel('训练轮次')
plt.ylabel('奖励')
plt.title('训练奖励曲线')
plt.legend()
plt.grid(True, alpha=0.3)

# Actor损失
plt.subplot(3, 1, 2)
plt.plot(actor_losses, alpha=0.5)
plt.xlabel('训练轮次')
plt.ylabel('Actor损失')
plt.title('Actor损失曲线')
plt.grid(True, alpha=0.3)

# Critic损失
plt.subplot(3, 1, 3)
plt.plot(critic1_losses, alpha=0.5)
plt.xlabel('训练轮次')
plt.ylabel('Critic损失')
plt.title('Critic损失曲线')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_analysis_700.png', dpi=150)
print(f"\n训练曲线已保存至: training_analysis_700.png")
