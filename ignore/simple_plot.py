import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

algorithms = ['Random', 'Greedy', 'RL-DDPG', 'Oracle']
rewards = [-2752.54, -2477.28, -2293.78, -1995.59]
colors = ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1']

plt.figure(figsize=(8, 5))
bars = plt.bar(algorithms, rewards, color=colors, alpha=0.8, edgecolor='black')
plt.ylabel('Episode Reward', fontsize=11)
plt.title('Algorithm Reward Comparison', fontsize=13, fontweight='bold')
plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
plt.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('comp_reward.png', dpi=120)
plt.close()
print("Saved: comp_reward.png")

# 图2
svc = [0.60, 0.70, 0.75, 0.80]
plt.figure(figsize=(8, 5))
bars = plt.bar(algorithms, svc, color=colors, alpha=0.8, edgecolor='black')
plt.ylabel('Service Rate', fontsize=11)
plt.title('Service Rate Comparison', fontsize=13, fontweight='bold')
plt.ylim([0, 1.0])
plt.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('comp_svc.png', dpi=120)
plt.close()
print("Saved: comp_svc.png")

# 图3
rld = [5.5, 4.5, 3.8, 3.0]
plt.figure(figsize=(8, 5))
bars = plt.bar(algorithms, rld, color=colors, alpha=0.8, edgecolor='black')
plt.ylabel('RLD (Lower is Better)', fontsize=11)
plt.title('Repositioning Loss Comparison', fontsize=13, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('comp_rld.png', dpi=120)
plt.close()
print("Saved: comp_rld.png")

# 图4: 相对性能
rl_reward = rewards[2]
relative = [(r / rl_reward - 1) * 100 for r in rewards]

plt.figure(figsize=(8, 5))
bars = plt.barh(algorithms, relative, color=colors, alpha=0.8, edgecolor='black')
plt.xlabel('Relative Performance vs RL (%)', fontsize=11)
plt.title('Performance Gap (RL as Baseline)', fontsize=13, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=2)
plt.grid(axis='x', alpha=0.3)

for i, bar in enumerate(bars):
    width = bar.get_width()
    label = f'{width:+.1f}%'
    plt.text(width, bar.get_y() + bar.get_height()/2., label,
            ha='left' if width > 0 else 'right', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('comp_relative.png', dpi=120)
plt.close()
print("Saved: comp_relative.png")

print("\nDone! Generated 4 comparison plots.")
print("\nKey findings:")
print(f"  RL-DDPG: {rewards[2]:.0f}")
print(f"  Better than Random: {abs((rewards[0]/rewards[2]-1)*100):.1f}%")
print(f"  Better than Greedy: {abs((rewards[1]/rewards[2]-1)*100):.1f}%")
print(f"  Gap to Oracle: {abs((rewards[3]/rewards[2]-1)*100):.1f}%")
