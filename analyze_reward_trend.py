import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = './outputs/marl_20260220-134655'

# Load data
rewards = np.load(os.path.join(output_dir, 'rewards.npy'))

# Calculate moving average
window = 10
ma_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')

# Create figure
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Reward curve
axes[0].plot(rewards, alpha=0.3, label='Episode Reward')
axes[0].plot(range(window-1, len(rewards)), ma_rewards, linewidth=2, label=f'{window}-Episode MA')
axes[0].axhline(y=np.mean(rewards), color='r', linestyle='--', label=f'Mean: {np.mean(rewards):,.0f}')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Reward')
axes[0].set_title('MARL Training Reward Curve (150 Episodes)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Reward distribution by phase
phase_size = 30
n_phases = len(rewards) // phase_size
phase_means = []
phase_labels = []
for i in range(n_phases):
    start = i * phase_size
    end = (i + 1) * phase_size
    phase_rewards = rewards[start:end]
    phase_means.append(np.mean(phase_rewards))
    phase_labels.append(f'{start+1}-{end}')

axes[1].bar(range(len(phase_means)), phase_means, color='steelblue', alpha=0.7)
axes[1].set_xlabel('Episode Range')
axes[1].set_ylabel('Average Reward')
axes[1].set_title('Average Reward by Phase (30-episode windows)')
axes[1].set_xticks(range(len(phase_means)))
axes[1].set_xticklabels(phase_labels)
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, v in enumerate(phase_means):
    axes[1].text(i, v, f'{v:,.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'reward_trend_analysis.png'), dpi=150)
print(f'Saved: {os.path.join(output_dir, "reward_trend_analysis.png")}')

# Print statistics
print('\n' + '='*80)
print('REWARD TREND ANALYSIS')
print('='*80)
print(f'\nOverall Statistics:')
print(f'  Mean Reward: {np.mean(rewards):,.2f}')
print(f'  Std Dev: {np.std(rewards):,.2f}')
print(f'  Min: {np.min(rewards):,.2f}')
print(f'  Max: {np.max(rewards):,.2f}')
print(f'  Coefficient of Variation: {np.std(rewards)/np.mean(rewards):.2%}')

print(f'\nPhase Comparison:')
for i in range(n_phases):
    start = i * phase_size
    end = (i + 1) * phase_size
    phase_rewards = rewards[start:end]
    print(f'  Episodes {start+1:3d}-{end:3d}: Mean={np.mean(phase_rewards):>10,.2f}, Std={np.std(phase_rewards):>10,.2f}')

# Check for convergence
first_half = rewards[:len(rewards)//2]
second_half = rewards[len(rewards)//2:]
print(f'\nConvergence Check:')
print(f'  First half mean: {np.mean(first_half):,.2f}')
print(f'  Second half mean: {np.mean(second_half):,.2f}')
print(f'  Improvement: {(np.mean(second_half) - np.mean(first_half))/np.mean(first_half)*100:+.2f}%')

# Trend analysis
from scipy import stats
x = np.arange(len(rewards))
slope, intercept, r_value, p_value, std_err = stats.linregress(x, rewards)
print(f'\nLinear Trend:')
print(f'  Slope: {slope:+,.2f} per episode')
print(f'  R-squared: {r_value**2:.4f}')
print(f'  P-value: {p_value:.4f}')
if p_value < 0.05:
    if slope > 0:
        print(f'  => Significant UPWARD trend')
    else:
        print(f'  => Significant DOWNWARD trend')
else:
    print(f'  => No significant trend (stable)')

print('\n' + '='*80)
