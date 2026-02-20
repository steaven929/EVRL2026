import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

output_dir = './outputs/marl_20260219-212040'

print('='*80)
print(f'Training Results Analysis: {output_dir}')
print('='*80)

# Load data
rewards = np.load(os.path.join(output_dir, 'rewards.npy'))
svc = np.load(os.path.join(output_dir, 'svc_history.npy'))
rld = np.load(os.path.join(output_dir, 'rld_history.npy'))
srd = np.load(os.path.join(output_dir, 'srd_history.npy'))
actor_losses = np.load(os.path.join(output_dir, 'actor_losses.npy'))
critic_losses = np.load(os.path.join(output_dir, 'critic_losses.npy'))

print(f'\nTotal Episodes: {len(rewards)}')

# Reward Statistics
print(f'\n[REWARD STATISTICS]')
print(f'  Mean: {np.mean(rewards):,.2f}')
print(f'  Std Dev: {np.std(rewards):,.2f}')
print(f'  Min: {np.min(rewards):,.2f}')
print(f'  Max: {np.max(rewards):,.2f}')
print(f'  CV: {np.std(rewards)/np.mean(rewards):.2%}')

# Phase comparison
phase_size = 100
n_phases = len(rewards) // phase_size
print(f'\n[REWARD BY PHASE (100-episode windows)]')
for i in range(n_phases):
    start = i * phase_size
    end = (i + 1) * phase_size
    phase_rewards = rewards[start:end]
    print(f'  Episodes {start+1:3d}-{end:3d}: Mean={np.mean(phase_rewards):>10,.2f}, Std={np.std(phase_rewards):>10,.2f}')

# Convergence check
first_half = rewards[:len(rewards)//2]
second_half = rewards[len(rewards)//2:]
print(f'\n[CONVERGENCE CHECK]')
print(f'  First half (1-{len(rewards)//2}): {np.mean(first_half):,.2f}')
print(f'  Second half ({len(rewards)//2+1}-{len(rewards)}): {np.mean(second_half):,.2f}')
improvement = (np.mean(second_half) - np.mean(first_half)) / np.mean(first_half) * 100
print(f'  Improvement: {improvement:+.2f}%')

# Trend analysis
x = np.arange(len(rewards))
slope, intercept, r_value, p_value, std_err = stats.linregress(x, rewards)
print(f'\n[LINEAR TREND]')
print(f'  Slope: {slope:+,.2f} per episode')
print(f'  R-squared: {r_value**2:.4f}')
print(f'  P-value: {p_value:.6f}')
if p_value < 0.05:
    if slope > 0:
        print(f'  => SIGNIFICANT UPWARD TREND')
    else:
        print(f'  => SIGNIFICANT DOWNWARD TREND')
else:
    print(f'  => No significant trend')

# Service rate
print(f'\n[SERVICE RATE]')
svc_nonzero = svc[svc > 0]
print(f'  Average: {np.mean(svc):.4f} ({np.mean(svc)*100:.2f}%)')
print(f'  Max: {np.max(svc):.4f} ({np.max(svc)*100:.2f}%)')
print(f'  Episodes with service: {len(svc_nonzero)}/{len(svc)} ({len(svc_nonzero)/len(svc)*100:.1f}%)')

# Loss statistics
print(f'\n[LOSS STATISTICS]')
print(f'  Critic Loss - Mean: {np.mean(critic_losses):,.2e}, Final: {critic_losses[-1]:,.2e}')
print(f'  Actor Loss - Mean: {np.mean(actor_losses):,.6f}, Final: {actor_losses[-1]:,.6f}')

# Check if losses are exploding or stable
critic_first_100 = critic_losses[:100]
critic_last_100 = critic_losses[-100:]
print(f'  Critic Loss - First 100: {np.mean(critic_first_100):,.2e}')
print(f'  Critic Loss - Last 100: {np.mean(critic_last_100):,.2e}')
if np.mean(critic_last_100) > np.mean(critic_first_100) * 2:
    print(f'  => WARNING: Critic loss INCREASING (unstable)')
elif np.mean(critic_last_100) < np.mean(critic_first_100) * 0.5:
    print(f'  => Critic loss DECREASING (learning)')
else:
    print(f'  => Critic loss STABLE')

print('\n' + '='*80)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Reward curve
window = 20
ma_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
axes[0, 0].plot(rewards, alpha=0.3, label='Episode Reward')
axes[0, 0].plot(range(window-1, len(rewards)), ma_rewards, linewidth=2, label=f'{window}-Episode MA')
axes[0, 0].axhline(y=np.mean(rewards), color='r', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(rewards):,.0f}')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Reward')
axes[0, 0].set_title(f'Reward Curve ({len(rewards)} Episodes)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Service rate
axes[0, 1].plot(svc, alpha=0.6, color='green')
axes[0, 1].axhline(y=np.mean(svc), color='r', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(svc):.4f}')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Service Rate')
axes[0, 1].set_title('Service Rate Over Time')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Critic loss
axes[1, 0].plot(critic_losses, alpha=0.6, color='orange')
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Critic Loss')
axes[1, 0].set_title('Critic Loss Curve')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Actor loss
axes[1, 1].plot(actor_losses, alpha=0.6, color='purple')
axes[1, 1].set_xlabel('Episode')
axes[1, 1].set_ylabel('Actor Loss')
axes[1, 1].set_title('Actor Loss Curve')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis_800ep.png', dpi=150)
print(f'\nVisualization saved: analysis_800ep.png')
