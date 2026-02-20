import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

output_dir = './outputs/marl_20260220-165650'

print('='*80)
print(f'LATEST TRAINING ANALYSIS: {output_dir}')
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
print(f'  First 10 avg: {np.mean(rewards[:10]):,.2f}')
print(f'  Last 10 avg: {np.mean(rewards[-10:]):,.2f}')
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
print(f'  Total change over {len(rewards)} eps: {slope * len(rewards):+,.2f}')
print(f'  R-squared: {r_value**2:.4f}')
print(f'  P-value: {p_value:.6f}')
if p_value < 0.05:
    if slope > 0:
        print(f'  => SIGNIFICANT UPWARD TREND ✓')
    else:
        print(f'  => SIGNIFICANT DOWNWARD TREND ✗')
else:
    print(f'  => No significant trend (FLAT)')

# Service rate
print(f'\n[SERVICE RATE]')
svc_nonzero = svc[svc > 0]
print(f'  Average: {np.mean(svc):.4f} ({np.mean(svc)*100:.2f}%)')
print(f'  Max: {np.max(svc):.4f} ({np.max(svc)*100:.2f}%)')
print(f'  Episodes with service: {len(svc_nonzero)}/{len(svc)} ({len(svc_nonzero)/len(svc)*100:.1f}%)')

# Loss statistics
print(f'\n[LOSS STATISTICS]')
print(f'  Critic Loss:')
print(f'    Mean: {np.mean(critic_losses):,.2e}')
print(f'    First 10: {np.mean(critic_losses[:10]):,.2e}')
print(f'    Last 10: {np.mean(critic_losses[-10:]):,.2e}')
print(f'    Min: {np.min(critic_losses):,.2e}')
print(f'    Max: {np.max(critic_losses):,.2e}')
print(f'  Actor Loss:')
print(f'    Mean: {np.mean(actor_losses):,.6f}')
print(f'    First 10: {np.mean(actor_losses[:10]):,.6f}')
print(f'    Last 10: {np.mean(actor_losses[-10:]):,.6f}')
print(f'    Min: {np.min(actor_losses):,.6f}')
print(f'    Max: {np.max(actor_losses):,.6f}')

# Check if losses are stable
critic_first_100 = critic_losses[:min(100, len(critic_losses))]
critic_last_100 = critic_losses[-min(100, len(critic_losses)):]
print(f'\n[LOSS TREND]')
print(f'  Critic - First 100 avg: {np.mean(critic_first_100):,.2e}')
print(f'  Critic - Last 100 avg: {np.mean(critic_last_100):,.2e}')
critic_change = (np.mean(critic_last_100) - np.mean(critic_first_100)) / np.mean(critic_first_100) * 100
print(f'  Critic change: {critic_change:+.1f}%')

if abs(critic_change) < 10:
    print(f'  => Critic loss STABLE (not learning)')
elif critic_change < -20:
    print(f'  => Critic loss DECREASING (good, learning)')
elif critic_change > 20:
    print(f'  => Critic loss INCREASING (bad, unstable)')

print('\n' + '='*80)
print('DIAGNOSIS & RECOMMENDATIONS')
print('='*80)

# Diagnosis
if p_value > 0.05 and abs(improvement) < 5:
    print('\n[PROBLEM] No learning detected:')
    print('  - Reward is flat across all episodes')
    print('  - No significant improvement trend')

if np.mean(critic_losses) > 1e8:
    print('\n[PROBLEM] Critic loss is extremely high:')
    print(f'  - Average: {np.mean(critic_losses):.2e}')
    print('  - This suggests value function is not learning properly')

if abs(np.mean(actor_losses)) > 1000:
    print('\n[PROBLEM] Actor loss is abnormally large:')
    print(f'  - Average: {np.mean(actor_losses):.2f}')
    print('  - This suggests policy gradient is not stable')

print('\n[RECOMMENDATIONS]')
print('1. Increase learning rates:')
print('   - critic_lr: 1e-5 -> 1e-4 (10x)')
print('   - actor_lr: 5e-6 -> 5e-5 (10x)')
print('2. Check reward scaling:')
print('   - Current rewards are ~1M, may need normalization')
print('3. Reduce batch size if memory allows:')
print('   - batch_size: 256 -> 128 (more frequent updates)')
print('4. Check if networks are updating:')
print('   - Verify gradients are not vanishing')

print('\n' + '='*80)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Reward curve
window = 20
ma_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
axes[0, 0].plot(rewards, alpha=0.3, label='Episode Reward', color='blue')
axes[0, 0].plot(range(window-1, len(rewards)), ma_rewards, linewidth=2, label=f'{window}-Episode MA', color='darkblue')
axes[0, 0].axhline(y=np.mean(rewards), color='r', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(rewards):,.0f}')
# Add trend line
axes[0, 0].plot(x, slope * x + intercept, 'g--', alpha=0.5, label=f'Trend: {slope:+.1f}/ep')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Reward')
axes[0, 0].set_title(f'Reward Curve ({len(rewards)} Episodes)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Service rate
axes[0, 1].plot(svc, alpha=0.6, color='green', marker='o', markersize=3)
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
plt.savefig('analysis_latest_600ep.png', dpi=150)
print(f'\nVisualization saved: analysis_latest_600ep.png')
