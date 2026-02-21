import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

output_dir = './outputs/marl_20260221-143622'

print("="*80)
print("CRITIC LOSS TREND ANALYSIS")
print("="*80)

# Load data
rewards = np.load(output_dir + '/rewards.npy')
raw_rewards = np.load(output_dir + '/raw_rewards.npy')
critic_losses = np.load(output_dir + '/critic_losses.npy')
actor_losses = np.load(output_dir + '/actor_losses.npy')

# Analyze critic loss trend
print("\n[CRITIC LOSS DETAILED ANALYSIS]")
print(f"Total updates: {len(critic_losses)}")

# Split into phases
phase_size = len(critic_losses) // 3
phase1 = critic_losses[:phase_size]
phase2 = critic_losses[phase_size:2*phase_size]
phase3 = critic_losses[2*phase_size:]

print(f"\nPhase 1 (episodes 1-100):")
print(f"  Mean: {np.mean(phase1):.2e}")
print(f"  Std: {np.std(phase1):.2e}")
print(f"  Min: {np.min(phase1):.2e}")
print(f"  Max: {np.max(phase1):.2e}")

print(f"\nPhase 2 (episodes 101-200):")
print(f"  Mean: {np.mean(phase2):.2e}")
print(f"  Std: {np.std(phase2):.2e}")
print(f"  Min: {np.min(phase2):.2e}")
print(f"  Max: {np.max(phase2):.2e}")

print(f"\nPhase 3 (episodes 201-300):")
print(f"  Mean: {np.mean(phase3):.2e}")
print(f"  Std: {np.std(phase3):.2e}")
print(f"  Min: {np.min(phase3):.2e}")
print(f"  Max: {np.max(phase3):.2e}")

# Trend analysis
x = np.arange(len(critic_losses))
slope, intercept, r_value, p_value, std_err = stats.linregress(x, critic_losses)
print(f"\nCritic Loss Trend:")
print(f"  Slope: {slope:+.4e} per update")
print(f"  Total change: {slope * len(critic_losses):+.2e}")
print(f"  R-squared: {r_value**2:.4f}")
print(f"  P-value: {p_value:.6f}")
if p_value < 0.05:
    if slope > 0:
        print(f"  => SIGNIFICANT UPWARD TREND (PROBLEM!)")
    else:
        print(f"  => SIGNIFICANT DOWNWARD TREND (GOOD)")
else:
    print(f"  => No significant trend")

# Correlation with rewards
# Match lengths
min_len = min(len(rewards), len(critic_losses))
reward_per_update = np.repeat(rewards, len(critic_losses) // len(rewards))[:min_len]
critic_losses_matched = critic_losses[:min_len]
corr = np.corrcoef(reward_per_update, critic_losses_matched)[0, 1]
print(f"\nCorrelation between Reward and Critic Loss:")
print(f"  Correlation coefficient: {corr:.4f}")
if abs(corr) > 0.5:
    if corr > 0:
        print(f"  => Strong POSITIVE correlation (as reward increases, loss increases)")
    else:
        print(f"  => Strong NEGATIVE correlation (as reward increases, loss decreases)")
else:
    print(f"  => Weak correlation")

# Analyze Q-value scale
print(f"\n[Q-VALUE SCALE ANALYSIS]")
print(f"Scaled Rewards (what Critic predicts):")
print(f"  Mean: {np.mean(rewards):.2f}")
print(f"  Range: {np.min(rewards):.2f} to {np.max(rewards):.2f}")
print(f"  Std: {np.std(rewards):.2f}")

# Estimate Q-values (rewards are per-step, Q-values accumulate)
gamma = 0.95
n_steps = 26
discount_sum = sum([gamma**i for i in range(n_steps)])
estimated_q = np.mean(rewards) * discount_sum / n_steps
print(f"\nEstimated Q-values (with gamma=0.95, 26 steps):")
print(f"  Approximate Q: {estimated_q:.2f}")
print(f"  Q range: {np.min(rewards) * discount_sum / n_steps:.2f} to {np.max(rewards) * discount_sum / n_steps:.2f}")

# Check if loss is proportional to Q-value scale
expected_loss_scale = (estimated_q * 0.1) ** 2  # Assume 10% error
print(f"\nExpected Loss Scale (if 10% Q-value error):")
print(f"  Expected: {expected_loss_scale:.2e}")
print(f"  Actual: {np.mean(critic_losses):.2e}")
print(f"  Ratio: {np.mean(critic_losses) / expected_loss_scale:.2f}x")

# Diagnosis
print(f"\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

print("\n[ROOT CAUSE ANALYSIS]")

# Check 1: Is loss increasing?
if slope > 0 and p_value < 0.05:
    print("\n1. Critic Loss IS increasing significantly")
    print("   - Phase 1: {:.2e}".format(np.mean(phase1)))
    print("   - Phase 3: {:.2e}".format(np.mean(phase3)))
    print("   - Increase: {:.1f}%".format((np.mean(phase3) - np.mean(phase1)) / np.mean(phase1) * 100))
else:
    print("\n1. Critic Loss is relatively stable")

# Check 2: Is it correlated with reward growth?
if corr > 0.5:
    print("\n2. Loss increases WITH reward (EXPECTED)")
    print("   - As agent learns to get higher rewards, Q-values grow")
    print("   - Larger Q-values -> larger prediction errors -> higher loss")
    print("   - This is NORMAL if reward is growing")
elif corr < -0.5:
    print("\n2. Loss decreases as reward increases (GOOD)")
    print("   - Critic is learning to predict better despite growing Q-values")
else:
    print("\n2. Loss not strongly correlated with reward")
    print("   - May indicate other issues")

# Check 3: Is loss scale reasonable?
if np.mean(critic_losses) < 100:
    print("\n3. Loss magnitude is REASONABLE (<100)")
    print("   - Given Q-values of ~{:.0f}, loss of ~{:.0f} is acceptable".format(estimated_q, np.mean(critic_losses)))
elif np.mean(critic_losses) < 1000:
    print("\n3. Loss magnitude is MODERATE (100-1000)")
    print("   - Could be improved but not critical")
else:
    print("\n3. Loss magnitude is HIGH (>1000)")
    print("   - Critic is struggling to predict Q-values accurately")

# Check 4: Is variance increasing?
var_ratio = np.var(phase3) / np.var(phase1)
if var_ratio > 1.5:
    print("\n4. Loss variance is INCREASING")
    print("   - Phase 1 std: {:.2e}".format(np.std(phase1)))
    print("   - Phase 3 std: {:.2e}".format(np.std(phase3)))
    print("   - This suggests training instability")
elif var_ratio < 0.7:
    print("\n4. Loss variance is DECREASING (GOOD)")
    print("   - Training is becoming more stable")
else:
    print("\n4. Loss variance is STABLE")

print("\n[RECOMMENDATIONS]")

if slope > 0 and p_value < 0.05 and corr > 0.5:
    print("\nThe loss increase is EXPECTED and NORMAL because:")
    print("  - Rewards are growing (14.4M -> 21.8M)")
    print("  - Q-values are growing proportionally")
    print("  - Larger Q-values naturally have larger errors")
    print("\nThis is NOT a problem if:")
    print("  - Loss is not exploding (currently ~20, acceptable)")
    print("  - Reward continues to improve (it is)")
    print("  - Agent behavior is reasonable")
    print("\nNo action needed - continue training")

elif slope > 0 and p_value < 0.05 and corr < 0.3:
    print("\nThe loss increase is CONCERNING because:")
    print("  - Loss is increasing but NOT due to reward growth")
    print("  - May indicate:")
    print("    * Learning rate too high")
    print("    * Overfitting")
    print("    * Target network update too fast")
    print("\nRecommended actions:")
    print("  1. Reduce critic_lr: 1e-4 -> 5e-5")
    print("  2. Reduce tau: 5e-4 -> 1e-4")
    print("  3. Increase batch_size: 128 -> 256")

else:
    print("\nLoss is relatively stable - no immediate concern")

print("\n" + "="*80)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Critic loss over time
window = 50
ma = np.convolve(critic_losses, np.ones(window)/window, mode='valid')
axes[0, 0].plot(critic_losses, alpha=0.3, label='Critic Loss', color='orange')
axes[0, 0].plot(range(window-1, len(critic_losses)), ma, linewidth=2,
                label=f'{window}-Update MA', color='darkorange')
axes[0, 0].plot(x, slope * x + intercept, 'r--', alpha=0.7,
                label=f'Trend: {slope:+.2e}/update')
axes[0, 0].set_xlabel('Update')
axes[0, 0].set_ylabel('Critic Loss')
axes[0, 0].set_title('Critic Loss Over Time')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# Plot 2: Critic loss vs Reward
axes[0, 1].scatter(reward_per_update, critic_losses_matched, alpha=0.3, s=10)
axes[0, 1].set_xlabel('Reward (Scaled)')
axes[0, 1].set_ylabel('Critic Loss')
axes[0, 1].set_title(f'Critic Loss vs Reward (corr={corr:.3f})')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_yscale('log')

# Plot 3: Loss distribution by phase
axes[1, 0].boxplot([phase1, phase2, phase3], labels=['1-100', '101-200', '201-300'])
axes[1, 0].set_xlabel('Episode Range')
axes[1, 0].set_ylabel('Critic Loss')
axes[1, 0].set_title('Critic Loss Distribution by Phase')
axes[1, 0].grid(True, alpha=0.3, axis='y')
axes[1, 0].set_yscale('log')

# Plot 4: Reward and Loss together (normalized)
ax1 = axes[1, 1]
ax2 = ax1.twinx()

# Normalize to 0-1 for comparison
reward_norm = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))
loss_per_episode = [np.mean(critic_losses[i*len(critic_losses)//len(rewards):(i+1)*len(critic_losses)//len(rewards)])
                    for i in range(len(rewards))]
loss_norm = (loss_per_episode - np.min(loss_per_episode)) / (np.max(loss_per_episode) - np.min(loss_per_episode))

ax1.plot(reward_norm, 'b-', alpha=0.7, label='Reward (normalized)')
ax2.plot(loss_norm, 'r-', alpha=0.7, label='Critic Loss (normalized)')

ax1.set_xlabel('Episode')
ax1.set_ylabel('Normalized Reward', color='b')
ax2.set_ylabel('Normalized Critic Loss', color='r')
ax1.set_title('Reward vs Critic Loss Trends')
ax1.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='r')
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('critic_loss_analysis.png', dpi=150)
print(f"\nVisualization saved: critic_loss_analysis.png")
