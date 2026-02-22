import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

output_dir = './outputs/marl_20260221-172845'

print("="*80)
print("COMPREHENSIVE TRAINING ANALYSIS (600 Episodes)")
print("="*80)

# Load data
rewards = np.load(output_dir + '/rewards.npy')
raw_rewards = np.load(output_dir + '/raw_rewards.npy')
svc = np.load(output_dir + '/svc_history.npy')
rld = np.load(output_dir + '/rld_history.npy')
srd = np.load(output_dir + '/srd_history.npy')
actor_losses = np.load(output_dir + '/actor_losses.npy')
critic_losses = np.load(output_dir + '/critic_losses.npy')

print(f"\nTotal Episodes: {len(rewards)}")
print(f"Total Updates: {len(critic_losses)}")

# ========== REWARD ANALYSIS ==========
print(f"\n{'='*80}")
print("REWARD ANALYSIS")
print("="*80)

print(f"\nScaled Rewards (after /10000):")
print(f"  Mean: {np.mean(rewards):,.2f}")
print(f"  Std: {np.std(rewards):,.2f}")
print(f"  Min: {np.min(rewards):,.2f}")
print(f"  Max: {np.max(rewards):,.2f}")

print(f"\nRaw Rewards (before scaling):")
print(f"  Mean: {np.mean(raw_rewards):,.2f}")
print(f"  Std: {np.std(raw_rewards):,.2f}")
print(f"  Min: {np.min(raw_rewards):,.2f}")
print(f"  Max: {np.max(raw_rewards):,.2f}")

# Trend analysis
x = np.arange(len(rewards))
slope, intercept, r_value, p_value, std_err = stats.linregress(x, rewards)
print(f"\nReward Trend:")
print(f"  Slope: {slope:+.4f} per episode")
print(f"  Total change: {slope * len(rewards):+,.2f}")
print(f"  R-squared: {r_value**2:.4f}")
print(f"  P-value: {p_value:.6f}")
if p_value < 0.05:
    if slope > 0:
        print(f"  => SIGNIFICANT UPWARD TREND")
    else:
        print(f"  => SIGNIFICANT DOWNWARD TREND")
else:
    print(f"  => No significant trend (STABLE)")

# Phase analysis
print(f"\n[REWARD BY PHASE (100-episode windows)]")
for i in range(0, len(rewards), 100):
    end = min(i + 100, len(rewards))
    phase_rewards = rewards[i:end]
    print(f"  Episodes {i+1:3d}-{end:3d}: {np.mean(phase_rewards):>10,.2f} (std: {np.std(phase_rewards):>8,.2f})")

# Compare first and last 100
first_100 = rewards[:100]
last_100 = rewards[-100:]
improvement = (np.mean(last_100) - np.mean(first_100)) / np.mean(first_100) * 100
print(f"\nFirst 100 vs Last 100:")
print(f"  First 100: {np.mean(first_100):,.2f}")
print(f"  Last 100: {np.mean(last_100):,.2f}")
print(f"  Improvement: {improvement:+.1f}%")

# ========== SERVICE RATE ANALYSIS ==========
print(f"\n{'='*80}")
print("SERVICE RATE ANALYSIS")
print("="*80)

svc_nonzero = svc[svc > 0]
print(f"\nOverall:")
print(f"  Average: {np.mean(svc):.4f} ({np.mean(svc)*100:.2f}%)")
print(f"  Max: {np.max(svc):.4f} ({np.max(svc)*100:.2f}%)")
print(f"  Episodes with service: {len(svc_nonzero)}/{len(svc)} ({len(svc_nonzero)/len(svc)*100:.1f}%)")
if len(svc_nonzero) > 0:
    print(f"  Avg (service episodes only): {np.mean(svc_nonzero):.4f} ({np.mean(svc_nonzero)*100:.2f}%)")

print(f"\n[SERVICE RATE BY PHASE]")
for i in range(0, len(svc), 100):
    end = min(i + 100, len(svc))
    phase_svc = svc[i:end]
    phase_svc_nonzero = phase_svc[phase_svc > 0]
    print(f"  Episodes {i+1:3d}-{end:3d}: {np.mean(phase_svc)*100:>6.2f}% ({len(phase_svc_nonzero):2d}/{len(phase_svc)} episodes)")

# Service rate trend
svc_slope, svc_intercept, svc_r, svc_p, _ = stats.linregress(x, svc)
print(f"\nService Rate Trend:")
print(f"  Slope: {svc_slope:+.6f} per episode")
print(f"  P-value: {svc_p:.6f}")
if svc_p < 0.05:
    if svc_slope > 0:
        print(f"  => Service rate is INCREASING")
    else:
        print(f"  => Service rate is DECREASING")
else:
    print(f"  => Service rate is STABLE")

# ========== LOSS ANALYSIS ==========
print(f"\n{'='*80}")
print("LOSS ANALYSIS")
print("="*80)

print(f"\nCritic Loss:")
print(f"  Mean: {np.mean(critic_losses):,.2e}")
print(f"  Std: {np.std(critic_losses):,.2e}")
print(f"  Min: {np.min(critic_losses):,.2e}")
print(f"  Max: {np.max(critic_losses):,.2e}")

# Critic loss by phase
phase_size = len(critic_losses) // 6
for i in range(6):
    start = i * phase_size
    end = (i + 1) * phase_size if i < 5 else len(critic_losses)
    phase_loss = critic_losses[start:end]
    print(f"  Episodes {i*100+1:3d}-{(i+1)*100:3d}: {np.mean(phase_loss):>10,.2e}")

# Critic loss trend
loss_x = np.arange(len(critic_losses))
loss_slope, loss_intercept, loss_r, loss_p, _ = stats.linregress(loss_x, critic_losses)
print(f"\nCritic Loss Trend:")
print(f"  Slope: {loss_slope:+.4e} per update")
print(f"  Total change: {loss_slope * len(critic_losses):+,.2e}")
print(f"  P-value: {loss_p:.6f}")
if loss_p < 0.05:
    if loss_slope > 0:
        print(f"  => Loss is INCREASING")
    else:
        print(f"  => Loss is DECREASING")

print(f"\nActor Loss:")
print(f"  Mean: {np.mean(actor_losses):,.2f}")
print(f"  Min: {np.min(actor_losses):,.2f}")
print(f"  Max: {np.max(actor_losses):,.2f}")

# ========== COMPARISON WITH PREVIOUS RUNS ==========
print(f"\n{'='*80}")
print("COMPARISON WITH PREVIOUS TRAINING")
print("="*80)

print(f"\n100 Episodes:")
print(f"  Avg Raw Reward: 14,369,921")
print(f"  Service Rate: 0.97%")
print(f"  Critic Loss: 17.5")

print(f"\n300 Episodes:")
print(f"  Avg Raw Reward: 21,831,124")
print(f"  Service Rate: 1.53%")
print(f"  Critic Loss: 20.3")

print(f"\n600 Episodes (current):")
print(f"  Avg Raw Reward: {np.mean(raw_rewards):,.2f}")
print(f"  Service Rate: {np.mean(svc)*100:.2f}%")
print(f"  Critic Loss: {np.mean(critic_losses):,.2e}")

# ========== DIAGNOSIS ==========
print(f"\n{'='*80}")
print("DIAGNOSIS & KEY INSIGHTS")
print("="*80)

# Check if reward is still growing
if p_value < 0.05 and slope > 0:
    print("\n[REWARD] Still increasing after 600 episodes")
    print(f"  - Growth rate: {slope:+.2f} per episode")
    print(f"  - Agent continues to learn")
elif abs(slope * len(rewards)) < np.std(rewards):
    print("\n[REWARD] Converged/Stable")
    print(f"  - No significant trend")
    print(f"  - Agent may have found optimal strategy")

# Check service rate
if np.mean(svc) < 0.02:
    print("\n[SERVICE RATE] CRITICALLY LOW (<2%)")
    print(f"  - Only {len(svc_nonzero)}/{len(svc)} episodes provided service")
    print(f"  - Agent is ignoring service orders")
    print(f"  - Charging bonuses dominate reward function")
elif np.mean(svc) < 0.05:
    print("\n[SERVICE RATE] LOW (2-5%)")
    print(f"  - Some service but insufficient")
elif np.mean(svc) > 0.1:
    print("\n[SERVICE RATE] GOOD (>10%)")
    print(f"  - Agent balancing charging and service")

# Check if service rate improved
svc_first_100 = svc[:100]
svc_last_100 = svc[-100:]
svc_change = (np.mean(svc_last_100) - np.mean(svc_first_100)) / max(np.mean(svc_first_100), 0.001) * 100
if abs(svc_change) > 20:
    if svc_change > 0:
        print(f"  - Service rate IMPROVED by {svc_change:.1f}% (first 100 to last 100)")
    else:
        print(f"  - Service rate DECLINED by {abs(svc_change):.1f}% (first 100 to last 100)")

# Check loss stability
if loss_p < 0.05 and loss_slope > 0:
    print("\n[CRITIC LOSS] INCREASING")
    print(f"  - May indicate training instability")
    print(f"  - Consider reducing learning rate")
elif loss_p < 0.05 and loss_slope < 0:
    print("\n[CRITIC LOSS] DECREASING (GOOD)")
    print(f"  - Critic is learning better predictions")
else:
    print("\n[CRITIC LOSS] STABLE")

# Overall assessment
print(f"\n{'='*80}")
print("OVERALL ASSESSMENT")
print("="*80)

if np.mean(raw_rewards) > 20000000 and np.mean(svc) < 0.02:
    print("\n[CONCLUSION] Agent learned CHARGING-ONLY strategy")
    print("  - High rewards (~{:.1f}M) from charging bonuses".format(np.mean(raw_rewards)/1e6))
    print("  - Service rate critically low ({:.2f}%)".format(np.mean(svc)*100))
    print("  - Agent rationally maximizes reward by charging")
    print("  - Reward function imbalance confirmed")
    print("\n[ACTION REQUIRED] Adjust reward weights:")
    print("  1. Reduce charging bonuses (15000 -> 5000)")
    print("  2. Increase service rewards (/10 -> /2)")
    print("  3. Or both")
elif np.mean(svc) > 0.05:
    print("\n[CONCLUSION] Agent learning to balance")
    print("  - Service rate improving")
    print("  - Continue training")
else:
    print("\n[CONCLUSION] Agent exploring strategies")
    print("  - May need more training or reward adjustment")

print("\n" + "="*80)

# Visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Reward curve
ax1 = fig.add_subplot(gs[0, :2])
window = 30
ma = np.convolve(raw_rewards, np.ones(window)/window, mode='valid')
ax1.plot(raw_rewards, alpha=0.3, label='Raw Reward', color='blue')
ax1.plot(range(window-1, len(raw_rewards)), ma, linewidth=2,
         label=f'{window}-Episode MA', color='darkblue')
ax1.axhline(y=np.mean(raw_rewards), color='green', linestyle='--', alpha=0.5,
            label=f'Mean: {np.mean(raw_rewards):,.0f}')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Raw Reward')
ax1.set_title('Reward Curve (600 Episodes)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Reward distribution
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(raw_rewards/1e6, bins=30, alpha=0.7, color='blue', edgecolor='black')
ax2.axvline(x=np.mean(raw_rewards)/1e6, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Raw Reward (Millions)')
ax2.set_ylabel('Frequency')
ax2.set_title('Reward Distribution')
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Service rate
ax3 = fig.add_subplot(gs[1, :2])
ax3.plot(svc*100, alpha=0.6, color='green', marker='o', markersize=2)
ax3.axhline(y=np.mean(svc)*100, color='r', linestyle='--', alpha=0.5,
            label=f'Mean: {np.mean(svc)*100:.2f}%')
ax3.set_xlabel('Episode')
ax3.set_ylabel('Service Rate (%)')
ax3.set_title('Service Rate Over Time')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Service rate by phase
ax4 = fig.add_subplot(gs[1, 2])
phase_svc_means = []
phase_labels = []
for i in range(0, len(svc), 100):
    end = min(i + 100, len(svc))
    phase_svc_means.append(np.mean(svc[i:end]) * 100)
    phase_labels.append(f'{i+1}-{end}')
ax4.bar(range(len(phase_svc_means)), phase_svc_means, color='green', alpha=0.7)
ax4.set_xlabel('Episode Range')
ax4.set_ylabel('Service Rate (%)')
ax4.set_title('Service Rate by Phase')
ax4.set_xticks(range(len(phase_svc_means)))
ax4.set_xticklabels(phase_labels, rotation=45, ha='right')
ax4.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(phase_svc_means):
    ax4.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

# Plot 5: Critic loss
ax5 = fig.add_subplot(gs[2, :2])
window_loss = 100
ma_loss = np.convolve(critic_losses, np.ones(window_loss)/window_loss, mode='valid')
ax5.plot(critic_losses, alpha=0.2, label='Critic Loss', color='orange')
ax5.plot(range(window_loss-1, len(critic_losses)), ma_loss, linewidth=2,
         label=f'{window_loss}-Update MA', color='darkorange')
ax5.set_xlabel('Update')
ax5.set_ylabel('Critic Loss')
ax5.set_title('Critic Loss Curve')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.set_yscale('log')

# Plot 6: Reward vs Service Rate
ax6 = fig.add_subplot(gs[2, 2])
ax6.scatter(rewards, svc*100, alpha=0.5, s=20)
ax6.set_xlabel('Reward (Scaled)')
ax6.set_ylabel('Service Rate (%)')
ax6.set_title('Reward vs Service Rate')
ax6.grid(True, alpha=0.3)

plt.savefig('comprehensive_600ep_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nVisualization saved: comprehensive_600ep_analysis.png")
