import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

output_dir = './outputs/marl_20260221-143622'

print("="*80)
print("SHAPING MODE TRAINING RESULTS (300 Episodes)")
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

# Reward Analysis
print(f"\n[REWARD ANALYSIS]")
print(f"Scaled Rewards (after /10000):")
print(f"  Mean: {np.mean(rewards):,.2f}")
print(f"  Std: {np.std(rewards):,.2f}")
print(f"  Min: {np.min(rewards):,.2f}")
print(f"  Max: {np.max(rewards):,.2f}")
print(f"  First 50 avg: {np.mean(rewards[:50]):,.2f}")
print(f"  Last 50 avg: {np.mean(rewards[-50:]):,.2f}")

print(f"\nRaw Rewards (before scaling):")
print(f"  Mean: {np.mean(raw_rewards):,.2f}")
print(f"  Std: {np.std(raw_rewards):,.2f}")
print(f"  Min: {np.min(raw_rewards):,.2f}")
print(f"  Max: {np.max(raw_rewards):,.2f}")
print(f"  First 50 avg: {np.mean(raw_rewards[:50]):,.2f}")
print(f"  Last 50 avg: {np.mean(raw_rewards[-50:]):,.2f}")

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
        print(f"  => SIGNIFICANT UPWARD TREND!")
    else:
        print(f"  => SIGNIFICANT DOWNWARD TREND")
else:
    print(f"  => No significant trend")

# Phase analysis
print(f"\n[PHASE ANALYSIS (100-episode windows)]")
for i in range(0, len(rewards), 100):
    end = min(i + 100, len(rewards))
    phase_rewards = rewards[i:end]
    phase_svc = svc[i:end]
    print(f"  Episodes {i+1:3d}-{end:3d}:")
    print(f"    Reward: {np.mean(phase_rewards):>10,.2f} (std: {np.std(phase_rewards):,.2f})")
    print(f"    Service Rate: {np.mean(phase_svc)*100:>6.2f}%")

# Loss Analysis
print(f"\n[LOSS ANALYSIS]")
print(f"Critic Loss:")
print(f"  Mean: {np.mean(critic_losses):,.2e}")
print(f"  First 50: {np.mean(critic_losses[:50]):,.2e}")
print(f"  Last 50: {np.mean(critic_losses[-50:]):,.2e}")
print(f"  Min: {np.min(critic_losses):,.2e}")
print(f"  Max: {np.max(critic_losses):,.2e}")

first_third = critic_losses[:len(critic_losses)//3]
last_third = critic_losses[-len(critic_losses)//3:]
critic_change = (np.mean(last_third) - np.mean(first_third)) / np.mean(first_third) * 100
print(f"  First 1/3 avg: {np.mean(first_third):,.2e}")
print(f"  Last 1/3 avg: {np.mean(last_third):,.2e}")
print(f"  Change: {critic_change:+.1f}%")

print(f"\nActor Loss:")
print(f"  Mean: {np.mean(actor_losses):,.2f}")
print(f"  First 50: {np.mean(actor_losses[:50]):,.2f}")
print(f"  Last 50: {np.mean(actor_losses[-50:]):,.2f}")
print(f"  Min: {np.min(actor_losses):,.2f}")
print(f"  Max: {np.max(actor_losses):,.2f}")

# Service Rate
print(f"\n[SERVICE RATE ANALYSIS]")
svc_nonzero = svc[svc > 0]
print(f"  Overall Average: {np.mean(svc):.4f} ({np.mean(svc)*100:.2f}%)")
print(f"  Max: {np.max(svc):.4f} ({np.max(svc)*100:.2f}%)")
print(f"  Episodes with service: {len(svc_nonzero)}/{len(svc)} ({len(svc_nonzero)/len(svc)*100:.1f}%)")
if len(svc_nonzero) > 0:
    print(f"  Avg (service episodes only): {np.mean(svc_nonzero):.4f} ({np.mean(svc_nonzero)*100:.2f}%)")

# Service rate by phase
print(f"\n  Service Rate by Phase:")
for i in range(0, len(svc), 100):
    end = min(i + 100, len(svc))
    phase_svc = svc[i:end]
    phase_svc_nonzero = phase_svc[phase_svc > 0]
    print(f"    Episodes {i+1:3d}-{end:3d}: {np.mean(phase_svc)*100:>6.2f}% ({len(phase_svc_nonzero)}/{len(phase_svc)} episodes)")

# Comparison with 100-episode training
print(f"\n[COMPARISON: 100 vs 300 Episodes]")
print(f"\n100 Episodes (previous):")
print(f"  Avg Raw Reward: 14,369,921")
print(f"  Service Rate: 0.97%")
print(f"  Critic Loss: 17.5")

print(f"\n300 Episodes (current):")
print(f"  Avg Raw Reward: {np.mean(raw_rewards):,.2f}")
print(f"  Service Rate: {np.mean(svc)*100:.2f}%")
print(f"  Critic Loss: {np.mean(critic_losses):,.2e}")

# Diagnosis
print(f"\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if np.mean(raw_rewards) > 10000000:
    print("\n[GOOD] Rewards remain positive and high")
else:
    print("\n[CONCERN] Rewards decreased")

if p_value < 0.05:
    if slope > 0:
        print("[GOOD] Reward is still increasing")
    else:
        print("[CONCERN] Reward is decreasing")
else:
    print("[NEUTRAL] Reward is stable (no significant trend)")

if critic_change < -20:
    print("[GOOD] Critic Loss is decreasing (active learning)")
elif abs(critic_change) < 10:
    print("[NEUTRAL] Critic Loss is stable")
else:
    print("[CONCERN] Critic Loss is increasing")

# Check if service rate improved
svc_first_100 = svc[:100]
svc_last_100 = svc[-100:]
svc_improvement = (np.mean(svc_last_100) - np.mean(svc_first_100)) / max(np.mean(svc_first_100), 0.001) * 100

if np.mean(svc) > 0.05:
    print(f"[GOOD] Service rate improved to {np.mean(svc)*100:.2f}%")
elif svc_improvement > 50:
    print(f"[GOOD] Service rate improving ({svc_improvement:+.1f}% from first 100 to last 100)")
elif np.mean(svc) > 0.02:
    print(f"[NEUTRAL] Service rate is {np.mean(svc)*100:.2f}% (moderate)")
else:
    print(f"[CONCERN] Service rate is still low ({np.mean(svc)*100:.2f}%)")

print("\n[KEY INSIGHTS]")
# Check if agent is stuck in charging-only strategy
if np.mean(svc) < 0.02 and np.mean(raw_rewards) > 10000000:
    print("  - Agent learned to maximize reward through charging")
    print("  - Charging bonuses dominate the reward function")
    print("  - Service is being ignored (not profitable enough)")
    print("  => RECOMMENDATION: Reduce charging bonuses or increase service rewards")
elif np.mean(svc) > 0.05:
    print("  - Agent is learning to balance charging and service")
    print("  - Continue training to see further improvement")
else:
    print("  - Agent is exploring different strategies")
    print("  - May need more training or reward adjustment")

print("\n" + "="*80)

# Visualization
fig, axes = plt.subplots(3, 2, figsize=(16, 12))

# Plot 1: Raw rewards
window = 20
ma = np.convolve(raw_rewards, np.ones(window)/window, mode='valid')
axes[0, 0].plot(raw_rewards, alpha=0.3, label='Raw Reward', color='blue')
axes[0, 0].plot(range(window-1, len(raw_rewards)), ma, linewidth=2,
                label=f'{window}-Episode MA', color='darkblue')
axes[0, 0].axhline(y=np.mean(raw_rewards), color='green', linestyle='--', alpha=0.5,
                   label=f'Mean: {np.mean(raw_rewards):,.0f}')
axes[0, 0].plot(x, slope * x * 10000 + intercept * 10000, 'r--', alpha=0.5,
                label=f'Trend: {slope*10000:+.0f}/ep')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Raw Reward')
axes[0, 0].set_title('Raw Reward Curve (300 Episodes)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Service rate
axes[0, 1].plot(svc, alpha=0.6, color='green', marker='o', markersize=2)
axes[0, 1].axhline(y=np.mean(svc), color='r', linestyle='--', alpha=0.5,
                   label=f'Mean: {np.mean(svc):.4f}')
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
axes[1, 1].plot(actor_losses, alpha=0.6, color='red')
axes[1, 1].set_xlabel('Episode')
axes[1, 1].set_ylabel('Actor Loss')
axes[1, 1].set_title('Actor Loss Curve')
axes[1, 1].grid(True, alpha=0.3)

# Plot 5: Reward distribution by phase
phase_means = []
phase_labels = []
for i in range(0, len(rewards), 100):
    end = min(i + 100, len(rewards))
    phase_means.append(np.mean(rewards[i:end]))
    phase_labels.append(f'{i+1}-{end}')

axes[2, 0].bar(range(len(phase_means)), phase_means, color='steelblue', alpha=0.7)
axes[2, 0].set_xlabel('Episode Range')
axes[2, 0].set_ylabel('Average Reward (Scaled)')
axes[2, 0].set_title('Average Reward by Phase')
axes[2, 0].set_xticks(range(len(phase_means)))
axes[2, 0].set_xticklabels(phase_labels)
axes[2, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(phase_means):
    axes[2, 0].text(i, v, f'{v:.0f}', ha='center', va='bottom')

# Plot 6: Service rate distribution by phase
phase_svc_means = []
for i in range(0, len(svc), 100):
    end = min(i + 100, len(svc))
    phase_svc_means.append(np.mean(svc[i:end]) * 100)

axes[2, 1].bar(range(len(phase_svc_means)), phase_svc_means, color='green', alpha=0.7)
axes[2, 1].set_xlabel('Episode Range')
axes[2, 1].set_ylabel('Service Rate (%)')
axes[2, 1].set_title('Service Rate by Phase')
axes[2, 1].set_xticks(range(len(phase_svc_means)))
axes[2, 1].set_xticklabels(phase_labels)
axes[2, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(phase_svc_means):
    axes[2, 1].text(i, v, f'{v:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('shaping_mode_300ep_analysis.png', dpi=150)
print(f"\nVisualization saved: shaping_mode_300ep_analysis.png")
