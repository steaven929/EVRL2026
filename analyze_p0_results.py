import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

output_dir = './outputs/marl_20260221-104922'

print("="*80)
print("P0 FIX TRAINING RESULTS ANALYSIS (100 Episodes)")
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
print(f"Scaled Rewards (stored):")
print(f"  Mean: {np.mean(rewards):,.2f}")
print(f"  Std: {np.std(rewards):,.2f}")
print(f"  Min: {np.min(rewards):,.2f}")
print(f"  Max: {np.max(rewards):,.2f}")
print(f"  First 10 avg: {np.mean(rewards[:10]):,.2f}")
print(f"  Last 10 avg: {np.mean(rewards[-10:]):,.2f}")

print(f"\nRaw Rewards (before scaling):")
print(f"  Mean: {np.mean(raw_rewards):,.2f}")
print(f"  First 10 avg: {np.mean(raw_rewards[:10]):,.2f}")
print(f"  Last 10 avg: {np.mean(raw_rewards[-10:]):,.2f}")

# Trend analysis
x = np.arange(len(rewards))
slope, intercept, r_value, p_value, std_err = stats.linregress(x, rewards)
print(f"\nReward Trend:")
print(f"  Slope: {slope:+.2f} per episode")
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

# Loss Analysis
print(f"\n[LOSS ANALYSIS]")
print(f"Critic Loss:")
print(f"  Mean: {np.mean(critic_losses):,.2e}")
print(f"  First 10: {np.mean(critic_losses[:10]):,.2e}")
print(f"  Last 10: {np.mean(critic_losses[-10:]):,.2e}")
print(f"  Min: {np.min(critic_losses):,.2e}")
print(f"  Max: {np.max(critic_losses):,.2e}")

# Check if decreasing
first_third = critic_losses[:len(critic_losses)//3]
last_third = critic_losses[-len(critic_losses)//3:]
critic_change = (np.mean(last_third) - np.mean(first_third)) / np.mean(first_third) * 100
print(f"  First 1/3 avg: {np.mean(first_third):,.2e}")
print(f"  Last 1/3 avg: {np.mean(last_third):,.2e}")
print(f"  Change: {critic_change:+.1f}%")

print(f"\nActor Loss:")
print(f"  Mean: {np.mean(actor_losses):,.6f}")
print(f"  First 10: {np.mean(actor_losses[:10]):,.6f}")
print(f"  Last 10: {np.mean(actor_losses[-10:]):,.6f}")
print(f"  Min: {np.min(actor_losses):,.6f}")
print(f"  Max: {np.max(actor_losses):,.6f}")

# Service Rate
print(f"\n[SERVICE RATE]")
svc_nonzero = svc[svc > 0]
print(f"  Average: {np.mean(svc):.4f} ({np.mean(svc)*100:.2f}%)")
print(f"  Max: {np.max(svc):.4f} ({np.max(svc)*100:.2f}%)")
print(f"  Episodes with service: {len(svc_nonzero)}/{len(svc)} ({len(svc_nonzero)/len(svc)*100:.1f}%)")

# Comparison with previous training (600 episodes before P0 fix)
print(f"\n[COMPARISON WITH PREVIOUS TRAINING]")
print(f"Before P0 Fix (600 episodes):")
print(f"  Avg Reward (scaled): 1,015,075")
print(f"  Critic Loss: 1.83e9")
print(f"  Actor Loss: -28,768")
print(f"  Service Rate: 1.75%")
print(f"\nAfter P0 Fix (100 episodes):")
print(f"  Avg Reward (scaled): {np.mean(rewards):,.2f}")
print(f"  Critic Loss: {np.mean(critic_losses):,.2e}")
print(f"  Actor Loss: {np.mean(actor_losses):,.2f}")
print(f"  Service Rate: {np.mean(svc)*100:.2f}%")

# Calculate improvements
if np.mean(rewards) > 1015075:
    reward_improvement = (np.mean(rewards) - 1015075) / 1015075 * 100
    print(f"\nReward improved by: {reward_improvement:+.2f}%")
else:
    reward_decline = (1015075 - np.mean(rewards)) / 1015075 * 100
    print(f"\nReward declined by: {reward_decline:.2f}%")

critic_improvement = (1.83e9 - np.mean(critic_losses)) / 1.83e9 * 100
print(f"Critic Loss improved by: {critic_improvement:.2f}%")

actor_improvement = (abs(-28768) - abs(np.mean(actor_losses))) / abs(-28768) * 100
print(f"Actor Loss improved by: {actor_improvement:.2f}%")

# Diagnosis
print(f"\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if p_value < 0.05 and slope > 0:
    print("\n[GOOD] Reward is increasing significantly!")
elif abs(slope * len(rewards)) < np.std(rewards):
    print("\n[PROBLEM] Reward is still flat (no learning)")
else:
    print("\n[NEUTRAL] Reward shows some variation but no clear trend")

if np.mean(critic_losses) < 1e6:
    print("[GOOD] Critic Loss is in reasonable range (<1e6)")
elif np.mean(critic_losses) < 1e8:
    print("[NEUTRAL] Critic Loss is moderate (1e6-1e8)")
else:
    print("[PROBLEM] Critic Loss is still very high (>1e8)")

if critic_change < -20:
    print("[GOOD] Critic Loss is decreasing (learning)")
elif abs(critic_change) < 10:
    print("[PROBLEM] Critic Loss is stable (not learning)")
else:
    print("[NEUTRAL] Critic Loss shows some change")

if abs(np.mean(actor_losses)) < 100:
    print("[GOOD] Actor Loss is in reasonable range")
elif abs(np.mean(actor_losses)) < 1000:
    print("[NEUTRAL] Actor Loss is moderate")
else:
    print("[PROBLEM] Actor Loss is still abnormally large")

print("\n" + "="*80)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Reward curve
window = 10
if len(rewards) >= window:
    ma_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    axes[0, 0].plot(rewards, alpha=0.3, label='Episode Reward', color='blue')
    axes[0, 0].plot(range(window-1, len(rewards)), ma_rewards, linewidth=2,
                    label=f'{window}-Episode MA', color='darkblue')
else:
    axes[0, 0].plot(rewards, alpha=0.6, label='Episode Reward', color='blue')
axes[0, 0].axhline(y=np.mean(rewards), color='r', linestyle='--', alpha=0.5,
                   label=f'Mean: {np.mean(rewards):,.0f}')
axes[0, 0].plot(x, slope * x + intercept, 'g--', alpha=0.5,
                label=f'Trend: {slope:+.1f}/ep')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Reward (Scaled)')
axes[0, 0].set_title(f'Reward Curve ({len(rewards)} Episodes)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Service rate
axes[0, 1].plot(svc, alpha=0.6, color='green', marker='o', markersize=3)
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
axes[1, 1].plot(actor_losses, alpha=0.6, color='purple')
axes[1, 1].set_xlabel('Episode')
axes[1, 1].set_ylabel('Actor Loss')
axes[1, 1].set_title('Actor Loss Curve')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('p0_fix_analysis_100ep.png', dpi=150)
print(f"\nVisualization saved: p0_fix_analysis_100ep.png")
