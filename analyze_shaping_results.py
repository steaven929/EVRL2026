import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

output_dir = './outputs/marl_20260221-121308'

print("="*80)
print("SHAPING MODE TRAINING RESULTS (100 Episodes)")
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
print(f"Scaled Rewards (stored, after /10000):")
print(f"  Mean: {np.mean(rewards):,.2f}")
print(f"  Std: {np.std(rewards):,.2f}")
print(f"  Min: {np.min(rewards):,.2f}")
print(f"  Max: {np.max(rewards):,.2f}")
print(f"  First 10 avg: {np.mean(rewards[:10]):,.2f}")
print(f"  Last 10 avg: {np.mean(rewards[-10:]):,.2f}")

print(f"\nRaw Rewards (before scaling):")
print(f"  Mean: {np.mean(raw_rewards):,.2f}")
print(f"  Std: {np.std(raw_rewards):,.2f}")
print(f"  Min: {np.min(raw_rewards):,.2f}")
print(f"  Max: {np.max(raw_rewards):,.2f}")
print(f"  First 10 avg: {np.mean(raw_rewards[:10]):,.2f}")
print(f"  Last 10 avg: {np.mean(raw_rewards[-10:]):,.2f}")

# Check if rewards are positive now
if np.mean(raw_rewards) > 0:
    print(f"\n  => SUCCESS! Rewards are now POSITIVE!")
else:
    print(f"\n  => WARNING: Rewards are still negative")

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

# Loss Analysis
print(f"\n[LOSS ANALYSIS]")
print(f"Critic Loss:")
print(f"  Mean: {np.mean(critic_losses):,.2e}")
print(f"  First 10: {np.mean(critic_losses[:10]):,.2e}")
print(f"  Last 10: {np.mean(critic_losses[-10:]):,.2e}")
print(f"  Min: {np.min(critic_losses):,.2e}")
print(f"  Max: {np.max(critic_losses):,.2e}")

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
if len(svc_nonzero) > 0:
    print(f"  Avg (service episodes only): {np.mean(svc_nonzero):.4f} ({np.mean(svc_nonzero)*100:.2f}%)")

# Comparison
print(f"\n[COMPARISON: BASE vs SHAPING MODE]")
print(f"\nBase Mode (previous 100 episodes):")
print(f"  Avg Raw Reward: -254")
print(f"  Avg Scaled Reward: -0.03")
print(f"  Service Rate: 2.34%")
print(f"  Critic Loss: 4.02e-06")

print(f"\nShaping Mode (current 100 episodes):")
print(f"  Avg Raw Reward: {np.mean(raw_rewards):,.2f}")
print(f"  Avg Scaled Reward: {np.mean(rewards):,.2f}")
print(f"  Service Rate: {np.mean(svc)*100:.2f}%")
print(f"  Critic Loss: {np.mean(critic_losses):,.2e}")

# Calculate improvement
raw_reward_change = np.mean(raw_rewards) - (-254)
print(f"\nImprovement:")
print(f"  Raw Reward: {raw_reward_change:+,.2f} ({raw_reward_change/abs(-254)*100:+.1f}%)")
if np.mean(raw_rewards) > 0:
    print(f"  => Reward changed from NEGATIVE to POSITIVE!")

# Diagnosis
print(f"\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if np.mean(raw_rewards) > 0:
    print("\n[EXCELLENT] Shaping mode successfully fixed negative rewards!")
    print("  - Charging bonuses are now included")
    print("  - Agents can learn from positive reinforcement")
else:
    print("\n[PROBLEM] Rewards are still negative despite shaping mode")
    print("  - May need to adjust reward weights")
    print("  - Or reduce penalties further")

if p_value < 0.05 and slope > 0:
    print("[GOOD] Reward is increasing significantly")
elif abs(slope * len(rewards)) < np.std(rewards):
    print("[NEUTRAL] Reward is relatively stable")
else:
    print("[CONCERN] Reward shows variation but no clear trend")

if np.mean(critic_losses) < 1e-3:
    print("[GOOD] Critic Loss is very low (learning well)")
elif np.mean(critic_losses) < 1e-1:
    print("[GOOD] Critic Loss is in reasonable range")
else:
    print("[CONCERN] Critic Loss is high")

if critic_change < -20:
    print("[GOOD] Critic Loss is decreasing (active learning)")
elif abs(critic_change) < 10:
    print("[NEUTRAL] Critic Loss is stable")

if np.mean(svc) > 0.05:
    print(f"[GOOD] Service rate improved to {np.mean(svc)*100:.2f}%")
elif np.mean(svc) > 0.02:
    print(f"[NEUTRAL] Service rate is {np.mean(svc)*100:.2f}% (moderate)")
else:
    print(f"[CONCERN] Service rate is still low ({np.mean(svc)*100:.2f}%)")

print("\n" + "="*80)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Raw rewards
window = 10
if len(raw_rewards) >= window:
    ma = np.convolve(raw_rewards, np.ones(window)/window, mode='valid')
    axes[0, 0].plot(raw_rewards, alpha=0.3, label='Raw Reward', color='blue')
    axes[0, 0].plot(range(window-1, len(raw_rewards)), ma, linewidth=2,
                    label=f'{window}-Episode MA', color='darkblue')
else:
    axes[0, 0].plot(raw_rewards, alpha=0.6, label='Raw Reward', color='blue')
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Line')
axes[0, 0].axhline(y=np.mean(raw_rewards), color='green', linestyle='--', alpha=0.5,
                   label=f'Mean: {np.mean(raw_rewards):,.0f}')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Raw Reward')
axes[0, 0].set_title('Raw Reward Curve (Before Scaling)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Scaled rewards
if len(rewards) >= window:
    ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
    axes[0, 1].plot(rewards, alpha=0.3, label='Scaled Reward', color='purple')
    axes[0, 1].plot(range(window-1, len(rewards)), ma, linewidth=2,
                    label=f'{window}-Episode MA', color='darkviolet')
else:
    axes[0, 1].plot(rewards, alpha=0.6, label='Scaled Reward', color='purple')
axes[0, 1].axhline(y=np.mean(rewards), color='r', linestyle='--', alpha=0.5,
                   label=f'Mean: {np.mean(rewards):.2f}')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Scaled Reward')
axes[0, 1].set_title('Scaled Reward Curve (After /10000)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Service rate
axes[0, 2].plot(svc, alpha=0.6, color='green', marker='o', markersize=3)
axes[0, 2].axhline(y=np.mean(svc), color='r', linestyle='--', alpha=0.5,
                   label=f'Mean: {np.mean(svc):.4f}')
axes[0, 2].set_xlabel('Episode')
axes[0, 2].set_ylabel('Service Rate')
axes[0, 2].set_title('Service Rate Over Time')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Critic loss
axes[1, 0].plot(critic_losses, alpha=0.6, color='orange')
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Critic Loss')
axes[1, 0].set_title('Critic Loss Curve')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Actor loss
axes[1, 1].plot(actor_losses, alpha=0.6, color='red')
axes[1, 1].set_xlabel('Episode')
axes[1, 1].set_ylabel('Actor Loss')
axes[1, 1].set_title('Actor Loss Curve')
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Reward distribution
axes[1, 2].hist(raw_rewards, bins=20, alpha=0.7, color='blue', edgecolor='black')
axes[1, 2].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
axes[1, 2].axvline(x=np.mean(raw_rewards), color='green', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(raw_rewards):,.0f}')
axes[1, 2].set_xlabel('Raw Reward')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('Raw Reward Distribution')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('shaping_mode_analysis_100ep.png', dpi=150)
print(f"\nVisualization saved: shaping_mode_analysis_100ep.png")
