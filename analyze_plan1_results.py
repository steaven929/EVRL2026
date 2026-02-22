import numpy as np
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

dir_path = './outputs/marl_20260222-011500'

# Load all data
rewards = np.load(os.path.join(dir_path, 'rewards.npy'))
critic_losses = np.load(os.path.join(dir_path, 'critic_losses.npy'))
actor_losses = np.load(os.path.join(dir_path, 'actor_losses.npy'))
svc_history = np.load(os.path.join(dir_path, 'svc_history.npy'))
rld_history = np.load(os.path.join(dir_path, 'rld_history.npy'))

print('=== Plan 1 Results: 1000 Episode Training Analysis ===')
print('Hyperparameters: reward_scale=1000, actor_lr=3e-6, critic_lr=1e-5, tau=5e-5')
print()
print(f'Total episodes: {len(rewards)}')
print(f'Total updates: {len(critic_losses)}')
print()

# Reward analysis
print('=== Reward Analysis ===')
print(f'Mean reward: {np.mean(rewards):.2f}')
print(f'Std reward: {np.std(rewards):.2f}')
print(f'Min reward: {np.min(rewards):.2f}')
print(f'Max reward: {np.max(rewards):.2f}')
print()
print('Reward trend (every 100 episodes):')
for i in range(0, len(rewards), 100):
    segment = rewards[i:min(i+100, len(rewards))]
    print(f'  Ep {i:4d}-{min(i+99, len(rewards)-1):4d}: Mean={np.mean(segment):10.2f}, Std={np.std(segment):8.2f}, Min={np.min(segment):8.2f}, Max={np.max(segment):8.2f}')
print()

# Loss analysis - detailed
print('=== Loss Analysis (Detailed) ===')
print(f'Critic Loss:')
print(f'  Mean: {np.mean(critic_losses):.2f}')
print(f'  Std: {np.std(critic_losses):.2f}')
print(f'  Min: {np.min(critic_losses):.2f}')
print(f'  Max: {np.max(critic_losses):.2f}')
print(f'  Median: {np.median(critic_losses):.2f}')
print()
print(f'Actor Loss:')
print(f'  Mean: {np.mean(actor_losses):.4f}')
print(f'  Std: {np.std(actor_losses):.4f}')
print(f'  Min: {np.min(actor_losses):.4f}')
print(f'  Max: {np.max(actor_losses):.4f}')
print(f'  Median: {np.median(actor_losses):.4f}')
print()

# Loss trend by segments (10 segments for 1000 episodes)
print('=== Loss Trend (by 10% segments) ===')
n = len(critic_losses)
for i in range(10):
    start = int(n * i / 10)
    end = int(n * (i + 1) / 10)
    c_seg = critic_losses[start:end]
    a_seg = actor_losses[start:end]
    print(f'Segment {i+1:2d} (updates {start:5d}-{end:5d}):')
    print(f'  Critic: Mean={np.mean(c_seg):10.2f}, Std={np.std(c_seg):8.2f}, Max={np.max(c_seg):10.2f}')
    print(f'  Actor:  Mean={np.mean(a_seg):10.4f}, Std={np.std(a_seg):8.4f}, Min={np.min(a_seg):10.4f}')
print()

# Check for explosion
print('=== Loss Explosion Check ===')
c_first_100 = np.mean(critic_losses[:100])
c_last_100 = np.mean(critic_losses[-100:])
a_first_100 = np.mean(actor_losses[:100])
a_last_100 = np.mean(actor_losses[-100:])

print(f'Critic Loss: First 100 updates = {c_first_100:.2f}, Last 100 updates = {c_last_100:.2f}')
print(f'  Change: {(c_last_100/c_first_100-1)*100:+.1f}%')
print(f'Actor Loss: First 100 updates = {a_first_100:.4f}, Last 100 updates = {a_last_100:.4f}')
print(f'  Change: {(a_last_100/a_first_100-1)*100:+.1f}%')
print()

# Compare with previous training (marl_20260221-172845)
print('=== Comparison with Previous Training (600ep, old hyperparameters) ===')
print('Previous (reward_scale=10000, actor_lr=1.5e-5):')
print('  Actor Loss change: +51,728% (EXPLOSION)')
print('  Final Actor Loss: -683.67')
print('  Reward: 1565 -> 3225 (+106%)')
print()
print('Current (reward_scale=1000, actor_lr=3e-6):')
print(f'  Actor Loss change: {(a_last_100/a_first_100-1)*100:+.1f}%')
print(f'  Final Actor Loss: {a_last_100:.4f}')
print(f'  Reward: {np.mean(rewards[:100]):.2f} -> {np.mean(rewards[-100:]):.2f} ({(np.mean(rewards[-100:])/np.mean(rewards[:100])-1)*100:+.1f}%)')
print()

# Check for NaN or Inf
print('=== Data Quality Check ===')
print(f'Critic Loss - NaN count: {np.isnan(critic_losses).sum()}, Inf count: {np.isinf(critic_losses).sum()}')
print(f'Actor Loss - NaN count: {np.isnan(actor_losses).sum()}, Inf count: {np.isinf(actor_losses).sum()}')
print(f'Rewards - NaN count: {np.isnan(rewards).sum()}, Inf count: {np.isinf(rewards).sum()}')
print()

# Convergence analysis
print('=== Convergence Analysis ===')
# Check if reward is converging (compare variance in first vs last 200 episodes)
first_200 = rewards[:200]
last_200 = rewards[-200:]
print(f'Reward variance:')
print(f'  First 200 episodes: {np.var(first_200):.2f}')
print(f'  Last 200 episodes: {np.var(last_200):.2f}')
print(f'  Change: {(np.var(last_200)/np.var(first_200)-1)*100:+.1f}%')
print()

# Check if loss is stabilizing
first_1000 = critic_losses[:1000]
last_1000 = critic_losses[-1000:]
print(f'Critic Loss variance:')
print(f'  First 1000 updates: {np.var(first_1000):.2f}')
print(f'  Last 1000 updates: {np.var(last_1000):.2f}')
print(f'  Change: {(np.var(last_1000)/np.var(first_1000)-1)*100:+.1f}%')
