import numpy as np
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

dir_path = './outputs/marl_20260221-172845'

# Load all data
rewards = np.load(os.path.join(dir_path, 'rewards.npy'))
critic_losses = np.load(os.path.join(dir_path, 'critic_losses.npy'))
actor_losses = np.load(os.path.join(dir_path, 'actor_losses.npy'))
svc_history = np.load(os.path.join(dir_path, 'svc_history.npy'))
rld_history = np.load(os.path.join(dir_path, 'rld_history.npy'))

print('=== 600 Episode Training Analysis (marl_20260221-172845) ===')
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
    print(f'  Ep {i:3d}-{min(i+99, len(rewards)-1):3d}: Mean={np.mean(segment):10.2f}, Std={np.std(segment):8.2f}, Min={np.min(segment):8.2f}, Max={np.max(segment):8.2f}')
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

# Loss trend by segments
print('=== Loss Trend (by 20% segments) ===')
n = len(critic_losses)
for i in range(5):
    start = int(n * i / 5)
    end = int(n * (i + 1) / 5)
    c_seg = critic_losses[start:end]
    a_seg = actor_losses[start:end]
    print(f'Segment {i+1} (updates {start:5d}-{end:5d}):')
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

# Check for NaN or Inf
print('=== Data Quality Check ===')
print(f'Critic Loss - NaN count: {np.isnan(critic_losses).sum()}, Inf count: {np.isinf(critic_losses).sum()}')
print(f'Actor Loss - NaN count: {np.isnan(actor_losses).sum()}, Inf count: {np.isinf(actor_losses).sum()}')
print(f'Rewards - NaN count: {np.isnan(rewards).sum()}, Inf count: {np.isinf(rewards).sum()}')
