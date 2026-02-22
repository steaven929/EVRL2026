import numpy as np
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

dir_path = './outputs/marl_20260221-143622-'

# Load all data
rewards = np.load(os.path.join(dir_path, 'rewards.npy'))
raw_rewards = np.load(os.path.join(dir_path, 'raw_rewards.npy'))
critic_losses = np.load(os.path.join(dir_path, 'critic_losses.npy'))
actor_losses = np.load(os.path.join(dir_path, 'actor_losses.npy'))
svc_history = np.load(os.path.join(dir_path, 'svc_history.npy'))
rld_history = np.load(os.path.join(dir_path, 'rld_history.npy'))

print('=== Training Analysis: marl_20260221-143622- (300 Episodes) ===')
print()
print(f'Total episodes: {len(rewards)}')
print(f'Total updates: {len(critic_losses)}')
print()

# Reward analysis
print('=== Reward Analysis ===')
print(f'Scaled Rewards:')
print(f'  Mean: {np.mean(rewards):.2f}')
print(f'  Std: {np.std(rewards):.2f}')
print(f'  Range: {np.min(rewards):.2f} to {np.max(rewards):.2f}')
print()
print(f'Raw Rewards:')
print(f'  Mean: {np.mean(raw_rewards):,.2f}')
print(f'  Std: {np.std(raw_rewards):,.2f}')
print(f'  Range: {np.min(raw_rewards):,.2f} to {np.max(raw_rewards):,.2f}')
print()

print('Reward trend (every 100 episodes):')
for i in range(0, len(rewards), 100):
    segment_raw = raw_rewards[i:min(i+100, len(rewards))]
    segment_scaled = rewards[i:min(i+100, len(rewards))]
    print(f'  Ep {i:3d}-{min(i+99, len(rewards)-1):3d}:')
    print(f'    Raw: {np.mean(segment_raw):12,.2f} (std: {np.std(segment_raw):10,.2f})')
    print(f'    Scaled: {np.mean(segment_scaled):10.2f} (std: {np.std(segment_scaled):8.2f})')
print()

# Loss analysis
print('=== Loss Analysis ===')
print(f'Critic Loss:')
print(f'  Mean: {np.mean(critic_losses):.2f}')
print(f'  Std: {np.std(critic_losses):.2f}')
print(f'  Range: {np.min(critic_losses):.2f} to {np.max(critic_losses):.2f}')
print()
print(f'Actor Loss:')
print(f'  Mean: {np.mean(actor_losses):.4f}')
print(f'  Std: {np.std(actor_losses):.4f}')
print(f'  Range: {np.min(actor_losses):.4f} to {np.max(actor_losses):.4f}')
print()

# Loss trend by thirds
n = len(critic_losses)
third = n // 3
print('Loss Trend (by thirds):')
for i in range(3):
    start = i * third
    end = (i + 1) * third if i < 2 else n
    c_seg = critic_losses[start:end]
    a_seg = actor_losses[start:end]
    print(f'Third {i+1} (updates {start:5d}-{end:5d}):')
    print(f'  Critic: Mean={np.mean(c_seg):10.2f}, Std={np.std(c_seg):8.2f}, Max={np.max(c_seg):10.2f}')
    print(f'  Actor:  Mean={np.mean(a_seg):10.4f}, Std={np.std(a_seg):8.4f}, Min={np.min(a_seg):10.4f}')
print()

# Explosion check
c_first_100 = np.mean(critic_losses[:100])
c_last_100 = np.mean(critic_losses[-100:])
a_first_100 = np.mean(actor_losses[:100])
a_last_100 = np.mean(actor_losses[-100:])

print('=== Loss Explosion Check ===')
print(f'Critic Loss: {c_first_100:.2f} -> {c_last_100:.2f} ({(c_last_100/c_first_100-1)*100:+.1f}%)')
print(f'Actor Loss: {a_first_100:.4f} -> {a_last_100:.4f} ({(a_last_100/a_first_100-1)*100:+.1f}%)')
print()

# Convergence check
first_100 = raw_rewards[:100]
last_100 = raw_rewards[-100:]
print('=== Convergence Check ===')
print(f'Reward: {np.mean(first_100):,.2f} -> {np.mean(last_100):,.2f} ({(np.mean(last_100)/np.mean(first_100)-1)*100:+.1f}%)')
print(f'Reward variance: {np.var(first_100):,.2f} -> {np.var(last_100):,.2f} ({(np.var(last_100)/np.var(first_100)-1)*100:+.1f}%)')
print()

# Data quality
print('=== Data Quality ===')
print(f'Critic Loss - NaN: {np.isnan(critic_losses).sum()}, Inf: {np.isinf(critic_losses).sum()}')
print(f'Actor Loss - NaN: {np.isnan(actor_losses).sum()}, Inf: {np.isinf(actor_losses).sum()}')
print(f'Rewards - NaN: {np.isnan(rewards).sum()}, Inf: {np.isinf(rewards).sum()}')
