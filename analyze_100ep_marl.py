import numpy as np
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

dir_path = './outputs/marl_20260221-234106'

# Load all data
rewards = np.load(os.path.join(dir_path, 'rewards.npy'))
critic_losses = np.load(os.path.join(dir_path, 'critic_losses.npy'))
actor_losses = np.load(os.path.join(dir_path, 'actor_losses.npy'))
svc_history = np.load(os.path.join(dir_path, 'svc_history.npy'))
rld_history = np.load(os.path.join(dir_path, 'rld_history.npy'))

print('=== 100 Episode Training Analysis ===')
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
print('Reward trend (every 20 episodes):')
for i in range(0, len(rewards), 20):
    segment = rewards[i:min(i+20, len(rewards))]
    print(f'  Ep {i:3d}-{min(i+19, len(rewards)-1):3d}: {np.mean(segment):10.2f} (std: {np.std(segment):8.2f})')
print()

# Loss analysis
print('=== Loss Analysis ===')
print(f'Critic Loss - Mean: {np.mean(critic_losses):.2f}, Std: {np.std(critic_losses):.2f}')
print(f'Actor Loss - Mean: {np.mean(actor_losses):.4f}, Std: {np.std(actor_losses):.4f}')
print()

# Loss trend
n = len(critic_losses)
first_20 = int(n * 0.2)
last_20 = int(n * 0.8)

print('Loss Trend (First 20% vs Last 20%):')
c_first = np.mean(critic_losses[:first_20])
c_last = np.mean(critic_losses[last_20:])
a_first = np.mean(actor_losses[:first_20])
a_last = np.mean(actor_losses[last_20:])

print(f'  Critic: {c_first:.2f} -> {c_last:.2f} ({(c_last/c_first-1)*100:+.1f}%)')
print(f'  Actor: {a_first:.4f} -> {a_last:.4f} ({(a_last/a_first-1)*100:+.1f}%)')
print()

# Service rate
print('=== Service Rate Analysis ===')
print(f'Mean service rate: {np.mean(svc_history):.2f}%')
print(f'Max service rate: {np.max(svc_history):.2f}%')
print(f'Min service rate: {np.min(svc_history):.2f}%')
print()

# RLD
print('=== RLD Analysis ===')
print(f'Mean RLD: {np.mean(rld_history):.2f}')
print(f'Max RLD: {np.max(rld_history):.2f}')
print(f'Min RLD: {np.min(rld_history):.2f}')
