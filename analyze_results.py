import numpy as np
import os

# Find the most recent training directory
output_dir = './outputs/marl_20260220-134655'

print('=' * 80)
print(f'Training Results Analysis: {output_dir}')
print('=' * 80)

# Load data
rewards = np.load(os.path.join(output_dir, 'rewards.npy'))
svc = np.load(os.path.join(output_dir, 'svc_history.npy'))
rld = np.load(os.path.join(output_dir, 'rld_history.npy'))
srd = np.load(os.path.join(output_dir, 'srd_history.npy'))

print(f'\nTotal Episodes: {len(rewards)}')
print(f'\n[Reward Statistics]')
print(f'  Average Reward: {np.mean(rewards):,.2f}')
print(f'  Max Reward: {np.max(rewards):,.2f}')
print(f'  Min Reward: {np.min(rewards):,.2f}')
print(f'  Final Reward: {rewards[-1]:,.2f}')

print(f'\n[Service Rate (SVC) Statistics]')
svc_nonzero = svc[svc > 0]
print(f'  Average SVC: {np.mean(svc):.4f} ({np.mean(svc)*100:.2f}%)')
print(f'  Max SVC: {np.max(svc):.4f} ({np.max(svc)*100:.2f}%)')
print(f'  Episodes with service: {len(svc_nonzero)}/{len(svc)} ({len(svc_nonzero)/len(svc)*100:.1f}%)')
if len(svc_nonzero) > 0:
    print(f'  Avg SVC (service episodes only): {np.mean(svc_nonzero):.4f} ({np.mean(svc_nonzero)*100:.2f}%)')

print(f'\n[Recharge Distance (RLD) Statistics]')
print(f'  Average RLD: {np.mean(rld):.2f}')
print(f'  Max RLD: {np.max(rld):.2f}')

print(f'\n[Service Distance (SRD) Statistics]')
print(f'  Average SRD: {np.mean(srd):.2f}')
print(f'  Max SRD: {np.max(srd):.2f}')

# Analyze trends
print(f'\n[Learning Trend Analysis]')
first_30 = svc[:30]
last_30 = svc[-30:]
print(f'  First 30 episodes avg SVC: {np.mean(first_30):.4f} ({np.mean(first_30)*100:.2f}%)')
print(f'  Last 30 episodes avg SVC: {np.mean(last_30):.4f} ({np.mean(last_30)*100:.2f}%)')
if np.mean(first_30) > 0:
    improvement = (np.mean(last_30) - np.mean(first_30)) / np.mean(first_30) * 100
    print(f'  Improvement: {improvement:+.1f}%')
else:
    print(f'  Improvement: N/A (first 30 episodes had 0% service rate)')

# Show episodes with service
print(f'\n[Episodes with Service]')
service_episodes = np.where(svc > 0)[0]
if len(service_episodes) > 0:
    print(f'  Episode indices: {service_episodes.tolist()}')
    for ep in service_episodes[:15]:
        print(f'    Episode {ep+1}: SVC={svc[ep]:.4f} ({svc[ep]*100:.2f}%), Reward={rewards[ep]:,.2f}')
    if len(service_episodes) > 15:
        print(f'  ... and {len(service_episodes)-15} more episodes')
else:
    print('  None')

print('\n' + '=' * 80)
