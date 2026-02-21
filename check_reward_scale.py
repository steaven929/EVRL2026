import numpy as np

# Check actual reward values
output_dir = './outputs/marl_20260220-165650'
rewards = np.load(output_dir + '/rewards.npy')

print('Scaled Rewards (stored):')
print(f'  Mean: {np.mean(rewards):,.2f}')
print(f'  Per step (~26 steps): {np.mean(rewards)/26:.2f}')
print()
print('If reward_scale=1000:')
print(f'  Raw reward mean: {np.mean(rewards) * 1000:,.2f}')
print(f'  Raw reward per step: {np.mean(rewards) * 1000 / 26:,.2f}')
print()
print('Theoretical Q-value estimation (gamma=0.95, 26 steps):')
discount_sum = sum([0.95**i for i in range(26)])
print(f'  Discount sum: {discount_sum:.2f}')
print(f'  Expected Q: {np.mean(rewards)/26 * discount_sum:.2f}')
print()
print('Actual Critic Loss: 1.83e9')
print(f'  Implied Q-value error: {np.sqrt(1.83e9):,.2f}')
print()
print('PROBLEM: Q-values are MASSIVELY overestimated!')
print(f'  Expected Q: ~{np.mean(rewards)/26 * discount_sum:.0f}')
print(f'  Actual Q error: ~{np.sqrt(1.83e9):,.0f}')
print(f'  Overestimation factor: ~{np.sqrt(1.83e9) / (np.mean(rewards)/26 * discount_sum):.0f}x')
