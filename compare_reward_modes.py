print('='*80)
print('Reward Function Comparison Analysis')
print('='*80)

# Typical scenario
charging_count = 10  # 10 vehicles charging
avg_soc = 1500  # Average SOC
service_revenue = 2600  # 26 steps, 100 per step
soc_penalty = 85500  # Previous estimate
service_rate_penalty = 3840  # 2% service rate
fixed_penalty = 0

print('\nCurrent (base mode):')
base_reward = (service_revenue - soc_penalty) / 10 - fixed_penalty - service_rate_penalty
print(f'  reward = (service_revenue - soc_penalty) / 10 - service_rate_penalty')
print(f'  reward = ({service_revenue} - {soc_penalty}) / 10 - {service_rate_penalty}')
print(f'  reward = {base_reward:.2f}')

print('\nSmall-scale (shaping mode):')
charging_bonus = 15000 * charging_count
charging_maintenance_bonus = 3000 * charging_count
soc_maintain_bonus = (avg_soc - 1200) * 2.0 if avg_soc >= 1200 else 0

shaping_reward = base_reward + charging_bonus + charging_maintenance_bonus + soc_maintain_bonus
print(f'  reward = base_reward + charging_bonus + charging_maintenance_bonus + soc_maintain_bonus')
print(f'  charging_bonus = 15000 x {charging_count} = {charging_bonus:,}')
print(f'  charging_maintenance_bonus = 3000 x {charging_count} = {charging_maintenance_bonus:,}')
print(f'  soc_maintain_bonus = ({avg_soc} - 1200) x 2.0 = {soc_maintain_bonus:,.0f}')
print(f'  reward = {base_reward:.2f} + {charging_bonus:,} + {charging_maintenance_bonus:,} + {soc_maintain_bonus:.0f}')
print(f'  reward = {shaping_reward:,.2f}')

print('\nDifference Analysis:')
print(f'  Missing rewards = {shaping_reward - base_reward:,.2f}')
print(f'  Reward change: {base_reward:.2f} -> {shaping_reward:,.2f}')
print(f'  Improvement: {(shaping_reward - base_reward) / abs(base_reward) * 100:.0f}%')

print('\nContribution Breakdown:')
total_bonus = charging_bonus + charging_maintenance_bonus + soc_maintain_bonus
print(f'  charging_bonus: {charging_bonus/total_bonus*100:.1f}%')
print(f'  charging_maintenance_bonus: {charging_maintenance_bonus/total_bonus*100:.1f}%')
print(f'  soc_maintain_bonus: {soc_maintain_bonus/total_bonus*100:.1f}%')

print('\n' + '='*80)
print('CONCLUSION: Must enable shaping mode, otherwise reward is always negative!')
print('='*80)

# Check different charging scenarios
print('\nScenario Analysis:')
for n_charging in [5, 10, 15, 20]:
    cb = 15000 * n_charging
    cmb = 3000 * n_charging
    smb = soc_maintain_bonus
    total = base_reward + cb + cmb + smb
    print(f'  {n_charging} vehicles charging: reward = {total:,.2f}')
