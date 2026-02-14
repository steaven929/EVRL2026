# -*- coding: utf-8 -*-
"""
Extract vehicle distribution per zone per timestep from RL actions
"""
import re
import json

# Read RL action file
with open('action/RL-DDPG_actions.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# Initialize vehicle tracking
# Assume initial distribution (you may need to adjust based on actual initial state)
ZONE_NUM = 4
CAR_NUM = 18
initial_distribution = [5, 4, 5, 4]  # Initial vehicles per zone (adjust if needed)

# Track vehicle distribution over time
vehicle_distribution = []
vehicle_distribution.append({
    'timestep': 0,
    'zones': initial_distribution.copy(),
    'total': sum(initial_distribution)
})

# Extract timestep information
timesteps = re.findall(
    r'时间步 (\d+).*?【服务订单】\s*\n(.*?)【调度动作】\s*\n(.*?)【充电动作】\s*\n(.*?)(?=【动作向量】|时间步|\Z)',
    content,
    re.DOTALL
)

current_distribution = initial_distribution.copy()

for match in timesteps:
    if len(match) >= 3:
        step_num = int(match[0])
        service = match[1]
        dispatch = match[2]
        # charging = match[3] if len(match) > 3 else ""
for match in timesteps:
    if len(match) >= 3:
        step_num = int(match[0])
        service = match[1]
        dispatch = match[2]
        # charging = match[3] if len(match) > 3 else ""

        # Parse service orders (vehicles leaving zones)
        service_lines = [l.strip() for l in service.split('\n') if '区域' in l and '辆' in l]
        for line in service_lines:
            match_service = re.search(r'区域(\d+)\s*→\s*区域(\d+):\s*(\d+)辆', line)
            if match_service:
                from_zone = int(match_service.group(1))
                to_zone = int(match_service.group(2))
                count = int(match_service.group(3))
                # Vehicle moves from from_zone to to_zone
                current_distribution[from_zone] -= count
                current_distribution[to_zone] += count

        # Parse dispatch actions (vehicles being repositioned)
        dispatch_lines = [l.strip() for l in dispatch.split('\n') if '区域' in l and '辆' in l]
        for line in dispatch_lines:
            match_dispatch = re.search(r'区域(\d+)\s*→\s*区域(\d+):\s*(\d+)辆', line)
            if match_dispatch:
                from_zone = int(match_dispatch.group(1))
                to_zone = int(match_dispatch.group(2))
                count = int(match_dispatch.group(3))
                # Vehicle moves from from_zone to to_zone
                current_distribution[from_zone] -= count
                current_distribution[to_zone] += count

        # Record distribution after this timestep
        vehicle_distribution.append({
            'timestep': step_num + 1,
            'zones': current_distribution.copy(),
            'total': sum(current_distribution)
        })

# Generate output
output_lines = []
output_lines.append("="*80)
output_lines.append("Vehicle Distribution per Zone per Timestep (RL-DDPG)")
output_lines.append("="*80)
output_lines.append("")

# Table header
output_lines.append(f"{'Timestep':<12} {'Zone 0':<10} {'Zone 1':<10} {'Zone 2':<10} {'Zone 3':<10} {'Total':<10}")
output_lines.append("-"*72)

# Table rows
for entry in vehicle_distribution:
    zones = entry['zones']
    output_lines.append(
        f"{entry['timestep']:<12} {zones[0]:<10} {zones[1]:<10} {zones[2]:<10} {zones[3]:<10} {entry['total']:<10}"
    )

output_lines.append("")
output_lines.append("="*80)
output_lines.append("Summary Statistics")
output_lines.append("="*80)
output_lines.append("")

# Calculate average distribution
avg_distribution = [0] * ZONE_NUM
for entry in vehicle_distribution[1:]:  # Skip initial state
    for i in range(ZONE_NUM):
        avg_distribution[i] += entry['zones'][i]

for i in range(ZONE_NUM):
    avg_distribution[i] /= len(vehicle_distribution) - 1

output_lines.append("Average vehicles per zone:")
for i in range(ZONE_NUM):
    output_lines.append(f"  Zone {i}: {avg_distribution[i]:.2f}")

output_lines.append("")
output_lines.append("Note: This analysis is based on service and dispatch actions.")
output_lines.append("Actual distribution may vary due to vehicle movements and charging.")

# Write to file
output_file = 'action/vehicle_distribution_per_timestep.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print(f"Vehicle distribution saved to: {output_file}")
print("")
print("Preview:")
print('\n'.join(output_lines[:20]))
