# -*- coding: utf-8 -*-
import json
import re

# Read RL action file
with open('action/RL-DDPG_actions.txt', 'r', encoding='utf-8') as f:
    rl_content = f.read()

# Read optimal solution action file
with open('action/global_optimal_solution_actions.txt', 'r', encoding='utf-8') as f:
    opt_content = f.read()

# Extract RL total reward
rl_reward_match = re.search(r'总奖励:\s*([\d.]+)', rl_content)
rl_total_reward = float(rl_reward_match.group(1)) if rl_reward_match else 0

# Extract optimal total reward
opt_reward_match = re.search(r'总奖励:\s*([\d.]+)', opt_content)
opt_total_reward = float(opt_reward_match.group(1)) if opt_reward_match else 0

# Count RL charging actions per timestep
rl_charging_pattern = re.findall(r'【充电动作】\s*\n((?:  区域\d+: \d+辆\n)*)', rl_content)
rl_charging_per_step = []
for match in rl_charging_pattern:
    if match.strip():
        counts = re.findall(r'区域\d+: (\d+)辆', match)
        step_total = sum(int(c) for c in counts)
        rl_charging_per_step.append(step_total)
    else:
        rl_charging_per_step.append(0)

# Count optimal charging actions per timestep
opt_charging_pattern = re.findall(r'【充电动作】\s*\n((?:  区域\d+: \d+辆\n)*)', opt_content)
opt_charging_per_step = []
for match in opt_charging_pattern:
    if match.strip():
        counts = re.findall(r'区域\d+: (\d+)辆', match)
        step_total = sum(int(c) for c in counts)
        opt_charging_per_step.append(step_total)
    else:
        opt_charging_per_step.append(0)

print("="*80)
print("Reward Comparison Analysis")
print("="*80)
print(f"\nRL Total Reward: {rl_total_reward:,.2f}")
print(f"Optimal Total Reward: {opt_total_reward:,.2f}")
print(f"Difference: {rl_total_reward - opt_total_reward:,.2f} ({(rl_total_reward/opt_total_reward - 1)*100:.2f}%)")

print(f"\nCharging Statistics:")
print(f"RL Total Charging: {sum(rl_charging_per_step)}")
print(f"Optimal Total Charging: {sum(opt_charging_per_step)}")

print(f"\nPer-Step Charging Comparison:")
print(f"{'Step':<8} {'RL':<10} {'Optimal':<10} {'Diff':<10}")
print("-"*40)
for i in range(min(len(rl_charging_per_step), len(opt_charging_per_step))):
    diff = rl_charging_per_step[i] - opt_charging_per_step[i]
    print(f"{i:<8} {rl_charging_per_step[i]:<10} {opt_charging_per_step[i]:<10} {diff:+<10}")

# Estimate reward composition
rl_charging_reward = sum(rl_charging_per_step) * 18000
opt_charging_reward = sum(opt_charging_per_step) * 18000

print(f"\nReward Composition Estimate:")
print(f"RL Charging Reward (18000/vehicle): {rl_charging_reward:,.2f}")
print(f"RL Service+SOC Reward: {rl_total_reward - rl_charging_reward:,.2f}")
print(f"\nOptimal Charging Reward (18000/vehicle): {opt_charging_reward:,.2f}")
print(f"Optimal Service+SOC Reward: {opt_total_reward - opt_charging_reward:,.2f}")

print(f"\nKey Findings:")
if rl_total_reward > opt_total_reward:
    print(f"RL exceeds Optimal by {rl_total_reward - opt_total_reward:,.2f}")
    print(f"  - Charging reward diff: {rl_charging_reward - opt_charging_reward:,.2f}")
    print(f"  - Service+SOC diff: {(rl_total_reward - rl_charging_reward) - (opt_total_reward - opt_charging_reward):,.2f}")
