# -*- coding: utf-8 -*-
"""
分析RL和最优解的奖励差异
"""

import json
import re

# 读取RL action文件
with open('action/RL-DDPG_actions.txt', 'r', encoding='utf-8') as f:
    rl_content = f.read()

# 读取最优解action文件
with open('action/global_optimal_solution_actions.txt', 'r', encoding='utf-8') as f:
    opt_content = f.read()

# 提取RL总奖励
rl_reward_match = re.search(r'总奖励:\s*([\d.]+)', rl_content)
rl_total_reward = float(rl_reward_match.group(1)) if rl_reward_match else 0

# 提取最优解总奖励
opt_reward_match = re.search(r'总奖励:\s*([\d.]+)', opt_content)
opt_total_reward = float(opt_reward_match.group(1)) if opt_reward_match else 0

# 统计RL充电次数（每个时间步的充电动作）
rl_charging_pattern = re.findall(r'【充电动作】\s*\n((?:  区域\d+: \d+辆\n)*)', rl_content)
rl_charging_per_step = []
for match in rl_charging_pattern:
    if match.strip():
        counts = re.findall(r'区域\d+: (\d+)辆', match)
        step_total = sum(int(c) for c in counts)
        rl_charging_per_step.append(step_total)
    else:
        rl_charging_per_step.append(0)

# 统计最优解充电次数
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
print("奖励对比分析")
print("="*80)
print(f"\nRL总奖励: {rl_total_reward:,.2f}")
print(f"最优解总奖励: {opt_total_reward:,.2f}")
print(f"差异: {rl_total_reward - opt_total_reward:,.2f} ({(rl_total_reward/opt_total_reward - 1)*100:.2f}%)")

print(f"\n充电统计:")
print(f"RL总充电次数: {sum(rl_charging_per_step)}")
print(f"最优解总充电次数: {sum(opt_charging_per_step)}")

print(f"\n每步充电对比:")
print(f"{'时间步':<8} {'RL充电':<10} {'最优解充电':<10} {'差异':<10}")
print("-"*40)
for i in range(min(len(rl_charging_per_step), len(opt_charging_per_step))):
    diff = rl_charging_per_step[i] - opt_charging_per_step[i]
    print(f"{i:<8} {rl_charging_per_step[i]:<10} {opt_charging_per_step[i]:<10} {diff:+<10}")

# 估算奖励构成
rl_charging_reward = sum(rl_charging_per_step) * 18000
opt_charging_reward = sum(opt_charging_per_step) * 18000

print(f"\n奖励构成估算:")
print(f"RL充电奖励 (18000/车): {rl_charging_reward:,.2f}")
print(f"RL服务+SOC奖励: {rl_total_reward - rl_charging_reward:,.2f}")
print(f"\n最优解充电奖励 (18000/车): {opt_charging_reward:,.2f}")
print(f"最优解服务+SOC奖励: {opt_total_reward - opt_charging_reward:,.2f}")

print(f"\n关键发现:")
if rl_total_reward > opt_total_reward:
    print(f"✓ RL超过最优解 {rl_total_reward - opt_total_reward:,.2f}")
    print(f"  - 充电奖励差异: {rl_charging_reward - opt_charging_reward:,.2f}")
    print(f"  - 服务+SOC差异: {(rl_total_reward - rl_charging_reward) - (opt_total_reward - opt_charging_reward):,.2f}")
