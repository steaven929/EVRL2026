"""
RL vs 最优解动作差异分析脚本
分析 RL-DDPG 和全局最优解之间的关键差异
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取数据
with open('baseline_comparison.json', 'r') as f:
    comparison = json.load(f)

with open('global_optimal_solution.json', 'r') as f:
    optimal = json.load(f)

print("=" * 80)
print("RL vs 最优解 - 关键差异分析")
print("=" * 80)

# 1. 性能差距分析
rl_reward = -3286.0
optimal_reward = 469.80
gap = optimal_reward - rl_reward
gap_percent = (gap / abs(optimal_reward)) * 100

print(f"\n【1. 性能差距】")
print(f"  RL-DDPG Reward:     {rl_reward:.2f}")
print(f"  最优解 Reward:      {optimal_reward:.2f}")
print(f"  绝对差距:           {gap:.2f}")
print(f"  相对差距:           {gap_percent:.1f}%")
print(f"  RL 仅达到最优解的:  {(rl_reward/optimal_reward)*100:.1f}%")

# 2. 充电策略分析
charging_actions = optimal['charging_actions']
total_charging = sum(sum(step) for step in charging_actions)
avg_charging_per_step = total_charging / len(charging_actions)

print(f"\n【2. 最优解充电策略】")
print(f"  总充电次数:         {total_charging}")
print(f"  平均每步充电:       {avg_charging_per_step:.2f}")
print(f"  充电分布:")
for t, charges in enumerate(charging_actions):
    if sum(charges) > 0:
        print(f"    时间步 {t}: {charges} (总计: {sum(charges)})")

# 3. 订单服务分析
served_orders = optimal['served_orders']
total_served = sum(sum(sum(row) for row in step) for step in served_orders)
avg_served_per_step = total_served / len(served_orders)

print(f"\n【3. 最优解订单服务】")
print(f"  总服务订单数:       {total_served}")
print(f"  平均每步服务:       {avg_served_per_step:.2f}")

# 4. 调度动作分析
repositioned = optimal['repositioned_orders']
total_reposition = sum(sum(sum(row) for row in step) for step in repositioned)
avg_reposition_per_step = total_reposition / len(repositioned)

print(f"\n【4. 最优解调度策略】")
print(f"  总调度次数:         {total_reposition}")
print(f"  平均每步调度:       {avg_reposition_per_step:.2f}")

# 5. RL 动作分析
print(f"\n【5. RL-DDPG 动作特征】")
print(f"  动作空间维度:       12 (4区域 × 3类型)")
print(f"  动作类型:")
print(f"    [0:4]   发车倾向")
print(f"    [4:8]   目的地倾向")
print(f"    [8:12]  充电倾向")

# 读取 RL 动作文件并分析
print(f"\n【6. RL 动作值分析】")
with open('RL-DDPG_actions.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# 提取所有动作向量
import re
action_pattern = r'动作向量: \[(.*?)\]'
actions = re.findall(action_pattern, content)

if actions:
    action_values = []
    for action_str in actions:
        values = [float(x.strip()) for x in action_str.split(',')]
        action_values.append(values)

    action_array = np.array(action_values)

    print(f"  总步数: {len(action_values)}")
    print(f"  动作值范围: [{action_array.min():.4f}, {action_array.max():.4f}]")
    print(f"  动作值均值: {action_array.mean():.4f}")
    print(f"  动作值标准差: {action_array.std():.4f}")

    # 分析三种动作类型
    departure_pref = action_array[:, 0:4]
    destination_pref = action_array[:, 4:8]
    charging_pref = action_array[:, 8:12]

    print(f"\n  发车倾向 [0:4]:")
    print(f"    均值: {departure_pref.mean():.4f}, 标准差: {departure_pref.std():.4f}")
    print(f"  目的地倾向 [4:8]:")
    print(f"    均值: {destination_pref.mean():.4f}, 标准差: {destination_pref.std():.4f}")
    print(f"  充电倾向 [8:12]:")
    print(f"    均值: {charging_pref.mean():.4f}, 标准差: {charging_pref.std():.4f}")

# 7. 关键问题识别
print(f"\n{'=' * 80}")
print(f"【关键问题识别】")
print(f"{'=' * 80}")

problems = []

# 问题1: 动作值过于集中
if action_array.std() < 0.02:
    problems.append({
        'id': 1,
        'name': '动作值过于集中',
        'severity': '严重',
        'description': f'RL 输出的动作值标准差仅 {action_array.std():.4f}，几乎没有区分度',
        'impact': '无法有效指导 Gurobi 进行差异化调度',
        'solution': '增加探索噪声、调整网络架构、修改 reward shaping'
    })

# 问题2: 充电策略差异
if avg_charging_per_step > 5:  # 最优解平均每步充电 9.33
    problems.append({
        'id': 2,
        'name': '充电策略不足',
        'severity': '严重',
        'description': f'最优解平均每步充电 {avg_charging_per_step:.2f} 辆，RL 可能充电不足',
        'impact': '车辆电量不足导致无法服务订单，reward 大幅下降',
        'solution': '增加充电奖励、调整 SOC 惩罚权重、修改充电决策逻辑'
    })

# 问题3: 性能差距巨大
if gap > 3000:
    problems.append({
        'id': 3,
        'name': '性能差距巨大',
        'severity': '严重',
        'description': f'RL 与最优解差距 {gap:.2f}，相差 {gap_percent:.1f}%',
        'impact': 'RL 学习效果极差，可能陷入局部最优或探索不足',
        'solution': '重新设计 reward 函数、增加训练轮数、调整超参数'
    })

for i, problem in enumerate(problems, 1):
    print(f"\n问题 {problem['id']}: {problem['name']} [{problem['severity']}]")
    print(f"  描述: {problem['description']}")
    print(f"  影响: {problem['impact']}")
    print(f"  建议: {problem['solution']}")

print(f"\n{'=' * 80}")
print(f"分析完成！")
print(f"{'=' * 80}")
