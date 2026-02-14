"""
小场景最优解求解指南
场景：4区域，18辆车

针对小场景的特点，提供多种求解最优解的方法
"""

# ============================================================
# 方法概述
# ============================================================

"""
对于 4区域 18辆车 的小场景，有以下几种找最优解的方法：

方法1：网格搜索 (Grid Search)
- 适用性：小场景
- 原理：遍历动作空间的离散化网格点
- 优点：简单直接，保证找到离散空间的最优解
- 缺点：计算量大，只能离散化

方法2：贝叶斯优化 (Bayesian Optimization)
- 适用性：中小场景
- 原理：用高斯过程建模 reward 函数，智能采样
- 优点：样本效率高，适合昂贵的评估
- 缺点：需要额外的库支持

方法3：进化算法 (Evolutionary Algorithm)
- 适用性：各种场景
- 原理：模拟自然选择，迭代优化动作序列
- 优点：不需要梯度，容易实现
- 缺点：收敛较慢

方法4：完美信息 MPC (Model Predictive Control)
- 适用性：小场景
- 原理：假设知道未来需求，用 Gurobi 求解多步优化
- 优点：理论最优
- 缺点：需要完美信息（实际不可得）

推荐方案：
- 对于你的场景（4区域18车），推荐使用 方法1（网格搜索）+ 方法3（进化算法）
"""

# ============================================================
# 实现：网格搜索
# ============================================================

import numpy as np
import itertools


def grid_search_optimal(env, grid_points=5, num_episodes=5):
    """
    网格搜索找最优动作分布

    参数:
        env: 环境
        grid_points: 每个维度的网格点数（越大越精确但越慢）
        num_episodes: 每个动作组合测试的 episode 数

    返回:
        best_action: 最优动作
        best_reward: 最优奖励
    """
    print("="*60)
    print(f"网格搜索最优解 (grid_points={grid_points})")
    print("="*60)

    zone_num = 4

    # 为每个区域生成网格点
    # 动作空间：[发车倾向(4), 目的地倾向(4), 充电倾向(4)]
    grid_values = np.linspace(0.1, 0.9, grid_points)

    # 简化搜索：只搜索关键参数
    # 完整搜索空间太大：grid_points^12 种组合
    # 简化：假设每类动作内部使用相同的分布模式

    print(f"\n搜索空间大小: {grid_points**3} 种组合")
    print("(简化为3个参数：发车集中度、目的地集中度、充电倾向)")

    best_reward = -np.inf
    best_params = None
    best_action = None

    tested = 0
    total = grid_points ** 3

    # 遍历参数组合
    for dep_conc in grid_values:  # 发车集中度
        for dest_conc in grid_values:  # 目的地集中度
            for charge_pref in grid_values:  # 充电倾向

                # 构造动作
                action = construct_action(dep_conc, dest_conc, charge_pref, zone_num)

                # 测试这个动作
                rewards = []
                for ep in range(num_episodes):
                    state = env.reset()
                    ep_reward = 0
                    done = False

                    while not done:
                        next_state, reward, done, _ = env.step(action)
                        ep_reward += reward
                        state = next_state

                    rewards.append(ep_reward)

                avg_reward = np.mean(rewards)

                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_params = (dep_conc, dest_conc, charge_pref)
                    best_action = action.copy()

                tested += 1
                if tested % 10 == 0:
                    print(f"  进度: {tested}/{total}, 当前最优: {best_reward:.2f}")

    print(f"\n搜索完成!")
    print(f"  最优参数: dep={best_params[0]:.2f}, dest={best_params[1]:.2f}, charge={best_params[2]:.2f}")
    print(f"  最优奖励: {best_reward:.2f}")

    return best_action, best_reward, best_params


def construct_action(dep_conc, dest_conc, charge_pref, zone_num):
    """
    根据参数构造动作

    参数:
        dep_conc: 发车集中度 (0-1)，越大越集中在某些区域
        dest_conc: 目的地集中度 (0-1)
        charge_pref: 充电倾向 (0-1)
    """
    action = np.zeros(zone_num * 3)

    # 发车倾向：集中度决定分布的不均匀程度
    dep_dist = generate_distribution(zone_num, dep_conc)
    action[0:zone_num] = dep_dist

    # 目的地倾向
    dest_dist = generate_distribution(zone_num, dest_conc)
    action[zone_num:2*zone_num] = dest_dist

    # 充电倾向：简单设置为固定值
    action[2*zone_num:3*zone_num] = charge_pref

    return action


def generate_distribution(n, concentration):
    """
    生成分布
    concentration: 0 = 均匀分布, 1 = 高度集中
    """
    if concentration < 0.5:
        # 趋向均匀
        dist = np.ones(n) / n
    else:
        # 趋向集中
        dist = np.zeros(n)
        # 集中在前几个区域
        focus = max(1, int(n * (1 - concentration)))
        dist[:focus] = 1.0 / focus

    return dist


# ============================================================
# 实现：进化算法
# ============================================================

def evolutionary_search(env, population_size=20, generations=50, num_episodes=3):
    """
    进化算法搜索最优动作

    参数:
        env: 环境
        population_size: 种群大小
        generations: 进化代数
        num_episodes: 每个个体评估的 episode 数
    """
    print("="*60)
    print(f"进化算法搜索 (pop={population_size}, gen={generations})")
    print("="*60)

    zone_num = 4
    action_dim = zone_num * 3

    # 初始化种群
    population = [np.random.rand(action_dim) for _ in range(population_size)]

    best_ever_reward = -np.inf
    best_ever_action = None

    for gen in range(generations):
        # 评估种群
        fitness = []
        for individual in population:
            # 归一化动作
            action = normalize_action(individual, zone_num)

            # 评估
            rewards = []
            for ep in range(num_episodes):
                state = env.reset()
                ep_reward = 0
                done = False

                while not done:
                    next_state, reward, done, _ = env.step(action)
                    ep_reward += reward
                    state = next_state

                rewards.append(ep_reward)

            avg_reward = np.mean(rewards)
            fitness.append(avg_reward)

            if avg_reward > best_ever_reward:
                best_ever_reward = avg_reward
                best_ever_action = action.copy()

        # 选择、交叉、变异
        population = evolve_population(population, fitness, population_size)

        if (gen + 1) % 10 == 0:
            print(f"  代数 {gen+1}/{generations}, 最优: {best_ever_reward:.2f}, 平均: {np.mean(fitness):.2f}")

    print(f"\n进化完成!")
    print(f"  最优奖励: {best_ever_reward:.2f}")

    return best_ever_action, best_ever_reward


def normalize_action(raw_action, zone_num):
    """归一化动作到合法范围"""
    action = np.clip(raw_action, 0, 1)

    # 归一化每个部分使其和为1
    for i in range(3):
        start = i * zone_num
        end = start + zone_num
        segment = action[start:end]
        segment_sum = np.sum(segment)
        if segment_sum > 0:
            action[start:end] = segment / segment_sum

    return action


def evolve_population(population, fitness, pop_size):
    """进化操作：选择、交叉、变异"""
    # 选择：锦标赛选择
    fitness = np.array(fitness)
    selected = []

    for _ in range(pop_size):
        # 随机选3个，取最好的
        candidates = np.random.choice(len(population), 3, replace=False)
        best_idx = candidates[np.argmax(fitness[candidates])]
        selected.append(population[best_idx].copy())

    # 交叉和变异
    new_population = []
    for i in range(0, pop_size, 2):
        parent1 = selected[i]
        parent2 = selected[i+1] if i+1 < pop_size else selected[0]

        # 交叉
        if np.random.rand() < 0.7:
            crossover_point = np.random.randint(1, len(parent1))
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        else:
            child1, child2 = parent1.copy(), parent2.copy()

        # 变异
        if np.random.rand() < 0.2:
            child1 += np.random.randn(len(child1)) * 0.1
        if np.random.rand() < 0.2:
            child2 += np.random.randn(len(child2)) * 0.1

        new_population.append(child1)
        new_population.append(child2)

    return new_population[:pop_size]


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    import gym
    from gym.envs.registration import register

    register(
        id='dclily',
        entry_point='specific.dclily_elec:dclily',
    )

    env = gym.make('dclily')
    env.set_training_mode(0)

    print("小场景最优解求解")
    print("场景：4区域，18辆车")
    print("="*60)

    # 方法1：网格搜索（精确但慢）
    print("\n方法1：网格搜索")
    grid_action, grid_reward, grid_params = grid_search_optimal(
        env, grid_points=3, num_episodes=5
    )

    # 方法2：进化算法（快速但近似）
    print("\n方法2：进化算法")
    evo_action, evo_reward = evolutionary_search(
        env, population_size=20, generations=30, num_episodes=3
    )

    # 对比结果
    print("\n" + "="*60)
    print("结果对比")
    print("="*60)
    print(f"网格搜索: {grid_reward:.2f}")
    print(f"进化算法: {evo_reward:.2f}")

    best_method = "网格搜索" if grid_reward > evo_reward else "进化算法"
    best_reward = max(grid_reward, evo_reward)
    best_action = grid_action if grid_reward > evo_reward else evo_action

    print(f"\n最优方法: {best_method}")
    print(f"最优奖励: {best_reward:.2f}")

    # 保存结果
    import json
    results = {
        "scene": {"zones": 4, "cars": 18},
        "grid_search": {"reward": float(grid_reward), "params": [float(p) for p in grid_params]},
        "evolutionary": {"reward": float(evo_reward)},
        "best_method": best_method,
        "best_reward": float(best_reward),
        "best_action": best_action.tolist()
    }

    with open('optimal_solution.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存到: optimal_solution.json")
