"""
基准测试脚本 - 对比不同策略的性能
用于对比真实执行的策略性能，并单独展示MILP理论上界

对比策略（真实执行）：
1. 随机策略 (Random)
2. 贪心策略 (Greedy) - 根据需求分配车辆
3. RL 策略 (DDPG) - 当前训练的模型

理论上界（单独展示）：
- MILP全局最优解（从optimal_solver.py求解结果读取）
- 滚动时域优化解（从optimal_solver.py求解结果读取）

对比指标：
- Episode Reward
- 服务率 (SVC)
- 调度损失距离 (RLD)
- 充电效率（平均电量、充电次数）

完整实验流程：
# 1. 训练 RL 模型（会自动保存到 trained_model.pth）
python main.py --train_eps 700

# 2. 求解理论上界（可选）
python optimal_solver.py --method both

# 3. 运行算法对比（会自动加载 trained_model.pth）
python baseline_comparison.py

# 4. 生成对比图表
python plot_comparison.py
"""

import numpy as np
import sys
import os
import gym
from gym.envs.registration import register
import json  # 修改

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 注册环境
register(
    id='dclily',
    entry_point='specific.dclily_elec:dclily',
)

from core.util import get_args
from core.environment import env_agent_config


class BaselinePolicy:
    """基准策略基类"""
    def __init__(self, env):
        self.env = env
        self.zone_num = env.action_space.shape[0] // 3

    def get_action(self, state):
        """返回动作，子类需要实现"""
        raise NotImplementedError


class RandomPolicy(BaselinePolicy):
    """随机策略"""
    def get_action(self, state):
        return self.env.action_space.sample()


class GreedyPolicy(BaselinePolicy):
    """贪心策略 - 根据当前需求分配车辆"""
    def get_action(self, state):
        # 简单的贪心策略：
        # 1. 发车倾向：均匀分布
        # 2. 目的地倾向：根据需求分布
        # 3. 充电倾向：根据平均电量

        action = np.zeros(self.zone_num * 3)

        # 发车倾向：均匀
        action[0:self.zone_num] = 1.0 / self.zone_num

        # 目的地倾向：根据 OD 需求（state 的前半部分）
        # 这里简化处理，实际需要解析 state
        demand_in = state[0:self.zone_num]
        demand_sum = np.sum(demand_in) + 1e-6
        action[self.zone_num:2*self.zone_num] = demand_in / demand_sum

        # 充电倾向：如果平均电量低，增加充电倾向
        # 简化：设置为固定值
        action[2*self.zone_num:3*self.zone_num] = 0.3

        return action


# =========2026.2.3修改，移除GlobalOptimalPolicy和RollingHorizonPolicy类==========
# 这两个类不是真正执行MILP/滚动时域的决策，只是启发式策略
# 它们从JSON文件读取reward但执行的是固定的启发式动作
# 这会导致误导性的对比结果（如SVC=0%, RLD=0%）
# 因此移除这两个类，改为单独展示MILP理论上界
# ======修改结束


# =========2026.2.3修改，添加读取MILP理论上界的函数==========
def load_theoretical_bounds():
    """
    读取MILP求解器计算的理论上界
    这些是离线求解的最优解，不参与env.step()的实际执行对比
    """
    bounds = {}

    # 读取全局最优解
    if os.path.exists("global_optimal_solution.json"):
        with open("global_optimal_solution.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            bounds['global_optimal'] = {
                'name': 'Global-Optimal (MILP)',
                'reward': data.get('total_reward', None),
                'method': data.get('method', 'Unknown'),
                'status': data.get('status', 'Unknown'),
                'solve_time': data.get('solve_time', 0),
                'mip_gap': data.get('mip_gap', 0)
            }

    # 读取滚动时域优化解
    if os.path.exists("rolling_horizon_solution.json"):
        with open("rolling_horizon_solution.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            bounds['rolling_horizon'] = {
                'name': 'Rolling-Horizon',
                'reward': data.get('total_reward', None),
                'method': data.get('method', 'Unknown'),
                'horizon': data.get('horizon', 0),
                'solve_time': data.get('solve_time', 0)
            }

    return bounds
# ======修改结束


def evaluate_policy(env, policy, num_episodes=10, policy_name="Unknown", save_actions=False):
    """评估策略性能"""
    print(f"\n{'='*60}")
    print(f"评估策略: {policy_name}")
    print(f"{'='*60}")

    all_rewards = []
    all_svc = []
    all_rld = []
    all_srd = []
    all_avg_soc = []

    # 新增：记录第一个episode的动作
    first_ep_actions = [] if save_actions else None

    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0
        ep_svc_list = []
        ep_rld_list = []
        ep_srd_list = []
        ep_soc_list = []

        done = False
        step = 0

        while not done:
            # 获取动作
            if hasattr(policy, 'predict_action'):
                action = policy.predict_action(state)
            else:
                action = policy.get_action(state)

            # 新增：记录第一个episode的动作
            if save_actions and ep == 0:
                first_ep_actions.append({
                    'step': step,
                    'action': action.tolist() if hasattr(action, 'tolist') else list(action),
                    'state': state.tolist() if hasattr(state, 'tolist') else list(state)
                })

            # 执行动作
            next_state, reward, done, info = env.step(action)
            ep_reward += reward

            # 记录指标
            if 'svc' in info:
                ep_svc_list.append(info['svc'])
            if 'rld' in info:
                ep_rld_list.append(info['rld'])
            if 'srd' in info:
                ep_srd_list.append(info['srd'])

            # 记录平均电量
            avg_soc = np.mean([car[1] for car in env.car])
            ep_soc_list.append(avg_soc)

            state = next_state
            step += 1

        all_rewards.append(ep_reward)
        all_svc.append(np.mean(ep_svc_list) if ep_svc_list else 0)
        all_rld.append(np.mean(ep_rld_list) if ep_rld_list else 0)
        all_srd.append(np.mean(ep_srd_list) if ep_srd_list else 0)
        all_avg_soc.append(np.mean(ep_soc_list) if ep_soc_list else 0)

        print(f"Episode {ep+1}/{num_episodes}: Reward={ep_reward:.2f}, "
              f"SVC={all_svc[-1]:.3f}, RLD={all_rld[-1]:.2f}, "
              f"AvgSOC={all_avg_soc[-1]:.1f}")

    # 计算统计信息
    results = {
        'policy_name': policy_name,
        'reward_mean': np.mean(all_rewards),
        'reward_std': np.std(all_rewards),
        'svc_mean': np.mean(all_svc),
        'svc_std': np.std(all_svc),
        'rld_mean': np.mean(all_rld),
        'rld_std': np.std(all_rld),
        'srd_mean': np.mean(all_srd),
        'srd_std': np.std(all_srd),
        'avg_soc_mean': np.mean(all_avg_soc),
        'avg_soc_std': np.std(all_avg_soc),
    }

    print(f"\n{policy_name} 平均结果:")
    print(f"  Reward: {results['reward_mean']:.2f} ± {results['reward_std']:.2f}")
    print(f"  SVC:    {results['svc_mean']:.3f} ± {results['svc_std']:.3f}")
    print(f"  RLD:    {results['rld_mean']:.2f} ± {results['rld_std']:.2f}")
    print(f"  SRD:    {results['srd_mean']:.3f} ± {results['srd_std']:.3f}")
    print(f"  AvgSOC: {results['avg_soc_mean']:.1f} ± {results['avg_soc_std']:.1f}")

    # 新增：保存动作到文件
    if save_actions and first_ep_actions:
        action_file = f'{policy_name}_actions.txt'
        with open(action_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"{policy_name} 策略 - 动作记录（第1个episode）\n")
            f.write("="*80 + "\n\n")
            f.write(f"总奖励: {all_rewards[0]:.2f}\n")
            f.write(f"总步数: {len(first_ep_actions)}\n\n")

            for action_data in first_ep_actions:
                f.write(f"{'='*80}\n")
                f.write(f"时间步 {action_data['step']}\n")
                f.write(f"{'='*80}\n")
                f.write(f"动作向量: {action_data['action']}\n\n")

        print(f"  动作已保存到: {action_file}")

    return results


def main():
    print("="*60)
    print("基准测试：对比不同策略的性能")
    print("="*60)

    # 加载配置
    cfg = get_args()

    # 初始化环境
    env = gym.make('dclily')

    # 设置测试模式
    env.set_training_mode(0)

    # 测试轮数
    num_test_episodes = 20

    all_results = []

    # 1. 测试随机策略
    print("\n" + "="*60)
    print("1. 随机策略 (Random Policy)")
    print("="*60)
    random_policy = RandomPolicy(env)
    random_results = evaluate_policy(env, random_policy, num_test_episodes, "Random")
    all_results.append(random_results)

    # 2. 测试贪心策略
    print("\n" + "="*60)
    print("2. 贪心策略 (Greedy Policy)")
    print("="*60)
    greedy_policy = GreedyPolicy(env)
    greedy_results = evaluate_policy(env, greedy_policy, num_test_episodes, "Greedy")
    all_results.append(greedy_results)

    # 3. 测试 RL 策略（如果有训练好的模型）
    print("\n" + "="*60)
    print("3. RL 策略 (DDPG Policy)")
    print("="*60)
    try:
        _, agent = env_agent_config(cfg)
        # =========2026.1.31修改，加载训练好的模型==========
        try:
            agent.load_model('trained_model.pth')
            print("[INFO] 成功加载训练好的模型")
        except Exception as e:
            print(f"[WARNING] 未找到训练好的模型: {e}")
            print("[WARNING] 使用随机初始化网络")
        # ======修改结束
        rl_results = evaluate_policy(env, agent, num_test_episodes, "RL-DDPG", save_actions=True)
        all_results.append(rl_results)
    except Exception as e:
        print(f"无法加载 RL 模型: {e}")
        print("跳过 RL 策略测试")

    # =========2026.2.3修改，读取MILP理论上界（不参与实际执行对比）==========
    # 读取理论上界
    theoretical_bounds = load_theoretical_bounds()
    # ======修改结束

    # =========2026.2.3修改，更新对比报告格式==========
    # 生成对比报告
    print("\n" + "="*60)
    print("实际执行策略对比报告")
    print("="*60)

    # 打印表格
    print(f"\n{'策略':<15} {'Reward':<20} {'SVC':<15} {'RLD':<15} {'AvgSOC':<15}")
    print("-" * 80)
    for result in all_results:
        print(f"{result['policy_name']:<15} "
              f"{result['reward_mean']:>8.2f}±{result['reward_std']:<8.2f} "
              f"{result['svc_mean']:>6.3f}±{result['svc_std']:<6.3f} "
              f"{result['rld_mean']:>6.2f}±{result['rld_std']:<6.2f} "
              f"{result['avg_soc_mean']:>6.1f}±{result['avg_soc_std']:<6.1f}")

    # 找出最优策略（只在实际执行的策略中比较）
    best_reward_idx = np.argmax([r['reward_mean'] for r in all_results])
    best_svc_idx = np.argmax([r['svc_mean'] for r in all_results])
    best_rld_idx = np.argmin([r['rld_mean'] for r in all_results])

    print(f"\n实际执行策略中的最优:")
    print(f"  最高 Reward: {all_results[best_reward_idx]['policy_name']}")
    print(f"  最高 SVC:    {all_results[best_svc_idx]['policy_name']}")
    print(f"  最低 RLD:    {all_results[best_rld_idx]['policy_name']}")

    # 显示理论上界
    if theoretical_bounds:
        print("\n" + "="*60)
        print("理论上界（MILP离线求解，不参与实际执行对比）")
        print("="*60)

        for key, bound in theoretical_bounds.items():
            if bound['reward'] is not None:
                print(f"\n{bound['name']}:")
                print(f"  Reward: {bound['reward']:.2f}")
                print(f"  方法: {bound['method']}")
                if 'status' in bound:
                    print(f"  状态: {bound['status']}")
                if 'mip_gap' in bound and bound['mip_gap'] > 0:
                    print(f"  MIP Gap: {bound['mip_gap']*100:.2f}%")
                print(f"  求解时间: {bound['solve_time']:.2f}秒")

        # 计算RL与理论上界的差距
        if 'global_optimal' in theoretical_bounds and theoretical_bounds['global_optimal']['reward'] is not None:
            rl_reward = all_results[best_reward_idx]['reward_mean']
            optimal_reward = theoretical_bounds['global_optimal']['reward']
            gap = abs(rl_reward - optimal_reward)
            gap_percent = gap / abs(optimal_reward) * 100

            print(f"\n性能差距分析:")
            print(f"  RL最佳 Reward: {rl_reward:.2f}")
            print(f"  理论最优 Reward: {optimal_reward:.2f}")
            print(f"  差距: {gap:.2f} ({gap_percent:.1f}%)")
            print(f"  RL达到理论最优的: {(1 - gap_percent/100)*100:.1f}%")

    # 保存结果（只保存实际执行的策略结果）
    output_data = {
        'executed_policies': all_results,
        'theoretical_bounds': theoretical_bounds
    }

    with open('baseline_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: baseline_comparison.json")
    print(f"  - executed_policies: 实际执行的策略结果（Random, Greedy, RL-DDPG）")
    print(f"  - theoretical_bounds: 理论上界（MILP, Rolling-Horizon）")
    # ======修改结束


if __name__ == "__main__":
    main()
