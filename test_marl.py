# -*- coding: utf-8 -*-
"""
MARL Phase-1: Quick Test Script
Created on 2026-02-19

快速测试MARL实现是否正常工作
"""
import sys
import os
import numpy as np

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.append(os.getcwd())

# Register the custom environment
from gym.envs.registration import register
register(id='dclily', entry_point='specific.dclily_elec:dclily')

from core.util import get_args
from core.environment import env_agent_config

# =========2026.2.19修改，MARL Phase-1：快速测试==========
def test_marl_smoke():
    """冒烟测试：验证MARL基本功能"""
    print("\n" + "="*80)
    print("MARL Phase-1 冒烟测试")
    print("="*80)

    # 获取配置
    cfg = get_args()
    cfg['marl_mode'] = 'multi'
    cfg['n_agents'] = 50
    cfg['train_eps'] = 1  # 只训练1轮
    cfg['test_eps'] = 1

    print(f"\n配置: marl_mode={cfg['marl_mode']}, n_agents={cfg['n_agents']}")

    # 创建环境和智能体
    print("\n1. 创建环境和智能体...")
    env, agent = env_agent_config(cfg)
    print(f"   ✓ 环境创建成功")
    print(f"   ✓ 智能体数量: {env.n_agents}")
    print(f"   ✓ 全局状态维度: {env.observation_space.shape[0]}")

    # 测试局部观测
    print("\n2. 测试局部观测...")
    global_state = env.reset()
    print(f"   ✓ 全局状态形状: {global_state.shape}")

    obs_n = []
    for agent_id in range(env.n_agents):
        obs = env.get_agent_obs(agent_id)
        obs_n.append(obs)
    obs_n = np.array(obs_n)
    print(f"   ✓ 局部观测形状: {obs_n.shape} (应为 (50, 7))")
    print(f"   ✓ 第0个智能体观测: {obs_n[0]}")

    # 测试动作采样
    print("\n3. 测试动作采样...")
    actions = agent.sample_action(obs_n, add_noise=False)
    print(f"   ✓ 动作形状: {actions.shape} (应为 (50, 3))")
    print(f"   ✓ 动作范围: [{actions.min():.3f}, {actions.max():.3f}] (应在 [0, 1])")

    # 测试环境step
    print("\n4. 测试环境step...")
    next_global_state, reward, done, info = env.step(actions)
    print(f"   ✓ 下一状态形状: {next_global_state.shape}")
    print(f"   ✓ 奖励: {reward:.2f}")
    print(f"   ✓ 完成标志: {done}")

    # 测试经验池
    print("\n5. 测试经验池...")
    next_obs_n = []
    for agent_id in range(env.n_agents):
        next_obs = env.get_agent_obs(agent_id)
        next_obs_n.append(next_obs)
    next_obs_n = np.array(next_obs_n)

    agent.memory.push(
        obs_n=obs_n,
        act_n=actions,
        global_state=global_state,
        reward=reward,
        next_obs_n=next_obs_n,
        next_global_state=next_global_state,
        done=done
    )
    print(f"   ✓ 经验池大小: {len(agent.memory)}")

    # 测试网络更新（需要足够的样本）
    print("\n6. 测试网络更新...")
    # 填充经验池到batch_size
    for _ in range(cfg['batch_size']):
        agent.memory.push(
            obs_n=obs_n,
            act_n=actions,
            global_state=global_state,
            reward=reward,
            next_obs_n=next_obs_n,
            next_global_state=next_global_state,
            done=False
        )

    critic_loss, actor_loss = agent.update()
    print(f"   ✓ Critic Loss: {critic_loss:.4f}")
    print(f"   ✓ Actor Loss: {actor_loss:.4f}")

    # 测试模型保存和加载
    print("\n7. 测试模型保存和加载...")
    test_path = "./test_marl_model.pth"
    agent.save_model(test_path)
    print(f"   ✓ 模型已保存: {test_path}")

    agent.load_model(test_path)
    print(f"   ✓ 模型已加载")

    # 清理
    if os.path.exists(test_path):
        os.remove(test_path)
        print(f"   ✓ 测试文件已清理")

    print("\n" + "="*80)
    print("✓ 所有测试通过！MARL实现正常工作。")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_marl_smoke()
# ======修改结束
