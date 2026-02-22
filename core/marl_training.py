# -*- coding: utf-8 -*-
"""
MARL Phase-1: Multi-Agent Training Loop
Created on 2026-02-19

实现多智能体训练循环
"""
import numpy as np
import torch

# =========2026.2.19修改，MARL Phase-1：多智能体训练循环==========
def marl_train(env, agent, cfg, path=None):
    """
    多智能体训练循环

    Args:
        env: 环境实例
        agent: MARL agent实例（IPPOAgent）
        cfg: 配置字典
        path: 模型保存路径

    Returns:
        res_dic: 包含所有训练数据的字典
    """
    print("开始MARL训练...")
    print(f"智能体数量: {env.n_agents}")
    print(f"局部观测维度: 7 (含agent_id)")
    print(f"动作维度: 3 (每个智能体)")
    print(f"全局状态维度: {env.observation_space.shape[0]}")

    rewards = []
    raw_rewards = []
    ma_rewards = []  # 移动平均奖励
    critic_losses = []
    actor_losses = []
    rld_history = []
    svc_history = []
    srd_history = []
    info_missing_counts = {'rld': 0, 'svc': 0, 'srd': 0}

    # P1-5: Early Stopping（防止过拟合）
    best_reward = -float('inf')
    best_episode = 0
    patience = 100  # 2026.2.22：方案3调整，进一步增加patience以允许充分训练（50->100）
    patience_counter = 0

    reward_scale = float(cfg.get('reward_scale', 1.0))
    if reward_scale <= 0:
        reward_scale = 1.0

    if hasattr(env, 'set_experiment_variant'):
        env.set_experiment_variant(cfg.get('exp_variant', 'a1'))
    if hasattr(env, 'set_reward_mode'):
        env.set_reward_mode(cfg.get('reward_mode', 'base'))

    for i_ep in range(cfg['train_eps']):
        ep_reward = 0.0
        ep_reward_raw = 0.0
        ep_step = 0

        # 重置环境
        global_state = env.reset()

        # 获取所有智能体的局部观测
        obs_n = []
        for agent_id in range(env.n_agents):
            obs = env.get_agent_obs(agent_id)
            obs_n.append(obs)
        obs_n = np.array(obs_n)  # (n_agents, 7)

        done = False

        while not done:
            # 采样联合动作
            add_noise = True  # 训练时添加探索噪声
            noise_scale = max(0.1, 0.3 * (1 - i_ep / cfg['train_eps']))  # 噪声衰减
            actions = agent.sample_action(obs_n, add_noise=add_noise, noise_scale=noise_scale)

            # 执行动作
            next_global_state, reward, done, info = env.step(actions)
            scaled_reward = reward / reward_scale

            # P0-3: 可选的reward监控（每10步打印一次）
            if ep_step % 10 == 0 and ep_step > 0:
                print(f"  Step {ep_step}: Raw reward={reward:.2f}, Scaled={scaled_reward:.4f}")

            # 获取下一时刻的局部观测
            next_obs_n = []
            for agent_id in range(env.n_agents):
                next_obs = env.get_agent_obs(agent_id)
                next_obs_n.append(next_obs)
            next_obs_n = np.array(next_obs_n)  # (n_agents, 7)

            # 存储transition
            agent.memory.push(
                obs_n=obs_n,
                act_n=actions,
                global_state=global_state,
                reward=scaled_reward,
                next_obs_n=next_obs_n,
                next_global_state=next_global_state,
                done=done
            )

            # 更新网络
            if len(agent.memory) >= cfg['batch_size']:
                update_result = agent.update()
                if update_result is not None:
                    # P1-6: 处理新的返回格式（包含梯度监控）
                    if len(update_result) == 3:
                        critic_loss, actor_loss, grad_info = update_result
                    else:
                        critic_loss, actor_loss = update_result
                        grad_info = {}
                    critic_losses.append(critic_loss)
                    actor_losses.append(actor_loss)

            # 更新状态
            obs_n = next_obs_n
            global_state = next_global_state
            ep_reward += scaled_reward
            ep_reward_raw += reward
            ep_step += 1

        # 记录环境指标（从info字典获取）
        if 'rld' in info:
            rld_history.append(info['rld'])
        else:
            info_missing_counts['rld'] += 1
        if 'svc' in info:
            svc_history.append(info['svc'])
        else:
            info_missing_counts['svc'] += 1
        if 'srd' in info:
            srd_history.append(info['srd'])
        else:
            info_missing_counts['srd'] += 1

        # 记录奖励
        rewards.append(ep_reward)
        raw_rewards.append(ep_reward_raw)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)

        # 打印进度 - 每轮都输出
        print(f"{i_ep+1}/{cfg['train_eps']}, "
              f"Reward(raw/scaled): {ep_reward_raw:.2f}/{ep_reward:.2f}, "
              f"MA: {ma_rewards[-1]:.2f}, "
              f"Steps: {ep_step}")

        # P1-5: Early Stopping检查
        if ma_rewards[-1] > best_reward:
            best_reward = ma_rewards[-1]
            best_episode = i_ep
            patience_counter = 0
            if path is not None:
                agent.save_model(path + 'marl_model_best.pth')
                print(f"  → New best model saved: {best_reward:.2f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\n早停触发！在第 {i_ep+1} 轮停止训练")
            print(f"最佳奖励 {best_reward:.2f} 出现在第 {best_episode+1} 轮")
            break

        # 定期保存模型
        if (i_ep + 1) % 100 == 0 and path is not None:
            agent.save_model(path + f'marl_model_ep{i_ep+1}.pth')
            print(f"模型已保存: {path}marl_model_ep{i_ep+1}.pth")

    # 返回结果字典（与main.py格式一致）
    res_dic = {
        'rewards': rewards,
        'raw_rewards': raw_rewards,
        'ma_rewards': ma_rewards,
        'critic_losses': critic_losses,
        'actor_losses': actor_losses,
        'rld_history': rld_history,
        'svc_history': svc_history,
        'srd_history': srd_history,
        'info_missing_counts': info_missing_counts,
    }

    return res_dic


def marl_test(env, agent, cfg, path=None):
    """
    多智能体测试循环

    Args:
        env: 环境实例
        agent: MARL agent实例（IPPOAgent）
        cfg: 配置字典
        path: 结果保存路径

    Returns:
        rewards: 测试奖励列表
    """
    print("开始MARL测试...")

    rewards = []
    reward_scale = float(cfg.get('reward_scale', 1.0))
    if reward_scale <= 0:
        reward_scale = 1.0

    if hasattr(env, 'set_experiment_variant'):
        env.set_experiment_variant(cfg.get('exp_variant', 'a1'))
    if hasattr(env, 'set_reward_mode'):
        env.set_reward_mode(cfg.get('reward_mode', 'base'))

    for i_ep in range(cfg['test_eps']):
        ep_reward = 0
        ep_step = 0

        # 重置环境
        global_state = env.reset()

        # 获取所有智能体的局部观测
        obs_n = []
        for agent_id in range(env.n_agents):
            obs = env.get_agent_obs(agent_id)
            obs_n.append(obs)
        obs_n = np.array(obs_n)  # (n_agents, 7)

        done = False

        while not done:
            # 采样联合动作（测试时不添加噪声）
            actions = agent.sample_action(obs_n, add_noise=False)

            # 执行动作
            next_global_state, reward, done, info = env.step(actions)

            # 获取下一时刻的局部观测
            next_obs_n = []
            for agent_id in range(env.n_agents):
                next_obs = env.get_agent_obs(agent_id)
                next_obs_n.append(next_obs)
            next_obs_n = np.array(next_obs_n)  # (n_agents, 7)

            # 更新状态
            obs_n = next_obs_n
            global_state = next_global_state
            ep_reward += reward / reward_scale
            ep_step += 1

        # 记录奖励
        rewards.append(ep_reward)

        print(f"测试回合：{i_ep+1}/{cfg['test_eps']}，"
              f"奖励：{ep_reward:.2f}，"
              f"步数：{ep_step}")

    # 保存测试数据
    if path is not None:
        np.save(path + 'marl_test_rewards.npy', rewards)

    print(f"测试完成，平均奖励：{np.mean(rewards):.2f}")

    return rewards
# ======修改结束
