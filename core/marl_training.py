# -*- coding: utf-8 -*-
"""
MARL Phase-1: Multi-Agent Training Loop
"""
import numpy as np

from core.marl_metrics import compute_all_metrics
from core.improved_metrics import compute_all_improved_metrics


def _aggregate_episode_metric(values):
    if len(values) == 0:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 0.0
    return float(np.mean(finite)), float(finite[-1])


def marl_train(env, agent, cfg, path=None):
    """多智能体训练循环。"""
    print("开始MARL训练...")
    print(f"智能体数量: {env.n_agents}")
    print("局部观测维度: 7 (含agent_id)")
    print("动作维度: 3 (每个智能体)")
    print(f"全局状态维度: {env.observation_space.shape[0]}")

    rewards = []
    raw_rewards = []
    ma_rewards = []
    critic_losses = []
    actor_losses = []
    rld_history = []
    svc_history = []
    srd_history = []
    rld_last_history = []
    svc_last_history = []
    srd_last_history = []
    service_rate_history = []
    service_rate_last_history = []
    info_missing_counts = {'rld': 0, 'svc': 0, 'srd': 0, 'service_rate': 0}

    charging_efficiency_history = []
    vehicle_utilization_history = []
    fleet_health_history = []
    zone_coordination_history = []
    response_readiness_history = []
    charger_occupancy_rate_history = []
    charger_queue_delay_history = []
    energy_not_served_history = []
    safety_soc_violation_count_history = []
    charge_shift_success_rate_history = []

    weighted_service_rate_history = []
    effective_service_rate_history = []
    service_quality_index_history = []
    dispatch_efficiency_history = []
    effective_dispatch_rate_history = []
    dispatch_roi_history = []

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

        if hasattr(env, 'set_epoch'):
            env.set_epoch(i_ep)

        global_state = env.reset()
        obs_n = np.array([env.get_agent_obs(agent_id) for agent_id in range(env.n_agents)])

        done = False
        info = {}
        info = {}
        ep_metric_steps = {'rld': [], 'svc': [], 'srd': [], 'service_rate': []}
        ep_marl_metric_steps = {
            'charging_efficiency': [],
            'vehicle_utilization': [],
            'fleet_health': [],
            'zone_coordination': [],
            'response_readiness': [],
            'charger_occupancy_rate': [],
            'charger_queue_delay': [],
            'energy_not_served': [],
            'safety_soc_violation_count': [],
            'charge_shift_success_rate': [],
        }
        ep_improved_metric_steps = {
            'weighted_service_rate': [],
            'effective_service_rate': [],
            'service_quality_index': [],
            'dispatch_efficiency': [],
            'effective_dispatch_rate': [],
            'dispatch_roi': [],
        }

        while not done:
            add_noise = True
            noise_scale = max(0.1, 0.3 * (1 - i_ep / cfg['train_eps']))
            actions = agent.sample_action(obs_n, add_noise=add_noise, noise_scale=noise_scale)

            next_global_state, reward, done, info = env.step(actions)
            scaled_reward = reward / reward_scale

            for metric_key in ep_metric_steps:
                if metric_key in info:
                    ep_metric_steps[metric_key].append(info[metric_key])
                else:
                    info_missing_counts[metric_key] += 1

            marl_metrics = compute_all_metrics(env, info)
            improved_metrics = compute_all_improved_metrics(env, info)
            for metric_key in ep_marl_metric_steps:
                ep_marl_metric_steps[metric_key].append(marl_metrics.get(metric_key, np.nan))
            for metric_key in ep_improved_metric_steps:
                ep_improved_metric_steps[metric_key].append(improved_metrics.get(metric_key, np.nan))

            next_obs_n = np.array([env.get_agent_obs(agent_id) for agent_id in range(env.n_agents)])

            agent.memory.push(
                obs_n=obs_n,
                act_n=actions,
                global_state=global_state,
                reward=scaled_reward,
                next_obs_n=next_obs_n,
                next_global_state=next_global_state,
                done=done,
            )

            if len(agent.memory) >= cfg['batch_size']:
                critic_loss, actor_loss = agent.update()
                if critic_loss is not None:
                    critic_losses.append(critic_loss)
                if actor_loss is not None and np.isfinite(actor_loss):
                    actor_losses.append(actor_loss)

            obs_n = next_obs_n
            global_state = next_global_state
            ep_reward += scaled_reward
            ep_reward_raw += reward
            ep_step += 1

        rld_mean, rld_last = _aggregate_episode_metric(ep_metric_steps['rld'])
        svc_mean, svc_last = _aggregate_episode_metric(ep_metric_steps['svc'])
        srd_mean, srd_last = _aggregate_episode_metric(ep_metric_steps['srd'])
        service_rate_mean, service_rate_last = _aggregate_episode_metric(ep_metric_steps['service_rate'])

        rld_history.append(rld_mean)
        svc_history.append(svc_mean)
        srd_history.append(srd_mean)
        rld_last_history.append(rld_last)
        svc_last_history.append(svc_last)
        srd_last_history.append(srd_last)
        service_rate_history.append(service_rate_mean)
        service_rate_last_history.append(service_rate_last)

        charging_efficiency_mean, _ = _aggregate_episode_metric(ep_marl_metric_steps['charging_efficiency'])
        vehicle_utilization_mean, _ = _aggregate_episode_metric(ep_marl_metric_steps['vehicle_utilization'])
        fleet_health_mean, _ = _aggregate_episode_metric(ep_marl_metric_steps['fleet_health'])
        zone_coordination_mean, _ = _aggregate_episode_metric(ep_marl_metric_steps['zone_coordination'])
        response_readiness_mean, _ = _aggregate_episode_metric(ep_marl_metric_steps['response_readiness'])
        charger_occupancy_rate_mean, _ = _aggregate_episode_metric(ep_marl_metric_steps['charger_occupancy_rate'])
        charger_queue_delay_mean, _ = _aggregate_episode_metric(ep_marl_metric_steps['charger_queue_delay'])
        energy_not_served_mean, _ = _aggregate_episode_metric(ep_marl_metric_steps['energy_not_served'])
        safety_soc_violation_count_mean, _ = _aggregate_episode_metric(ep_marl_metric_steps['safety_soc_violation_count'])
        charge_shift_success_rate_mean, _ = _aggregate_episode_metric(ep_marl_metric_steps['charge_shift_success_rate'])

        charging_efficiency_history.append(charging_efficiency_mean)
        vehicle_utilization_history.append(vehicle_utilization_mean)
        fleet_health_history.append(fleet_health_mean)
        zone_coordination_history.append(zone_coordination_mean)
        response_readiness_history.append(response_readiness_mean)
        charger_occupancy_rate_history.append(charger_occupancy_rate_mean)
        charger_queue_delay_history.append(charger_queue_delay_mean)
        energy_not_served_history.append(energy_not_served_mean)
        safety_soc_violation_count_history.append(safety_soc_violation_count_mean)
        charge_shift_success_rate_history.append(charge_shift_success_rate_mean)

        weighted_service_rate_mean, _ = _aggregate_episode_metric(ep_improved_metric_steps['weighted_service_rate'])
        effective_service_rate_mean, _ = _aggregate_episode_metric(ep_improved_metric_steps['effective_service_rate'])
        service_quality_index_mean, _ = _aggregate_episode_metric(ep_improved_metric_steps['service_quality_index'])
        dispatch_efficiency_mean, _ = _aggregate_episode_metric(ep_improved_metric_steps['dispatch_efficiency'])
        effective_dispatch_rate_mean, _ = _aggregate_episode_metric(ep_improved_metric_steps['effective_dispatch_rate'])
        dispatch_roi_mean, _ = _aggregate_episode_metric(ep_improved_metric_steps['dispatch_roi'])

        weighted_service_rate_history.append(weighted_service_rate_mean)
        effective_service_rate_history.append(effective_service_rate_mean)
        service_quality_index_history.append(service_quality_index_mean)
        dispatch_efficiency_history.append(dispatch_efficiency_mean)
        effective_dispatch_rate_history.append(effective_dispatch_rate_mean)
        dispatch_roi_history.append(dispatch_roi_mean)

        rewards.append(ep_reward)
        raw_rewards.append(ep_reward_raw)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)

        print(
            f"{i_ep+1}/{cfg['train_eps']}, "
            f"Reward(raw/scaled): {ep_reward_raw:.2f}/{ep_reward:.2f}, "
            f"MA: {ma_rewards[-1]:.2f}, "
            f"ServiceRate(mean/last): {service_rate_mean:.4f}/{service_rate_last:.4f}, "
            f"Steps: {ep_step}"
        )

        if (i_ep + 1) % 100 == 0 and path is not None:
            agent.save_model(path + f'marl_model_ep{i_ep+1}.pth')
            print(f"模型已保存: {path}marl_model_ep{i_ep+1}.pth")

    res_dic = {
        'rewards': rewards,
        'raw_rewards': raw_rewards,
        'ma_rewards': ma_rewards,
        'critic_losses': critic_losses,
        'actor_losses': actor_losses,
        'rld_history': rld_history,
        'svc_history': svc_history,
        'srd_history': srd_history,
        'rld_last_history': rld_last_history,
        'svc_last_history': svc_last_history,
        'srd_last_history': srd_last_history,
        'service_rate_history': service_rate_history,
        'service_rate_last_history': service_rate_last_history,
        'info_missing_counts': info_missing_counts,
        'charging_efficiency_history': charging_efficiency_history,
        'vehicle_utilization_history': vehicle_utilization_history,
        'fleet_health_history': fleet_health_history,
        'zone_coordination_history': zone_coordination_history,
        'response_readiness_history': response_readiness_history,
        'charger_occupancy_rate_history': charger_occupancy_rate_history,
        'charger_queue_delay_history': charger_queue_delay_history,
        'energy_not_served_history': energy_not_served_history,
        'safety_soc_violation_count_history': safety_soc_violation_count_history,
        'charge_shift_success_rate_history': charge_shift_success_rate_history,
        'weighted_service_rate_history': weighted_service_rate_history,
        'effective_service_rate_history': effective_service_rate_history,
        'service_quality_index_history': service_quality_index_history,
        'dispatch_efficiency_history': dispatch_efficiency_history,
        'effective_dispatch_rate_history': effective_dispatch_rate_history,
        'dispatch_roi_history': dispatch_roi_history,
    }
    return res_dic


def marl_test(env, agent, cfg, path=None):
    """多智能体测试循环。"""
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
        global_state = env.reset()
        obs_n = np.array([env.get_agent_obs(agent_id) for agent_id in range(env.n_agents)])

        done = False
        info = {}
        while not done:
            actions = agent.sample_action(obs_n, add_noise=False)
            next_global_state, reward, done, info = env.step(actions)
            next_obs_n = np.array([env.get_agent_obs(agent_id) for agent_id in range(env.n_agents)])

            obs_n = next_obs_n
            global_state = next_global_state
            ep_reward += reward / reward_scale
            ep_step += 1

        rewards.append(ep_reward)
        print(f"测试回合：{i_ep+1}/{cfg['test_eps']}，奖励：{ep_reward:.2f}，步数：{ep_step}")

    if path is not None:
        np.save(path + 'marl_test_rewards.npy', rewards)

    print(f"测试完成，平均奖励：{np.mean(rewards):.2f}")
    return rewards
