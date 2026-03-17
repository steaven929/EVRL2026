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
    # =========2026.3.3修改，注释掉详细信息，只保留关键日志==========
    # print("开始MARL训练...")
    # print(f"智能体数量: {env.n_agents}")
    # print("局部观测维度: 7 (含agent_id)")
    # print("动作维度: 3 (每个智能体)")
    # print(f"全局状态维度: {env.observation_space.shape[0]}")
    # ======修改结束

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

    # 时间片级(0~TIME_NUM-1)指标：用于后续按26时间片分析
    env_time_num = int(getattr(env, 'TIME_NUM', cfg.get('TIME_NUM', 26)))
    if env_time_num <= 0:
        env_time_num = 26
    max_steps = int(cfg.get('max_steps', 0))
    if max_steps <= 0:
        max_steps = env_time_num
    time_num = int(cfg.get('log_time_num', env_time_num))
    if time_num <= 0:
        time_num = env_time_num
    time_num = min(time_num, max_steps)

    eval_every = int(cfg.get('eval_every', 0))
    eval_eps = max(1, int(cfg.get('eval_eps', 1)))
    marl_metric_sample_every = int(cfg.get('marl_metric_sample_every', 1))
    if marl_metric_sample_every <= 0:
        marl_metric_sample_every = 1

    marl_noise_start = float(cfg.get('marl_noise_start', 0.35))
    marl_noise_min = float(cfg.get('marl_noise_min', 0.12))
    marl_noise_decay_power = float(cfg.get('marl_noise_decay_power', 1.4))
    marl_noise_warmup_eps = int(cfg.get('marl_noise_warmup_eps', 50))
    marl_service_target = float(cfg.get('marl_service_target', 0.85))
    marl_noise_service_boost = float(cfg.get('marl_noise_service_boost', 0.10))

    marl_noise_start = max(0.0, marl_noise_start)
    marl_noise_min = max(0.0, min(marl_noise_min, marl_noise_start if marl_noise_start > 0 else marl_noise_min))
    marl_noise_decay_power = max(0.1, marl_noise_decay_power)
    marl_noise_warmup_eps = max(0, marl_noise_warmup_eps)
    marl_service_target = max(0.0, min(1.0, marl_service_target))
    marl_noise_service_boost = max(0.0, marl_noise_service_boost)
    zone_num = int(getattr(env, 'n_agents', cfg.get('n_agents', 50)))
    svc_by_timestep = []
    rld_by_timestep = []
    srd_by_timestep = []
    reward_by_timestep = []
    total_demand_by_timestep = []
    served_orders_by_timestep = []
    charging_efficiency_by_timestep = []
    charger_queue_delay_by_timestep = []
    energy_not_served_by_timestep = []
    soc_violation_by_timestep = []

    # 时间片×区域级指标（shape: [train_eps, time_num, zone_num]）
    demand_by_timestep_zone = []
    served_by_timestep_zone = []
    service_rate_by_timestep_zone = []
    charging_actions_by_timestep_zone = []
    dispatch_by_timestep_zone = []
    parking_available_by_timestep_zone = []
    parking_used_by_timestep_zone = []
    charger_available_by_timestep_zone = []
    charger_used_by_timestep_zone = []
    eval_rewards = []
    eval_episodes = []

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

        # 每个episode内的时间片序列（固定长度time_num，不足补nan）
        ep_svc_by_t = np.full(time_num, np.nan, dtype=float)
        ep_rld_by_t = np.full(time_num, np.nan, dtype=float)
        ep_srd_by_t = np.full(time_num, np.nan, dtype=float)
        ep_reward_by_t = np.full(time_num, np.nan, dtype=float)
        ep_total_demand_by_t = np.full(time_num, np.nan, dtype=float)
        ep_served_orders_by_t = np.full(time_num, np.nan, dtype=float)
        ep_ce_by_t = np.full(time_num, np.nan, dtype=float)
        ep_qd_by_t = np.full(time_num, np.nan, dtype=float)
        ep_ens_by_t = np.full(time_num, np.nan, dtype=float)
        ep_socv_by_t = np.full(time_num, np.nan, dtype=float)

        # 区域级序列
        ep_demand_by_tz = np.full((time_num, zone_num), np.nan, dtype=float)
        ep_served_by_tz = np.full((time_num, zone_num), np.nan, dtype=float)
        ep_service_rate_by_tz = np.full((time_num, zone_num), np.nan, dtype=float)
        ep_charging_by_tz = np.full((time_num, zone_num), np.nan, dtype=float)
        ep_dispatch_by_tz = np.full((time_num, zone_num), np.nan, dtype=float)
        ep_parking_available_by_tz = np.full((time_num, zone_num), np.nan, dtype=float)
        ep_parking_used_by_tz = np.full((time_num, zone_num), np.nan, dtype=float)
        ep_charger_available_by_tz = np.full((time_num, zone_num), np.nan, dtype=float)
        ep_charger_used_by_tz = np.full((time_num, zone_num), np.nan, dtype=float)

        # 自适应探索：先高噪声探索，再按训练进度和上一轮服务率退火
        if cfg['train_eps'] > marl_noise_warmup_eps:
            progress = max(0.0, min(1.0, (i_ep - marl_noise_warmup_eps) / max(1, cfg['train_eps'] - marl_noise_warmup_eps)))
        else:
            progress = 1.0
        base_noise = marl_noise_min + (marl_noise_start - marl_noise_min) * ((1.0 - progress) ** marl_noise_decay_power)
        last_svc = float(svc_last_history[-1]) if len(svc_last_history) > 0 else marl_service_target
        svc_gap = max(0.0, marl_service_target - last_svc)
        noise_scale_ep = base_noise + marl_noise_service_boost * svc_gap
        noise_scale_ep = max(marl_noise_min, min(marl_noise_start + marl_noise_service_boost, noise_scale_ep))

        while (not done) and (ep_step < max_steps):
            step_idx = min(ep_step, time_num - 1)
            add_noise = True
            noise_scale = noise_scale_ep
            # =========2026.3.5修改，方案2：传递全局状态给sample_action==========
            actions = agent.sample_action(obs_n, global_state=global_state, add_noise=add_noise, noise_scale=noise_scale)
            # ======修改结束

            next_global_state, reward, done, info = env.step(actions)
            scaled_reward = reward / reward_scale

            # 时间片级原始记录（每一步都记，后续可直接做26时间片分析）
            ep_reward_by_t[step_idx] = float(scaled_reward)
            if isinstance(info, dict):
                if 'svc' in info:
                    ep_svc_by_t[step_idx] = float(info['svc'])
                if 'rld' in info:
                    ep_rld_by_t[step_idx] = float(info['rld'])
                if 'srd' in info:
                    ep_srd_by_t[step_idx] = float(info['srd'])
                if 'total_demand' in info:
                    ep_total_demand_by_t[step_idx] = float(info['total_demand'])
                if 'served_orders' in info:
                    ep_served_orders_by_t[step_idx] = float(info['served_orders'])

                # 区域级：需求与满足（按origin区域汇总）
                if 'demand_matrix' in info:
                    dm = np.asarray(info['demand_matrix'], dtype=float)
                    if dm.ndim == 2:
                        z = min(zone_num, dm.shape[0])
                        ep_demand_by_tz[step_idx, :z] = np.sum(dm[:z, :], axis=1)
                if 'demand_orders' in info:
                    sm = np.asarray(info['demand_orders'], dtype=float)
                    if sm.ndim == 2:
                        z = min(zone_num, sm.shape[0])
                        ep_served_by_tz[step_idx, :z] = np.sum(sm[:z, :], axis=1)
                # 区域级服务率（无需求记为nan，避免把0需求误当1）
                if np.any(np.isfinite(ep_demand_by_tz[step_idx])) and np.any(np.isfinite(ep_served_by_tz[step_idx])):
                    dem = ep_demand_by_tz[step_idx]
                    srv = ep_served_by_tz[step_idx]
                    rate = np.full(zone_num, np.nan, dtype=float)
                    valid = np.isfinite(dem) & np.isfinite(srv) & (dem > 0)
                    rate[valid] = srv[valid] / dem[valid]
                    ep_service_rate_by_tz[step_idx] = rate

                # 区域级充电执行（zone_id: count）
                if 'charging_actions' in info and isinstance(info['charging_actions'], dict):
                    carr = np.zeros(zone_num, dtype=float)
                    for k, v in info['charging_actions'].items():
                        try:
                            zi = int(k)
                            if 0 <= zi < zone_num:
                                carr[zi] = float(v)
                        except Exception:
                            continue
                    ep_charging_by_tz[step_idx] = carr

                # 区域级调度执行（按origin行和）
                if 'action_table_true' in info:
                    am = np.asarray(info['action_table_true'], dtype=float)
                    if am.ndim == 2:
                        z = min(zone_num, am.shape[0])
                        ep_dispatch_by_tz[step_idx, :z] = np.sum(am[:z, :], axis=1)

                # 区域级停车/充电占用
                if 'parking_available_by_zone' in info:
                    pa = np.asarray(info['parking_available_by_zone'], dtype=float)
                    z = min(zone_num, pa.shape[0])
                    ep_parking_available_by_tz[step_idx, :z] = pa[:z]
                if 'parking_used_by_zone' in info:
                    pu = np.asarray(info['parking_used_by_zone'], dtype=float)
                    z = min(zone_num, pu.shape[0])
                    ep_parking_used_by_tz[step_idx, :z] = pu[:z]
                if 'charger_available_by_zone' in info:
                    ca = np.asarray(info['charger_available_by_zone'], dtype=float)
                    z = min(zone_num, ca.shape[0])
                    ep_charger_available_by_tz[step_idx, :z] = ca[:z]
                if 'charger_used_by_zone' in info:
                    cu = np.asarray(info['charger_used_by_zone'], dtype=float)
                    z = min(zone_num, cu.shape[0])
                    ep_charger_used_by_tz[step_idx, :z] = cu[:z]

            # 充电机制按步记录（不降采样）
            step_marl_metrics = compute_all_metrics(env, info)
            ep_ce_by_t[step_idx] = float(step_marl_metrics.get('charging_efficiency', np.nan))
            ep_qd_by_t[step_idx] = float(step_marl_metrics.get('charger_queue_delay', np.nan))
            ep_ens_by_t[step_idx] = float(step_marl_metrics.get('energy_not_served', np.nan))
            ep_socv_by_t[step_idx] = float(step_marl_metrics.get('safety_soc_violation_count', np.nan))

            # =========2026.3.3修改，减少指标计算频率以加速训练==========
            # 每5步计算一次指标（而非每步都计算）
            if ep_step % marl_metric_sample_every == 0 or done:
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
            # ======修改结束

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

        # 保存该episode的时间片序列
        svc_by_timestep.append(ep_svc_by_t)
        rld_by_timestep.append(ep_rld_by_t)
        srd_by_timestep.append(ep_srd_by_t)
        reward_by_timestep.append(ep_reward_by_t)
        total_demand_by_timestep.append(ep_total_demand_by_t)
        served_orders_by_timestep.append(ep_served_orders_by_t)
        charging_efficiency_by_timestep.append(ep_ce_by_t)
        charger_queue_delay_by_timestep.append(ep_qd_by_t)
        energy_not_served_by_timestep.append(ep_ens_by_t)
        soc_violation_by_timestep.append(ep_socv_by_t)

        # 保存该episode的区域级序列
        demand_by_timestep_zone.append(ep_demand_by_tz)
        served_by_timestep_zone.append(ep_served_by_tz)
        service_rate_by_timestep_zone.append(ep_service_rate_by_tz)
        charging_actions_by_timestep_zone.append(ep_charging_by_tz)
        dispatch_by_timestep_zone.append(ep_dispatch_by_tz)
        parking_available_by_timestep_zone.append(ep_parking_available_by_tz)
        parking_used_by_timestep_zone.append(ep_parking_used_by_tz)
        charger_available_by_timestep_zone.append(ep_charger_available_by_tz)
        charger_used_by_timestep_zone.append(ep_charger_used_by_tz)

        rewards.append(ep_reward)
        raw_rewards.append(ep_reward_raw)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)

        # =========2026.3.3修改，简化日志输出，只保留训练轮数==========
        if (i_ep + 1) % 50 == 0 or i_ep == 0:  # 每50轮打印一次
            print(f"Episode: {i_ep+1}/{cfg['train_eps']}")
        # ======修改结束

        if (i_ep + 1) % 100 == 0 and path is not None:
            agent.save_model(path + f'marl_model_ep{i_ep+1}.pth')
            # =========2026.3.3修改，注释掉模型保存日志==========
            # print(f"模型已保存: {path}marl_model_ep{i_ep+1}.pth")
            # ======修改结束

        if eval_every > 0 and (i_ep + 1) % eval_every == 0:
            eval_mean = _marl_evaluate_policy(env, agent, cfg, eval_eps, max_steps)
            eval_rewards.append(float(eval_mean))
            eval_episodes.append(int(i_ep + 1))
            print(f"  [MARL Eval] Episode {i_ep + 1}: mean reward={eval_mean:.2f} over {eval_eps} eps")

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
        # 时间片级原始序列（shape: [train_eps, time_num]）
        'svc_by_timestep': np.asarray(svc_by_timestep, dtype=float),
        'rld_by_timestep': np.asarray(rld_by_timestep, dtype=float),
        'srd_by_timestep': np.asarray(srd_by_timestep, dtype=float),
        'reward_by_timestep': np.asarray(reward_by_timestep, dtype=float),
        'total_demand_by_timestep': np.asarray(total_demand_by_timestep, dtype=float),
        'served_orders_by_timestep': np.asarray(served_orders_by_timestep, dtype=float),
        'charging_efficiency_by_timestep': np.asarray(charging_efficiency_by_timestep, dtype=float),
        'charger_queue_delay_by_timestep': np.asarray(charger_queue_delay_by_timestep, dtype=float),
        'energy_not_served_by_timestep': np.asarray(energy_not_served_by_timestep, dtype=float),
        'soc_violation_by_timestep': np.asarray(soc_violation_by_timestep, dtype=float),
        # 时间片均值曲线（跨episode）
        'svc_timestep_mean': np.nanmean(np.asarray(svc_by_timestep, dtype=float), axis=0),
        'rld_timestep_mean': np.nanmean(np.asarray(rld_by_timestep, dtype=float), axis=0),
        'srd_timestep_mean': np.nanmean(np.asarray(srd_by_timestep, dtype=float), axis=0),
        'reward_timestep_mean': np.nanmean(np.asarray(reward_by_timestep, dtype=float), axis=0),
        'total_demand_timestep_mean': np.nanmean(np.asarray(total_demand_by_timestep, dtype=float), axis=0),
        'served_orders_timestep_mean': np.nanmean(np.asarray(served_orders_by_timestep, dtype=float), axis=0),
        'charging_efficiency_timestep_mean': np.nanmean(np.asarray(charging_efficiency_by_timestep, dtype=float), axis=0),
        'charger_queue_delay_timestep_mean': np.nanmean(np.asarray(charger_queue_delay_by_timestep, dtype=float), axis=0),
        'energy_not_served_timestep_mean': np.nanmean(np.asarray(energy_not_served_by_timestep, dtype=float), axis=0),
        'soc_violation_timestep_mean': np.nanmean(np.asarray(soc_violation_by_timestep, dtype=float), axis=0),
        # 区域级矩阵与索引
        'zone_ids': np.arange(zone_num, dtype=int),
        'timestep_ids': np.arange(time_num, dtype=int),
        'demand_by_timestep_zone': np.asarray(demand_by_timestep_zone, dtype=float),
        'served_by_timestep_zone': np.asarray(served_by_timestep_zone, dtype=float),
        'service_rate_by_timestep_zone': np.asarray(service_rate_by_timestep_zone, dtype=float),
        'charging_actions_by_timestep_zone': np.asarray(charging_actions_by_timestep_zone, dtype=float),
        'dispatch_by_timestep_zone': np.asarray(dispatch_by_timestep_zone, dtype=float),
        'demand_timestep_zone_mean': np.nanmean(np.asarray(demand_by_timestep_zone, dtype=float), axis=0),
        'served_timestep_zone_mean': np.nanmean(np.asarray(served_by_timestep_zone, dtype=float), axis=0),
        'service_rate_timestep_zone_mean': np.nanmean(np.asarray(service_rate_by_timestep_zone, dtype=float), axis=0),
        'charging_actions_timestep_zone_mean': np.nanmean(np.asarray(charging_actions_by_timestep_zone, dtype=float), axis=0),
        'dispatch_timestep_zone_mean': np.nanmean(np.asarray(dispatch_by_timestep_zone, dtype=float), axis=0),
        'parking_available_by_timestep_zone': np.asarray(parking_available_by_timestep_zone, dtype=float),
        'parking_used_by_timestep_zone': np.asarray(parking_used_by_timestep_zone, dtype=float),
        'charger_available_by_timestep_zone': np.asarray(charger_available_by_timestep_zone, dtype=float),
        'charger_used_by_timestep_zone': np.asarray(charger_used_by_timestep_zone, dtype=float),
        'parking_available_timestep_zone_mean': np.nanmean(np.asarray(parking_available_by_timestep_zone, dtype=float), axis=0),
        'parking_used_timestep_zone_mean': np.nanmean(np.asarray(parking_used_by_timestep_zone, dtype=float), axis=0),
        'charger_available_timestep_zone_mean': np.nanmean(np.asarray(charger_available_by_timestep_zone, dtype=float), axis=0),
        'charger_used_timestep_zone_mean': np.nanmean(np.asarray(charger_used_by_timestep_zone, dtype=float), axis=0),
        'eval_rewards': np.asarray(eval_rewards, dtype=float),
        'eval_episodes': np.asarray(eval_episodes, dtype=int),
    }
    return res_dic


def marl_test(env, agent, cfg, path=None):
    """多智能体测试循环。"""
    # =========2026.3.3修改，注释掉测试开始日志==========
    # print("开始MARL测试...")
    # ======修改结束

    rewards = []
    reward_scale = float(cfg.get('reward_scale', 1.0))
    if reward_scale <= 0:
        reward_scale = 1.0

    if hasattr(env, 'set_experiment_variant'):
        env.set_experiment_variant(cfg.get('exp_variant', 'a1'))
    if hasattr(env, 'set_reward_mode'):
        env.set_reward_mode(cfg.get('reward_mode', 'base'))

    env_time_num = int(getattr(env, 'TIME_NUM', cfg.get('TIME_NUM', 26)))
    if env_time_num <= 0:
        env_time_num = 26
    max_steps = int(cfg.get('max_steps', 0))
    if max_steps <= 0:
        max_steps = env_time_num

    for i_ep in range(cfg['test_eps']):
        ep_reward = 0
        ep_step = 0
        global_state = env.reset()
        obs_n = np.array([env.get_agent_obs(agent_id) for agent_id in range(env.n_agents)])

        done = False
        info = {}
        while (not done) and (ep_step < max_steps):
            # =========2026.3.5修改，方案2：传递全局状态给sample_action==========
            actions = agent.sample_action(obs_n, global_state=global_state, add_noise=False)
            # ======修改结束
            next_global_state, reward, done, info = env.step(actions)
            next_obs_n = np.array([env.get_agent_obs(agent_id) for agent_id in range(env.n_agents)])

            obs_n = next_obs_n
            global_state = next_global_state
            ep_reward += reward / reward_scale
            ep_step += 1

        rewards.append(ep_reward)
        # =========2026.3.3修改，注释掉测试回合日志==========
        # print(f"测试回合：{i_ep+1}/{cfg['test_eps']}，奖励：{ep_reward:.2f}，步数：{ep_step}")
        # ======修改结束

    if path is not None:
        np.save(path + 'marl_test_rewards.npy', rewards)

    # =========2026.3.3修改，注释掉测试完成日志==========
    # print(f"测试完成，平均奖励：{np.mean(rewards):.2f}")
    # ======修改结束
    return rewards


def _marl_evaluate_policy(env, agent, cfg, num_episodes, max_steps):
    rewards = []
    reward_scale = float(cfg.get('reward_scale', 1.0))
    if reward_scale <= 0:
        reward_scale = 1.0

    for _ in range(num_episodes):
        global_state = env.reset()
        obs_n = np.array([env.get_agent_obs(agent_id) for agent_id in range(env.n_agents)])
        ep_reward = 0.0
        ep_step = 0
        done = False
        while (not done) and (ep_step < max_steps):
            actions = agent.sample_action(obs_n, global_state=global_state, add_noise=False)
            next_global_state, reward, done, _ = env.step(actions)
            next_obs_n = np.array([env.get_agent_obs(agent_id) for agent_id in range(env.n_agents)])
            obs_n = next_obs_n
            global_state = next_global_state
            ep_reward += reward / reward_scale
            ep_step += 1
        rewards.append(ep_reward)

    return float(np.mean(rewards)) if rewards else 0.0
