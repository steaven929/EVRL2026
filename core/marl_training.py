# -*- coding: utf-8 -*-
"""
MARL Phase-1: Multi-Agent Training Loop
"""
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from core.marl_metrics import compute_all_metrics
from core.improved_metrics import compute_all_improved_metrics
from core.reward_components import REWARD_COMPONENT_KEYS
from core.evaluation_logging import EvaluationIdentity, EvaluationWriter


@dataclass(frozen=True)
class ExplorationNoise:
    schedule: str
    base_sigma: float
    service_rate_ema: float
    service_gap: float
    service_boost: float
    final_sigma: float


def compute_exploration_noise(
    episode,
    train_episodes,
    schedule,
    noise_start,
    noise_min,
    decay_power=1.0,
    service_rate_ema=None,
    service_target=0.85,
    service_boost=0.05,
    warmup_episodes=0,
):
    schedule = str(schedule).lower()
    if schedule not in {"adaptive", "fixed", "linear", "exponential"}:
        raise ValueError(f"unsupported exploration schedule: {schedule}")
    start = float(noise_start)
    minimum = float(noise_min)
    if minimum < 0 or start < minimum:
        raise ValueError("noise bounds must satisfy 0 <= noise_min <= noise_start")
    denominator = max(1, int(train_episodes) - int(warmup_episodes))
    progress = np.clip((int(episode) - int(warmup_episodes)) / denominator, 0.0, 1.0)
    if schedule == "fixed":
        base = start
    elif schedule == "linear":
        base = start - (start - minimum) * progress
    elif schedule == "exponential":
        power = max(float(decay_power), 1e-9)
        end = np.exp(-power)
        normalized = (np.exp(-power * progress) - end) / max(1.0 - end, 1e-12)
        base = minimum + (start - minimum) * normalized
    else:
        base = minimum + (start - minimum) * ((1.0 - progress) ** max(float(decay_power), 1e-9))

    ema_value = float("nan") if service_rate_ema is None else float(service_rate_ema)
    gap = 0.0
    boost = 0.0
    if schedule == "adaptive" and service_rate_ema is not None:
        target = max(float(service_target), 1e-12)
        gap = max(0.0, target - float(service_rate_ema))
        boost = float(service_boost) * gap / target
    upper = start + (float(service_boost) if schedule == "adaptive" else 0.0)
    final = float(np.clip(base + boost, minimum, upper))
    return ExplorationNoise(schedule, float(base), ema_value, float(gap), float(boost), final)


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
    exploration_base_sigma_history = []
    exploration_service_ema_history = []
    exploration_service_gap_history = []
    exploration_boost_history = []
    exploration_sigma_history = []
    service_rate_ema = None
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
    reward_component_history = {key: [] for key in REWARD_COMPONENT_KEYS}
    reward_component_by_timestep = {key: [] for key in REWARD_COMPONENT_KEYS}
    training_solver_rows = []
    training_charging_event_rows = []

    # 时间片级(0~TIME_NUM-1)指标：用于后续按26时间片分析
    time_num = int(getattr(env, 'TIME_NUM', cfg.get('TIME_NUM', 26)))
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

    reward_scale = float(cfg.get('reward_scale', 1.0))
    if reward_scale <= 0:
        reward_scale = 1.0

    if hasattr(env, 'set_experiment_variant'):
        env.set_experiment_variant(cfg.get('exp_variant', 'a1'))
    if hasattr(env, 'set_reward_mode'):
        env.set_reward_mode(cfg.get('reward_mode', 'base'))
    if hasattr(env, 'set_charging_assignment_mode'):
        env.set_charging_assignment_mode(cfg.get('charging_assignment_mode', 'p1'))
    if hasattr(env, 'set_charger_ratio'):
        env.set_charger_ratio(float(cfg.get('charger_ratio', 0.30)))

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
        ep_reward_components = {
            key: np.full(time_num, np.nan, dtype=float) for key in REWARD_COMPONENT_KEYS
        }

        noise_state = compute_exploration_noise(
            episode=i_ep,
            train_episodes=cfg['train_eps'],
            schedule=cfg.get('exploration_schedule', 'adaptive'),
            noise_start=cfg.get('marl_noise_start', 0.12),
            noise_min=cfg.get('marl_noise_min', 0.02),
            decay_power=cfg.get('marl_noise_decay_power', 2.0),
            service_rate_ema=service_rate_ema,
            service_target=cfg.get('marl_service_target', 0.85),
            service_boost=cfg.get('marl_noise_service_boost', 0.05),
            warmup_episodes=cfg.get('marl_noise_warmup_eps', 30),
        )
        exploration_base_sigma_history.append(noise_state.base_sigma)
        exploration_service_ema_history.append(noise_state.service_rate_ema)
        exploration_service_gap_history.append(noise_state.service_gap)
        exploration_boost_history.append(noise_state.service_boost)
        exploration_sigma_history.append(noise_state.final_sigma)

        while not done:
            step_idx = min(ep_step, time_num - 1)
            add_noise = True
            noise_scale = noise_state.final_sigma
            # =========2026.3.5修改，方案2：传递全局状态给sample_action==========
            actions = agent.sample_action(obs_n, global_state=global_state, add_noise=add_noise, noise_scale=noise_scale)
            # ======修改结束

            next_global_state, reward, done, info = env.step(actions)
            scaled_reward = reward / reward_scale
            if isinstance(info, dict):
                info['environment_reward'] = float(reward)
                info['scaled_training_reward'] = float(scaled_reward)
                for component_key in REWARD_COMPONENT_KEYS:
                    if component_key in info:
                        ep_reward_components[component_key][step_idx] = float(info[component_key])
                for solver in info.get('solver_stats', []):
                    training_solver_rows.append({
                        'episode': i_ep + 1, 'timestep': ep_step,
                        'demand_day': info.get('demand_day'), **dict(solver),
                    })
                for event in info.get('charging_events', []):
                    training_charging_event_rows.append({
                        'episode': i_ep + 1, 'timestep': ep_step,
                        'demand_day': info.get('demand_day'), **dict(event),
                    })

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
            if ep_step % 5 == 0 or done:
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
        ema_alpha = float(cfg.get('marl_service_ema_alpha', 0.2))
        ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        if service_rate_ema is None:
            service_rate_ema = service_rate_mean
        else:
            service_rate_ema = ema_alpha * service_rate_mean + (1.0 - ema_alpha) * service_rate_ema

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
        for component_key in REWARD_COMPONENT_KEYS:
            component_values = ep_reward_components[component_key]
            reward_component_by_timestep[component_key].append(component_values)
            finite_values = component_values[np.isfinite(component_values)]
            reward_component_history[component_key].append(
                float(np.sum(finite_values)) if finite_values.size else np.nan
            )

        rewards.append(ep_reward)
        raw_rewards.append(ep_reward_raw)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)

        # =========2026.3.3修改，简化日志输出，只保留训练轮数==========
        progress_every = max(1, int(cfg.get('progress_every', 50)))
        if (i_ep + 1) % progress_every == 0 or i_ep == 0:
            print(
                f"Episode: {i_ep+1}/{cfg['train_eps']} | "
                f"reward={ep_reward:.6f} | raw_reward={ep_reward_raw:.2f}",
                flush=True,
            )
        # ======修改结束

        if (i_ep + 1) % 100 == 0 and path is not None:
            agent.save_model(path + f'marl_model_ep{i_ep+1}.pth')
            # =========2026.3.3修改，注释掉模型保存日志==========
            # print(f"模型已保存: {path}marl_model_ep{i_ep+1}.pth")
            # ======修改结束

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
        'exploration_base_sigma_history': exploration_base_sigma_history,
        'exploration_service_ema_history': exploration_service_ema_history,
        'exploration_service_gap_history': exploration_service_gap_history,
        'exploration_boost_history': exploration_boost_history,
        'exploration_sigma_history': exploration_sigma_history,
        'info_missing_counts': info_missing_counts,
        'training_solver_rows': training_solver_rows,
        'training_charging_event_rows': training_charging_event_rows,
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
        **{f'{key}_history': np.asarray(reward_component_history[key], dtype=float)
           for key in REWARD_COMPONENT_KEYS},
        **{f'{key}_by_timestep': np.asarray(reward_component_by_timestep[key], dtype=float)
           for key in REWARD_COMPONENT_KEYS},
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

    for i_ep in range(cfg['test_eps']):
        ep_reward = 0
        ep_step = 0
        global_state = env.reset()
        obs_n = np.array([env.get_agent_obs(agent_id) for agent_id in range(env.n_agents)])

        done = False
        info = {}
        while not done:
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


def _plain_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def marl_test_days(env, agent, cfg, output_root, experiment="review-evaluation"):
    """Evaluate five explicit held-out days and persist every raw day separately."""
    test_days = [int(day) for day in cfg.get('test_day_indices', [26, 27, 28, 29, 30])]
    if len(test_days) != len(set(test_days)):
        raise ValueError("test_day_indices must contain distinct days")
    writer = EvaluationWriter(Path(output_root), experiment=experiment)
    reward_scale = max(float(cfg.get('reward_scale', 1.0)), 1e-12)
    results = {}
    if hasattr(env, 'set_training_mode'):
        env.set_training_mode(0)
    if hasattr(env, 'set_experiment_variant'):
        env.set_experiment_variant(cfg.get('exp_variant', 'a1'))
    if hasattr(env, 'set_reward_mode'):
        env.set_reward_mode(cfg.get('reward_mode', 'base'))
    if hasattr(env, 'set_charging_assignment_mode'):
        env.set_charging_assignment_mode(cfg.get('charging_assignment_mode', 'p1'))
    if hasattr(env, 'set_charger_ratio'):
        env.set_charger_ratio(float(cfg.get('charger_ratio', 0.30)))

    for day in test_days:
        if not hasattr(env, 'set_test_day'):
            raise AttributeError("environment does not support set_test_day")
        env.set_test_day(day)
        reset_value = env.reset()
        global_state = reset_value[0] if isinstance(reset_value, tuple) else reset_value
        obs_n = np.array([env.get_agent_obs(agent_id) for agent_id in range(env.n_agents)])
        timestep_rows = []
        zone_rows = []
        charging_events = []
        reward_rows = []
        solver_rows = []
        daily_reward = 0.0
        daily_scaled_reward = 0.0
        daily_served = 0.0
        daily_demand = 0.0
        daily_operating_profit = 0.0
        actor_times = []
        environment_times = []
        decision_times = []
        daily_marl_metrics = {
            key: [] for key in (
                'charging_efficiency',
                'vehicle_utilization',
                'fleet_health',
                'zone_coordination',
                'response_readiness',
                'charger_occupancy_rate',
                'charger_queue_delay',
                'energy_not_served',
                'safety_soc_violation_count',
                'charge_shift_success_rate',
            )
        }
        done = False
        timestep = 0
        while not done:
            decision_started = perf_counter()
            actor_started = perf_counter()
            actions = agent.sample_action(obs_n, global_state=global_state, add_noise=False)
            actor_elapsed = perf_counter() - actor_started
            environment_started = perf_counter()
            next_state, reward, done, info = env.step(actions)
            environment_elapsed = perf_counter() - environment_started
            decision_elapsed = perf_counter() - decision_started
            info = dict(info)
            info['environment_reward'] = float(reward)
            info['scaled_training_reward'] = float(reward) / reward_scale
            step_marl_metrics = compute_all_metrics(env, info)
            info.update(step_marl_metrics)
            for key in daily_marl_metrics:
                daily_marl_metrics[key].append(float(step_marl_metrics.get(key, np.nan)))
            actor_times.append(actor_elapsed)
            environment_times.append(environment_elapsed)
            decision_times.append(decision_elapsed)
            obs_n = np.array([env.get_agent_obs(agent_id) for agent_id in range(env.n_agents)])
            global_state = next_state
            daily_reward += float(reward)
            daily_scaled_reward += float(reward) / reward_scale
            daily_served += float(info.get('served_orders', 0.0))
            daily_demand += float(info.get('total_demand', 0.0))
            daily_operating_profit += float(info.get('operating_profit', 0.0))

            step_row = {
                'test_day': day,
                'timestep': timestep,
                'reward': float(reward),
                'actor_inference_time_seconds': actor_elapsed,
                'environment_step_time_seconds': environment_elapsed,
                'total_decision_time_seconds': decision_elapsed,
            }
            for key, value in info.items():
                scalar = _plain_scalar(value)
                if scalar is not None:
                    step_row[key] = scalar
            timestep_rows.append(step_row)

            demand = np.asarray(info.get('demand_matrix', []), dtype=float)
            served = np.asarray(info.get('demand_orders', []), dtype=float)
            charging_by_zone = info.get('charging_actions', {})
            parking_available = np.asarray(info.get('parking_available_by_zone', []), dtype=float)
            parking_used = np.asarray(info.get('parking_used_by_zone', []), dtype=float)
            charger_available = np.asarray(info.get('charger_available_by_zone', []), dtype=float)
            charger_used = np.asarray(info.get('charger_used_by_zone', []), dtype=float)
            zone_count = int(getattr(env, 'n_agents', demand.shape[0] if demand.ndim else 0))
            for zone in range(zone_count):
                zone_demand = float(np.sum(demand[zone])) if demand.ndim == 2 and zone < demand.shape[0] else 0.0
                zone_served = float(np.sum(served[zone])) if served.ndim == 2 and zone < served.shape[0] else 0.0
                zone_rows.append({
                    'test_day': day,
                    'timestep': timestep,
                    'zone': zone,
                    'demand': zone_demand,
                    'served': zone_served,
                    'service_rate': zone_served / zone_demand if zone_demand > 0 else 1.0,
                    'charging_assignments': int(charging_by_zone.get(zone, 0)),
                    'parking_available': float(parking_available[zone]) if zone < parking_available.size else np.nan,
                    'parking_used': float(parking_used[zone]) if zone < parking_used.size else np.nan,
                    'charger_available': float(charger_available[zone]) if zone < charger_available.size else np.nan,
                    'charger_used': float(charger_used[zone]) if zone < charger_used.size else np.nan,
                })
            for event in info.get('charging_events', []):
                charging_events.append({'test_day': day, 'timestep': timestep, **dict(event)})
            reward_rows.append({
                'test_day': day,
                'timestep': timestep,
                **{key: info.get(key, np.nan) for key in REWARD_COMPONENT_KEYS},
                'legacy_NOR': info.get('legacy_NOR', info.get('NOR', np.nan)),
                'operating_profit': info.get('operating_profit', np.nan),
            })
            for solver in info.get('solver_stats', []):
                solver_rows.append({'test_day': day, 'timestep': timestep, **dict(solver)})
            timestep += 1

        summary = {
            'environment_reward': daily_reward,
            'scaled_training_reward': daily_scaled_reward,
            'served_orders': daily_served,
            'total_demand': daily_demand,
            'service_rate': daily_served / daily_demand if daily_demand > 0 else 1.0,
            'operating_profit': daily_operating_profit,
            'timesteps': timestep,
            'actor_inference_time_mean_seconds': float(np.mean(actor_times)),
            'actor_inference_time_p95_seconds': float(np.percentile(actor_times, 95)),
            'total_decision_time_mean_seconds': float(np.mean(decision_times)),
            'total_decision_time_p50_seconds': float(np.percentile(decision_times, 50)),
            'total_decision_time_p95_seconds': float(np.percentile(decision_times, 95)),
            'total_decision_time_max_seconds': float(np.max(decision_times)),
            'environment_step_time_mean_seconds': float(np.mean(environment_times)),
        }
        for key, values in daily_marl_metrics.items():
            array = np.asarray(values, dtype=float)
            finite = array[np.isfinite(array)]
            summary[key] = float(np.mean(finite)) if finite.size else np.nan
        energy_not_served_values = np.asarray(
            daily_marl_metrics['energy_not_served'], dtype=float
        )
        soc_violation_values = np.asarray(
            daily_marl_metrics['safety_soc_violation_count'], dtype=float
        )
        summary['energy_not_served_total'] = float(np.nansum(energy_not_served_values))
        summary['safety_soc_violation_count_total'] = float(np.nansum(soc_violation_values))
        daily_component_totals = {}
        for key in REWARD_COMPONENT_KEYS:
            values = np.asarray([row.get(key, np.nan) for row in reward_rows], dtype=float)
            finite = values[np.isfinite(values)]
            daily_component_totals[key] = float(np.sum(finite)) if finite.size else np.nan
        summary.update(daily_component_totals)
        expected_pft = (
            daily_component_totals['NOR']
            - daily_component_totals['P_soc']
            - daily_component_totals['P_svr']
            + daily_component_totals['B_chg']
            + daily_component_totals['B_soc']
        )
        summary['reward_minus_PFT'] = (
            daily_component_totals['environment_reward'] - daily_component_totals['PFT']
        )
        summary['pft_identity_error'] = daily_component_totals['PFT'] - expected_pft
        summary['reward_scaling_error'] = (
            daily_component_totals['scaled_training_reward']
            - daily_component_totals['environment_reward'] / reward_scale
        )
        identity = EvaluationIdentity(
            method=str(cfg.get('method', 'hmarl-rc')),
            fleet_size=int(getattr(env, 'CAR_NUM', cfg.get('fleet_size', 200))),
            seed=int(cfg.get('seed', 0)),
            charger_ratio=float(cfg.get('charger_ratio', 0.30)),
            test_day=day,
        )
        writer.write_day(
            identity=identity,
            config=dict(cfg),
            summary=summary,
            timestep_rows=timestep_rows,
            zone_timestep_rows=zone_rows,
            charging_event_rows=charging_events,
            reward_rows=reward_rows,
            solver_rows=solver_rows,
        )
        results[day] = summary
    return results

