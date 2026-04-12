# -*- coding: utf-8 -*-
"""
MARL training and evaluation loops.

This file intentionally uses the earlier single-critic training schedule while
keeping the newer metric exports so 1000-car rollback runs remain comparable to
recent experiments.
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
    info_missing_counts = {"rld": 0, "svc": 0, "srd": 0, "service_rate": 0}

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

    time_num = int(getattr(env, "TIME_NUM", cfg.get("TIME_NUM", 26)))
    zone_num = int(getattr(env, "n_agents", cfg.get("n_agents", 50)))
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

    demand_by_timestep_zone = []
    served_by_timestep_zone = []
    service_rate_by_timestep_zone = []
    charging_actions_by_timestep_zone = []
    dispatch_by_timestep_zone = []
    parking_available_by_timestep_zone = []
    parking_used_by_timestep_zone = []
    charger_available_by_timestep_zone = []
    charger_used_by_timestep_zone = []

    reward_scale = float(cfg.get("reward_scale", 1.0))
    if reward_scale <= 0:
        reward_scale = 1.0

    if hasattr(env, "set_experiment_variant"):
        env.set_experiment_variant(cfg.get("exp_variant", "a1"))
    if hasattr(env, "set_reward_mode"):
        env.set_reward_mode(cfg.get("reward_mode", "base"))

    for i_ep in range(cfg["train_eps"]):
        ep_reward = 0.0
        ep_reward_raw = 0.0
        ep_step = 0

        if hasattr(env, "set_epoch"):
            env.set_epoch(i_ep)

        global_state = env.reset()
        obs_n = np.array([env.get_agent_obs(agent_id) for agent_id in range(env.n_agents)])

        done = False
        info = {}
        ep_metric_steps = {"rld": [], "svc": [], "srd": [], "service_rate": []}
        ep_marl_metric_steps = {
            "charging_efficiency": [],
            "vehicle_utilization": [],
            "fleet_health": [],
            "zone_coordination": [],
            "response_readiness": [],
            "charger_occupancy_rate": [],
            "charger_queue_delay": [],
            "energy_not_served": [],
            "safety_soc_violation_count": [],
            "charge_shift_success_rate": [],
        }
        ep_improved_metric_steps = {
            "weighted_service_rate": [],
            "effective_service_rate": [],
            "service_quality_index": [],
            "dispatch_efficiency": [],
            "effective_dispatch_rate": [],
            "dispatch_roi": [],
        }

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

        ep_demand_by_tz = np.full((time_num, zone_num), np.nan, dtype=float)
        ep_served_by_tz = np.full((time_num, zone_num), np.nan, dtype=float)
        ep_service_rate_by_tz = np.full((time_num, zone_num), np.nan, dtype=float)
        ep_charging_by_tz = np.full((time_num, zone_num), np.nan, dtype=float)
        ep_dispatch_by_tz = np.full((time_num, zone_num), np.nan, dtype=float)
        ep_parking_available_by_tz = np.full((time_num, zone_num), np.nan, dtype=float)
        ep_parking_used_by_tz = np.full((time_num, zone_num), np.nan, dtype=float)
        ep_charger_available_by_tz = np.full((time_num, zone_num), np.nan, dtype=float)
        ep_charger_used_by_tz = np.full((time_num, zone_num), np.nan, dtype=float)

        max_steps = int(cfg.get("max_steps", 0))
        while (not done) and (max_steps <= 0 or ep_step < max_steps):
            step_idx = min(ep_step, time_num - 1)
            noise_scale = max(0.1, 0.3 * (1 - i_ep / cfg["train_eps"]))
            actions = agent.sample_action(
                obs_n,
                global_state=global_state,
                add_noise=True,
                noise_scale=noise_scale,
            )

            next_global_state, reward, done, info = env.step(actions)
            scaled_reward = reward / reward_scale

            ep_reward_by_t[step_idx] = float(scaled_reward)
            if isinstance(info, dict):
                if "svc" in info:
                    ep_svc_by_t[step_idx] = float(info["svc"])
                if "rld" in info:
                    ep_rld_by_t[step_idx] = float(info["rld"])
                if "srd" in info:
                    ep_srd_by_t[step_idx] = float(info["srd"])
                if "total_demand" in info:
                    ep_total_demand_by_t[step_idx] = float(info["total_demand"])
                if "served_orders" in info:
                    ep_served_orders_by_t[step_idx] = float(info["served_orders"])

                if "demand_matrix" in info:
                    dm = np.asarray(info["demand_matrix"], dtype=float)
                    if dm.ndim == 2:
                        z = min(zone_num, dm.shape[0])
                        ep_demand_by_tz[step_idx, :z] = np.sum(dm[:z, :], axis=1)
                if "demand_orders" in info:
                    sm = np.asarray(info["demand_orders"], dtype=float)
                    if sm.ndim == 2:
                        z = min(zone_num, sm.shape[0])
                        ep_served_by_tz[step_idx, :z] = np.sum(sm[:z, :], axis=1)
                if np.any(np.isfinite(ep_demand_by_tz[step_idx])) and np.any(
                    np.isfinite(ep_served_by_tz[step_idx])
                ):
                    dem = ep_demand_by_tz[step_idx]
                    srv = ep_served_by_tz[step_idx]
                    rate = np.full(zone_num, np.nan, dtype=float)
                    valid = np.isfinite(dem) & np.isfinite(srv) & (dem > 0)
                    rate[valid] = srv[valid] / dem[valid]
                    ep_service_rate_by_tz[step_idx] = rate

                if "charging_actions" in info and isinstance(info["charging_actions"], dict):
                    carr = np.zeros(zone_num, dtype=float)
                    for key, value in info["charging_actions"].items():
                        try:
                            zone_idx = int(key)
                        except Exception:
                            continue
                        if 0 <= zone_idx < zone_num:
                            carr[zone_idx] = float(value)
                    ep_charging_by_tz[step_idx] = carr

                if "action_table_true" in info:
                    am = np.asarray(info["action_table_true"], dtype=float)
                    if am.ndim == 2:
                        z = min(zone_num, am.shape[0])
                        ep_dispatch_by_tz[step_idx, :z] = np.sum(am[:z, :], axis=1)

                if "parking_available_by_zone" in info:
                    pa = np.asarray(info["parking_available_by_zone"], dtype=float)
                    z = min(zone_num, pa.shape[0])
                    ep_parking_available_by_tz[step_idx, :z] = pa[:z]
                if "parking_used_by_zone" in info:
                    pu = np.asarray(info["parking_used_by_zone"], dtype=float)
                    z = min(zone_num, pu.shape[0])
                    ep_parking_used_by_tz[step_idx, :z] = pu[:z]
                if "charger_available_by_zone" in info:
                    ca = np.asarray(info["charger_available_by_zone"], dtype=float)
                    z = min(zone_num, ca.shape[0])
                    ep_charger_available_by_tz[step_idx, :z] = ca[:z]
                if "charger_used_by_zone" in info:
                    cu = np.asarray(info["charger_used_by_zone"], dtype=float)
                    z = min(zone_num, cu.shape[0])
                    ep_charger_used_by_tz[step_idx, :z] = cu[:z]

            step_marl_metrics = compute_all_metrics(env, info)
            ep_ce_by_t[step_idx] = float(step_marl_metrics.get("charging_efficiency", np.nan))
            ep_qd_by_t[step_idx] = float(step_marl_metrics.get("charger_queue_delay", np.nan))
            ep_ens_by_t[step_idx] = float(step_marl_metrics.get("energy_not_served", np.nan))
            ep_socv_by_t[step_idx] = float(step_marl_metrics.get("safety_soc_violation_count", np.nan))

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

            if len(agent.memory) >= cfg["batch_size"]:
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

        rld_mean, rld_last = _aggregate_episode_metric(ep_metric_steps["rld"])
        svc_mean, svc_last = _aggregate_episode_metric(ep_metric_steps["svc"])
        srd_mean, srd_last = _aggregate_episode_metric(ep_metric_steps["srd"])
        service_rate_mean, service_rate_last = _aggregate_episode_metric(ep_metric_steps["service_rate"])

        rld_history.append(rld_mean)
        svc_history.append(svc_mean)
        srd_history.append(srd_mean)
        rld_last_history.append(rld_last)
        svc_last_history.append(svc_last)
        srd_last_history.append(srd_last)
        service_rate_history.append(service_rate_mean)
        service_rate_last_history.append(service_rate_last)

        charging_efficiency_mean, _ = _aggregate_episode_metric(ep_marl_metric_steps["charging_efficiency"])
        vehicle_utilization_mean, _ = _aggregate_episode_metric(ep_marl_metric_steps["vehicle_utilization"])
        fleet_health_mean, _ = _aggregate_episode_metric(ep_marl_metric_steps["fleet_health"])
        zone_coordination_mean, _ = _aggregate_episode_metric(ep_marl_metric_steps["zone_coordination"])
        response_readiness_mean, _ = _aggregate_episode_metric(ep_marl_metric_steps["response_readiness"])
        charger_occupancy_rate_mean, _ = _aggregate_episode_metric(ep_marl_metric_steps["charger_occupancy_rate"])
        charger_queue_delay_mean, _ = _aggregate_episode_metric(ep_marl_metric_steps["charger_queue_delay"])
        energy_not_served_mean, _ = _aggregate_episode_metric(ep_marl_metric_steps["energy_not_served"])
        safety_soc_violation_count_mean, _ = _aggregate_episode_metric(
            ep_marl_metric_steps["safety_soc_violation_count"]
        )
        charge_shift_success_rate_mean, _ = _aggregate_episode_metric(
            ep_marl_metric_steps["charge_shift_success_rate"]
        )

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

        weighted_service_rate_mean, _ = _aggregate_episode_metric(ep_improved_metric_steps["weighted_service_rate"])
        effective_service_rate_mean, _ = _aggregate_episode_metric(
            ep_improved_metric_steps["effective_service_rate"]
        )
        service_quality_index_mean, _ = _aggregate_episode_metric(
            ep_improved_metric_steps["service_quality_index"]
        )
        dispatch_efficiency_mean, _ = _aggregate_episode_metric(
            ep_improved_metric_steps["dispatch_efficiency"]
        )
        effective_dispatch_rate_mean, _ = _aggregate_episode_metric(
            ep_improved_metric_steps["effective_dispatch_rate"]
        )
        dispatch_roi_mean, _ = _aggregate_episode_metric(ep_improved_metric_steps["dispatch_roi"])

        weighted_service_rate_history.append(weighted_service_rate_mean)
        effective_service_rate_history.append(effective_service_rate_mean)
        service_quality_index_history.append(service_quality_index_mean)
        dispatch_efficiency_history.append(dispatch_efficiency_mean)
        effective_dispatch_rate_history.append(effective_dispatch_rate_mean)
        dispatch_roi_history.append(dispatch_roi_mean)

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

        if (i_ep + 1) % 50 == 0 or i_ep == 0:
            print(f"Episode: {i_ep + 1}/{cfg['train_eps']}")

        if (i_ep + 1) % 100 == 0 and path is not None:
            agent.save_model(path + f"marl_model_ep{i_ep + 1}.pth")

    res_dic = {
        "rewards": rewards,
        "raw_rewards": raw_rewards,
        "ma_rewards": ma_rewards,
        "critic_losses": critic_losses,
        "actor_losses": actor_losses,
        "rld_history": rld_history,
        "svc_history": svc_history,
        "srd_history": srd_history,
        "rld_last_history": rld_last_history,
        "svc_last_history": svc_last_history,
        "srd_last_history": srd_last_history,
        "service_rate_history": service_rate_history,
        "service_rate_last_history": service_rate_last_history,
        "info_missing_counts": info_missing_counts,
        "charging_efficiency_history": charging_efficiency_history,
        "vehicle_utilization_history": vehicle_utilization_history,
        "fleet_health_history": fleet_health_history,
        "zone_coordination_history": zone_coordination_history,
        "response_readiness_history": response_readiness_history,
        "charger_occupancy_rate_history": charger_occupancy_rate_history,
        "charger_queue_delay_history": charger_queue_delay_history,
        "energy_not_served_history": energy_not_served_history,
        "safety_soc_violation_count_history": safety_soc_violation_count_history,
        "charge_shift_success_rate_history": charge_shift_success_rate_history,
        "weighted_service_rate_history": weighted_service_rate_history,
        "effective_service_rate_history": effective_service_rate_history,
        "service_quality_index_history": service_quality_index_history,
        "dispatch_efficiency_history": dispatch_efficiency_history,
        "effective_dispatch_rate_history": effective_dispatch_rate_history,
        "dispatch_roi_history": dispatch_roi_history,
        "svc_by_timestep": np.asarray(svc_by_timestep, dtype=float),
        "rld_by_timestep": np.asarray(rld_by_timestep, dtype=float),
        "srd_by_timestep": np.asarray(srd_by_timestep, dtype=float),
        "reward_by_timestep": np.asarray(reward_by_timestep, dtype=float),
        "total_demand_by_timestep": np.asarray(total_demand_by_timestep, dtype=float),
        "served_orders_by_timestep": np.asarray(served_orders_by_timestep, dtype=float),
        "charging_efficiency_by_timestep": np.asarray(charging_efficiency_by_timestep, dtype=float),
        "charger_queue_delay_by_timestep": np.asarray(charger_queue_delay_by_timestep, dtype=float),
        "energy_not_served_by_timestep": np.asarray(energy_not_served_by_timestep, dtype=float),
        "soc_violation_by_timestep": np.asarray(soc_violation_by_timestep, dtype=float),
        "svc_timestep_mean": np.nanmean(np.asarray(svc_by_timestep, dtype=float), axis=0),
        "rld_timestep_mean": np.nanmean(np.asarray(rld_by_timestep, dtype=float), axis=0),
        "srd_timestep_mean": np.nanmean(np.asarray(srd_by_timestep, dtype=float), axis=0),
        "reward_timestep_mean": np.nanmean(np.asarray(reward_by_timestep, dtype=float), axis=0),
        "total_demand_timestep_mean": np.nanmean(np.asarray(total_demand_by_timestep, dtype=float), axis=0),
        "served_orders_timestep_mean": np.nanmean(np.asarray(served_orders_by_timestep, dtype=float), axis=0),
        "charging_efficiency_timestep_mean": np.nanmean(
            np.asarray(charging_efficiency_by_timestep, dtype=float), axis=0
        ),
        "charger_queue_delay_timestep_mean": np.nanmean(
            np.asarray(charger_queue_delay_by_timestep, dtype=float), axis=0
        ),
        "energy_not_served_timestep_mean": np.nanmean(
            np.asarray(energy_not_served_by_timestep, dtype=float), axis=0
        ),
        "soc_violation_timestep_mean": np.nanmean(np.asarray(soc_violation_by_timestep, dtype=float), axis=0),
        "zone_ids": np.arange(zone_num, dtype=int),
        "timestep_ids": np.arange(time_num, dtype=int),
        "demand_by_timestep_zone": np.asarray(demand_by_timestep_zone, dtype=float),
        "served_by_timestep_zone": np.asarray(served_by_timestep_zone, dtype=float),
        "service_rate_by_timestep_zone": np.asarray(service_rate_by_timestep_zone, dtype=float),
        "charging_actions_by_timestep_zone": np.asarray(charging_actions_by_timestep_zone, dtype=float),
        "dispatch_by_timestep_zone": np.asarray(dispatch_by_timestep_zone, dtype=float),
        "demand_timestep_zone_mean": np.nanmean(np.asarray(demand_by_timestep_zone, dtype=float), axis=0),
        "served_timestep_zone_mean": np.nanmean(np.asarray(served_by_timestep_zone, dtype=float), axis=0),
        "service_rate_timestep_zone_mean": np.nanmean(
            np.asarray(service_rate_by_timestep_zone, dtype=float), axis=0
        ),
        "charging_actions_timestep_zone_mean": np.nanmean(
            np.asarray(charging_actions_by_timestep_zone, dtype=float), axis=0
        ),
        "dispatch_timestep_zone_mean": np.nanmean(np.asarray(dispatch_by_timestep_zone, dtype=float), axis=0),
        "parking_available_by_timestep_zone": np.asarray(parking_available_by_timestep_zone, dtype=float),
        "parking_used_by_timestep_zone": np.asarray(parking_used_by_timestep_zone, dtype=float),
        "charger_available_by_timestep_zone": np.asarray(charger_available_by_timestep_zone, dtype=float),
        "charger_used_by_timestep_zone": np.asarray(charger_used_by_timestep_zone, dtype=float),
        "parking_available_timestep_zone_mean": np.nanmean(
            np.asarray(parking_available_by_timestep_zone, dtype=float), axis=0
        ),
        "parking_used_timestep_zone_mean": np.nanmean(
            np.asarray(parking_used_by_timestep_zone, dtype=float), axis=0
        ),
        "charger_available_timestep_zone_mean": np.nanmean(
            np.asarray(charger_available_by_timestep_zone, dtype=float), axis=0
        ),
        "charger_used_timestep_zone_mean": np.nanmean(
            np.asarray(charger_used_by_timestep_zone, dtype=float), axis=0
        ),
    }
    return res_dic


def marl_test(env, agent, cfg, path=None):
    rewards = []
    reward_scale = float(cfg.get("reward_scale", 1.0))
    if reward_scale <= 0:
        reward_scale = 1.0

    if hasattr(env, "set_experiment_variant"):
        env.set_experiment_variant(cfg.get("exp_variant", "a1"))
    if hasattr(env, "set_reward_mode"):
        env.set_reward_mode(cfg.get("reward_mode", "base"))

    for _ in range(cfg["test_eps"]):
        ep_reward = 0.0
        ep_step = 0
        global_state = env.reset()
        obs_n = np.array([env.get_agent_obs(agent_id) for agent_id in range(env.n_agents)])

        done = False
        info = {}
        max_steps = int(cfg.get("max_steps", 0))
        while (not done) and (max_steps <= 0 or ep_step < max_steps):
            actions = agent.sample_action(obs_n, global_state=global_state, add_noise=False)
            next_global_state, reward, done, info = env.step(actions)
            next_obs_n = np.array([env.get_agent_obs(agent_id) for agent_id in range(env.n_agents)])

            obs_n = next_obs_n
            global_state = next_global_state
            ep_reward += reward / reward_scale
            ep_step += 1

        rewards.append(ep_reward)

    if path is not None:
        np.save(path + "marl_test_rewards.npy", rewards)

    return rewards
