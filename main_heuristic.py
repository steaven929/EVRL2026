# -*- coding: utf-8 -*-
"""Heuristic baseline entrypoint with MARL-compatible artifact output."""

import datetime
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gym.envs.registration import register, registry

from core.improved_metrics import compute_all_improved_metrics
from core.output_paths import result_run_directory
from core.marl_metrics import compute_all_metrics
from core.util import get_args
from main_marl_nooptim import save_training_artifacts


HEURISTIC_ENV_ID = "dclily_heuristic"
HEURISTIC_ENTRY_POINT = "specific.dclily_elec_heuristic:dclily"


def register_heuristic_env():
    if HEURISTIC_ENV_ID not in registry:
        register(id=HEURISTIC_ENV_ID, entry_point=HEURISTIC_ENTRY_POINT)


def build_cfg():
    cfg = get_args()
    cfg["env_name"] = HEURISTIC_ENV_ID
    cfg["algo_name"] = "HEURISTIC"
    cfg["marl_mode"] = "single"
    return cfg


def build_output_path(train_eps, timestamp):
    return os.fspath(result_run_directory(f"heuristic_{train_eps}ep_{timestamp}")) + os.sep


def _apply_runtime_env_config(env, cfg, epoch=None):
    if hasattr(env, "set_experiment_variant"):
        env.set_experiment_variant(cfg.get("exp_variant", "a1"))
    if hasattr(env, "set_reward_mode"):
        env.set_reward_mode(cfg.get("reward_mode", "base"))
    if epoch is not None and hasattr(env, "set_epoch"):
        env.set_epoch(epoch)


def _aggregate_episode_metric(values):
    if len(values) == 0:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 0.0
    return float(np.mean(finite)), float(finite[-1])


def _nanmean_axis0(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    valid = np.isfinite(arr)
    counts = np.sum(valid, axis=0)
    sums = np.sum(np.where(valid, arr, 0.0), axis=0)
    out = np.full(arr.shape[1:], np.nan, dtype=float)
    mask = counts > 0
    out[mask] = sums[mask] / counts[mask]
    return out


def _run_episode(env, cfg):
    reward_scale = float(cfg.get("reward_scale", 1.0))
    if reward_scale <= 0:
        reward_scale = 1.0

    _ = env.reset()
    done = False
    flat_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    time_num = int(getattr(env, "TIME_NUM", cfg.get("TIME_NUM", 26)))
    zone_num = int(getattr(env, "ZONE_NUM", cfg.get("n_agents", 50)))

    ep_reward = 0.0
    ep_reward_raw = 0.0
    ep_step = 0
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

    while not done:
        step_idx = min(ep_step, time_num - 1)
        _, reward, done, info = env.step(flat_action)
        scaled_reward = reward / reward_scale
        ep_reward += scaled_reward
        ep_reward_raw += reward

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

            for key in ep_metric_steps:
                if key in info:
                    ep_metric_steps[key].append(float(info[key]))

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
            if np.any(np.isfinite(ep_demand_by_tz[step_idx])) and np.any(np.isfinite(ep_served_by_tz[step_idx])):
                dem = ep_demand_by_tz[step_idx]
                srv = ep_served_by_tz[step_idx]
                rate = np.full(zone_num, np.nan, dtype=float)
                valid = np.isfinite(dem) & np.isfinite(srv) & (dem > 0)
                rate[valid] = srv[valid] / dem[valid]
                ep_service_rate_by_tz[step_idx] = rate

            if "charging_actions" in info and isinstance(info["charging_actions"], dict):
                carr = np.zeros(zone_num, dtype=float)
                for k, v in info["charging_actions"].items():
                    try:
                        zi = int(k)
                    except Exception:
                        continue
                    if 0 <= zi < zone_num:
                        carr[zi] = float(v)
                ep_charging_by_tz[step_idx] = carr

            if "action_table_true" in info:
                am = np.asarray(info["action_table_true"], dtype=float)
                if am.ndim == 2:
                    z = min(zone_num, am.shape[0])
                    ep_dispatch_by_tz[step_idx, :z] = np.sum(am[:z, :], axis=1)

            for info_key, target in [
                ("parking_available_by_zone", ep_parking_available_by_tz),
                ("parking_used_by_zone", ep_parking_used_by_tz),
                ("charger_available_by_zone", ep_charger_available_by_tz),
                ("charger_used_by_zone", ep_charger_used_by_tz),
            ]:
                if info_key in info:
                    arr = np.asarray(info[info_key], dtype=float)
                    z = min(zone_num, arr.shape[0])
                    target[step_idx, :z] = arr[:z]

            marl_metrics = compute_all_metrics(env, info)
            improved_metrics = compute_all_improved_metrics(env, info)
            ep_ce_by_t[step_idx] = float(marl_metrics.get("charging_efficiency", np.nan))
            ep_qd_by_t[step_idx] = float(marl_metrics.get("charger_queue_delay", np.nan))
            ep_ens_by_t[step_idx] = float(marl_metrics.get("energy_not_served", np.nan))
            ep_socv_by_t[step_idx] = float(marl_metrics.get("safety_soc_violation_count", np.nan))
            for key in ep_marl_metric_steps:
                ep_marl_metric_steps[key].append(marl_metrics.get(key, np.nan))
            for key in ep_improved_metric_steps:
                ep_improved_metric_steps[key].append(improved_metrics.get(key, np.nan))

        ep_step += 1

    return {
        "reward": ep_reward,
        "raw_reward": ep_reward_raw,
        "svc": _aggregate_episode_metric(ep_metric_steps["svc"])[0],
        "rld": _aggregate_episode_metric(ep_metric_steps["rld"])[0],
        "srd": _aggregate_episode_metric(ep_metric_steps["srd"])[0],
        "service_rate": _aggregate_episode_metric(ep_metric_steps["service_rate"])[0],
        "charging_efficiency": _aggregate_episode_metric(ep_marl_metric_steps["charging_efficiency"])[0],
        "vehicle_utilization": _aggregate_episode_metric(ep_marl_metric_steps["vehicle_utilization"])[0],
        "fleet_health": _aggregate_episode_metric(ep_marl_metric_steps["fleet_health"])[0],
        "zone_coordination": _aggregate_episode_metric(ep_marl_metric_steps["zone_coordination"])[0],
        "response_readiness": _aggregate_episode_metric(ep_marl_metric_steps["response_readiness"])[0],
        "charger_occupancy_rate": _aggregate_episode_metric(ep_marl_metric_steps["charger_occupancy_rate"])[0],
        "charger_queue_delay": _aggregate_episode_metric(ep_marl_metric_steps["charger_queue_delay"])[0],
        "energy_not_served": _aggregate_episode_metric(ep_marl_metric_steps["energy_not_served"])[0],
        "safety_soc_violation_count": _aggregate_episode_metric(ep_marl_metric_steps["safety_soc_violation_count"])[0],
        "charge_shift_success_rate": _aggregate_episode_metric(ep_marl_metric_steps["charge_shift_success_rate"])[0],
        "weighted_service_rate": _aggregate_episode_metric(ep_improved_metric_steps["weighted_service_rate"])[0],
        "effective_service_rate": _aggregate_episode_metric(ep_improved_metric_steps["effective_service_rate"])[0],
        "service_quality_index": _aggregate_episode_metric(ep_improved_metric_steps["service_quality_index"])[0],
        "dispatch_efficiency": _aggregate_episode_metric(ep_improved_metric_steps["dispatch_efficiency"])[0],
        "effective_dispatch_rate": _aggregate_episode_metric(ep_improved_metric_steps["effective_dispatch_rate"])[0],
        "dispatch_roi": _aggregate_episode_metric(ep_improved_metric_steps["dispatch_roi"])[0],
        "svc_by_timestep": ep_svc_by_t,
        "rld_by_timestep": ep_rld_by_t,
        "srd_by_timestep": ep_srd_by_t,
        "reward_by_timestep": ep_reward_by_t,
        "total_demand_by_timestep": ep_total_demand_by_t,
        "served_orders_by_timestep": ep_served_orders_by_t,
        "charging_efficiency_by_timestep": ep_ce_by_t,
        "charger_queue_delay_by_timestep": ep_qd_by_t,
        "energy_not_served_by_timestep": ep_ens_by_t,
        "soc_violation_by_timestep": ep_socv_by_t,
        "demand_by_timestep_zone": ep_demand_by_tz,
        "served_by_timestep_zone": ep_served_by_tz,
        "service_rate_by_timestep_zone": ep_service_rate_by_tz,
        "charging_actions_by_timestep_zone": ep_charging_by_tz,
        "dispatch_by_timestep_zone": ep_dispatch_by_tz,
        "parking_available_by_timestep_zone": ep_parking_available_by_tz,
        "parking_used_by_timestep_zone": ep_parking_used_by_tz,
        "charger_available_by_timestep_zone": ep_charger_available_by_tz,
        "charger_used_by_timestep_zone": ep_charger_used_by_tz,
    }


def run_heuristic(env, cfg):
    rewards = []
    raw_rewards = []
    ma_rewards = []
    svc_history = []
    rld_history = []
    srd_history = []
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
    zone_num = int(getattr(env, "ZONE_NUM", cfg.get("n_agents", 50)))
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

    for i_ep in range(int(cfg["train_eps"])):
        _apply_runtime_env_config(env, cfg, epoch=i_ep)
        episode_stats = _run_episode(env, cfg)
        rewards.append(episode_stats["reward"])
        raw_rewards.append(episode_stats["raw_reward"])
        svc_history.append(episode_stats["svc"])
        rld_history.append(episode_stats["rld"])
        srd_history.append(episode_stats["srd"])
        charging_efficiency_history.append(episode_stats["charging_efficiency"])
        vehicle_utilization_history.append(episode_stats["vehicle_utilization"])
        fleet_health_history.append(episode_stats["fleet_health"])
        zone_coordination_history.append(episode_stats["zone_coordination"])
        response_readiness_history.append(episode_stats["response_readiness"])
        charger_occupancy_rate_history.append(episode_stats["charger_occupancy_rate"])
        charger_queue_delay_history.append(episode_stats["charger_queue_delay"])
        energy_not_served_history.append(episode_stats["energy_not_served"])
        safety_soc_violation_count_history.append(episode_stats["safety_soc_violation_count"])
        charge_shift_success_rate_history.append(episode_stats["charge_shift_success_rate"])
        weighted_service_rate_history.append(episode_stats["weighted_service_rate"])
        effective_service_rate_history.append(episode_stats["effective_service_rate"])
        service_quality_index_history.append(episode_stats["service_quality_index"])
        dispatch_efficiency_history.append(episode_stats["dispatch_efficiency"])
        effective_dispatch_rate_history.append(episode_stats["effective_dispatch_rate"])
        dispatch_roi_history.append(episode_stats["dispatch_roi"])
        svc_by_timestep.append(episode_stats["svc_by_timestep"])
        rld_by_timestep.append(episode_stats["rld_by_timestep"])
        srd_by_timestep.append(episode_stats["srd_by_timestep"])
        reward_by_timestep.append(episode_stats["reward_by_timestep"])
        total_demand_by_timestep.append(episode_stats["total_demand_by_timestep"])
        served_orders_by_timestep.append(episode_stats["served_orders_by_timestep"])
        charging_efficiency_by_timestep.append(episode_stats["charging_efficiency_by_timestep"])
        charger_queue_delay_by_timestep.append(episode_stats["charger_queue_delay_by_timestep"])
        energy_not_served_by_timestep.append(episode_stats["energy_not_served_by_timestep"])
        soc_violation_by_timestep.append(episode_stats["soc_violation_by_timestep"])
        demand_by_timestep_zone.append(episode_stats["demand_by_timestep_zone"])
        served_by_timestep_zone.append(episode_stats["served_by_timestep_zone"])
        service_rate_by_timestep_zone.append(episode_stats["service_rate_by_timestep_zone"])
        charging_actions_by_timestep_zone.append(episode_stats["charging_actions_by_timestep_zone"])
        dispatch_by_timestep_zone.append(episode_stats["dispatch_by_timestep_zone"])
        parking_available_by_timestep_zone.append(episode_stats["parking_available_by_timestep_zone"])
        parking_used_by_timestep_zone.append(episode_stats["parking_used_by_timestep_zone"])
        charger_available_by_timestep_zone.append(episode_stats["charger_available_by_timestep_zone"])
        charger_used_by_timestep_zone.append(episode_stats["charger_used_by_timestep_zone"])
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * episode_stats["reward"])
        else:
            ma_rewards.append(episode_stats["reward"])

        if (i_ep + 1) % 50 == 0 or i_ep == 0:
            print(f"Episode: {i_ep + 1}/{cfg['train_eps']}, reward={episode_stats['reward']:.2f}")

    return {
        "critic_losses": np.asarray([], dtype=float),
        "actor_losses": np.asarray([], dtype=float),
        "rewards": np.asarray(rewards, dtype=float),
        "raw_rewards": np.asarray(raw_rewards, dtype=float),
        "ma_rewards": np.asarray(ma_rewards, dtype=float),
        "svc_history": np.asarray(svc_history, dtype=float),
        "rld_history": np.asarray(rld_history, dtype=float),
        "srd_history": np.asarray(srd_history, dtype=float),
        "charging_efficiency_history": np.asarray(charging_efficiency_history, dtype=float),
        "vehicle_utilization_history": np.asarray(vehicle_utilization_history, dtype=float),
        "fleet_health_history": np.asarray(fleet_health_history, dtype=float),
        "zone_coordination_history": np.asarray(zone_coordination_history, dtype=float),
        "response_readiness_history": np.asarray(response_readiness_history, dtype=float),
        "charger_occupancy_rate_history": np.asarray(charger_occupancy_rate_history, dtype=float),
        "charger_queue_delay_history": np.asarray(charger_queue_delay_history, dtype=float),
        "energy_not_served_history": np.asarray(energy_not_served_history, dtype=float),
        "safety_soc_violation_count_history": np.asarray(safety_soc_violation_count_history, dtype=float),
        "charge_shift_success_rate_history": np.asarray(charge_shift_success_rate_history, dtype=float),
        "weighted_service_rate_history": np.asarray(weighted_service_rate_history, dtype=float),
        "effective_service_rate_history": np.asarray(effective_service_rate_history, dtype=float),
        "service_quality_index_history": np.asarray(service_quality_index_history, dtype=float),
        "dispatch_efficiency_history": np.asarray(dispatch_efficiency_history, dtype=float),
        "effective_dispatch_rate_history": np.asarray(effective_dispatch_rate_history, dtype=float),
        "dispatch_roi_history": np.asarray(dispatch_roi_history, dtype=float),
        "zone_ids": np.arange(zone_num, dtype=int),
        "timestep_ids": np.arange(time_num, dtype=int),
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
        "svc_timestep_mean": _nanmean_axis0(svc_by_timestep),
        "rld_timestep_mean": _nanmean_axis0(rld_by_timestep),
        "srd_timestep_mean": _nanmean_axis0(srd_by_timestep),
        "reward_timestep_mean": _nanmean_axis0(reward_by_timestep),
        "total_demand_timestep_mean": _nanmean_axis0(total_demand_by_timestep),
        "served_orders_timestep_mean": _nanmean_axis0(served_orders_by_timestep),
        "charging_efficiency_timestep_mean": _nanmean_axis0(charging_efficiency_by_timestep),
        "charger_queue_delay_timestep_mean": _nanmean_axis0(charger_queue_delay_by_timestep),
        "energy_not_served_timestep_mean": _nanmean_axis0(energy_not_served_by_timestep),
        "soc_violation_timestep_mean": _nanmean_axis0(soc_violation_by_timestep),
        "demand_by_timestep_zone": np.asarray(demand_by_timestep_zone, dtype=float),
        "served_by_timestep_zone": np.asarray(served_by_timestep_zone, dtype=float),
        "service_rate_by_timestep_zone": np.asarray(service_rate_by_timestep_zone, dtype=float),
        "charging_actions_by_timestep_zone": np.asarray(charging_actions_by_timestep_zone, dtype=float),
        "dispatch_by_timestep_zone": np.asarray(dispatch_by_timestep_zone, dtype=float),
        "demand_timestep_zone_mean": _nanmean_axis0(demand_by_timestep_zone),
        "served_timestep_zone_mean": _nanmean_axis0(served_by_timestep_zone),
        "service_rate_timestep_zone_mean": _nanmean_axis0(service_rate_by_timestep_zone),
        "charging_actions_timestep_zone_mean": _nanmean_axis0(charging_actions_by_timestep_zone),
        "dispatch_timestep_zone_mean": _nanmean_axis0(dispatch_by_timestep_zone),
        "parking_available_by_timestep_zone": np.asarray(parking_available_by_timestep_zone, dtype=float),
        "parking_used_by_timestep_zone": np.asarray(parking_used_by_timestep_zone, dtype=float),
        "charger_available_by_timestep_zone": np.asarray(charger_available_by_timestep_zone, dtype=float),
        "charger_used_by_timestep_zone": np.asarray(charger_used_by_timestep_zone, dtype=float),
        "parking_available_timestep_zone_mean": _nanmean_axis0(parking_available_by_timestep_zone),
        "parking_used_timestep_zone_mean": _nanmean_axis0(parking_used_by_timestep_zone),
        "charger_available_timestep_zone_mean": _nanmean_axis0(charger_available_by_timestep_zone),
        "charger_used_timestep_zone_mean": _nanmean_axis0(charger_used_by_timestep_zone),
    }


def run_heuristic_test(env, cfg, path=None):
    test_rewards = []
    original_eps = int(cfg.get("test_eps", 1))
    for _ in range(original_eps):
        _apply_runtime_env_config(env, cfg, epoch=None)
        episode_stats = _run_episode(env, cfg)
        test_rewards.append(episode_stats["reward"])
    if path is not None:
        np.save(os.path.join(path, "marl_test_rewards.npy"), np.asarray(test_rewards, dtype=float))
        np.save(os.path.join(path, "eval_episodes.npy"), np.asarray([int(cfg.get("train_eps", 0))], dtype=int))
        np.save(os.path.join(path, "eval_rewards.npy"), np.asarray([float(np.mean(test_rewards))], dtype=float))
    return np.asarray(test_rewards, dtype=float)


def main():
    register_heuristic_env()
    cfg = build_cfg()

    import gym

    env = gym.make(cfg["env_name"])
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_eps = cfg.get("train_eps", 0)
    path = build_output_path(train_eps, curr_time)
    os.makedirs(path, exist_ok=True)

    res_dic = run_heuristic(env, cfg)
    save_training_artifacts(path, res_dic, cfg)
    run_heuristic_test(env, cfg, path=path)
    env.close()
    return res_dic


if __name__ == "__main__":
    main()
