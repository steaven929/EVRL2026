# -*- coding: utf-8 -*-
import datetime
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
from gym.envs.registration import register, registry

from core.environment import env_agent_config
from core.marl_metrics import compute_all_metrics
from core.util import get_args, plot_rewards
from output_paths import build_sa_run_dir
from ri_metrics import build_ri_exports
from specific.sa_solver import SimulatedAnnealingSolver

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

if 'dclily' not in registry:
    register(id='dclily', entry_point='specific.dclily_elec:dclily')


REQUIRED_OUTPUT_FILES = [
    'rewards.npy',
    'raw_rewards.npy',
    'rld_history.npy',
    'svc_history.npy',
    'srd_history.npy',
    'ri_history.npy',
    'charging_efficiency_history.npy',
    'charger_queue_delay_history.npy',
    'energy_not_served_history.npy',
    'safety_soc_violation_count_history.npy',
    'svc_by_timestep.npy',
    'rld_by_timestep.npy',
    'ri_by_timestep.npy',
    'srd_by_timestep.npy',
    'reward_by_timestep.npy',
    'total_demand_by_timestep.npy',
    'served_orders_by_timestep.npy',
    'ri_timestep_mean.npy',
    'training_metadata.json',
]


def _battery_capacity(env):
    return float(getattr(env, 'BATTERY_CAPACITY', None) or getattr(getattr(env, 'unwrapped', env), 'BATTERY_CAPACITY', 3000.0))


def _resolve_available_by_zone(env):
    getter = getattr(env, 'get_zone_vehicle_counts', None)
    if callable(getter):
        values = getter()
    elif hasattr(env, 'available') and len(env.available) > 0:
        values = env.available[0]
    else:
        raise AttributeError('Environment does not expose per-zone available vehicles')
    return np.asarray(values, dtype=float).copy()


def _zone_avg_soc(env):
    zone_num = int(getattr(env, 'ZONE_NUM', len(_resolve_available_by_zone(env))))
    if not hasattr(env, 'car'):
        raise AttributeError('Environment does not expose vehicle state for SOC aggregation')
    totals = np.zeros(zone_num, dtype=float)
    counts = np.zeros(zone_num, dtype=float)
    cap = max(_battery_capacity(env), 1.0)
    for vehicle in np.asarray(env.car):
        zone = int(vehicle[0])
        if 0 <= zone < zone_num:
            totals[zone] += float(vehicle[1]) / cap
            counts[zone] += 1.0
    avg = np.full(zone_num, 0.5, dtype=float)
    valid = counts > 0
    avg[valid] = totals[valid] / counts[valid]
    return avg


def _free_chargers(env):
    getter = getattr(env, 'get_free_charger_counts', None)
    if callable(getter):
        values = getter()
        return np.asarray(values, dtype=float).copy()
    stations = getattr(env, 'charging_stations', None)
    if not isinstance(stations, dict):
        raise AttributeError('Environment does not expose charger state for SA snapshot extraction')
    zone_num = int(getattr(env, 'ZONE_NUM', len(stations)))
    free = np.zeros(zone_num, dtype=float)
    for zone in range(zone_num):
        chargers = stations.get(zone, [])
        free[zone] = float(sum(1 for charger in chargers if charger.get('occupied_by') is None))
    return free


def collect_sa_snapshot(env):
    if not hasattr(env, 'od_demand_now'):
        raise AttributeError('Environment does not expose current OD demand')
    demand_now = np.asarray(env.od_demand_now, dtype=float)
    if demand_now.ndim != 3 or demand_now.shape[0] <= 0:
        raise ValueError(f'Expected od_demand_now with shape (time_slice, zone, zone), got {demand_now.shape}')
    demand_matrix = demand_now[0]
    return {
        'demand_out': np.sum(demand_matrix, axis=1),
        'demand_in': np.sum(demand_matrix, axis=0),
        'available': _resolve_available_by_zone(env),
        'avg_soc': _zone_avg_soc(env),
        'charger_free': _free_chargers(env),
        'distance': np.asarray(env.dist, dtype=float).copy(),
    }


def build_sa_metadata(cfg, rewards, run_dir):
    rewards_arr = np.asarray(rewards, dtype=float)
    return {
        'algo_name': 'SA',
        'seed': int(cfg.get('seed', 0)),
        'train_eps': int(cfg.get('train_eps', 0)),
        'timestamp': os.path.basename(os.path.normpath(run_dir)),
        'total_episodes': int(rewards_arr.size),
        'mean_reward': float(np.mean(rewards_arr)) if rewards_arr.size else 0.0,
        'final_reward': float(rewards_arr[-1]) if rewards_arr.size else 0.0,
    }


def _safe_nanmean(arr, axis, fallback_shape):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return np.full(fallback_shape, np.nan, dtype=float)
    finite = np.isfinite(arr)
    counts = np.sum(finite, axis=axis)
    summed = np.sum(np.where(finite, arr, 0.0), axis=axis)
    denom = np.where(counts > 0, counts, 1)
    mean = summed / denom
    mean = np.asarray(mean, dtype=float)
    mean[np.asarray(counts) == 0] = np.nan
    return mean


def _save_array(path, name, value):
    np.save(os.path.join(path, name), np.asarray(value, dtype=float))


def _run_episode(env, solver, reward_scale, time_num):
    reset_ret = env.reset()
    _ = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
    done = False
    ep_reward = 0.0
    ep_reward_raw = 0.0
    ep_ri_steps = []
    ep_svc = np.full(time_num, np.nan, dtype=float)
    ep_rld = np.full(time_num, np.nan, dtype=float)
    ep_srd = np.full(time_num, np.nan, dtype=float)
    ep_reward_t = np.full(time_num, np.nan, dtype=float)
    ep_total_demand = np.full(time_num, np.nan, dtype=float)
    ep_served = np.full(time_num, np.nan, dtype=float)
    ce_steps, qd_steps, ens_steps, socv_steps = [], [], [], []

    step_idx = 0
    while not done and step_idx < time_num:
        snapshot = collect_sa_snapshot(env)
        action = solver.solve(snapshot)
        step_ret = env.step(action)
        if len(step_ret) == 5:
            _, reward, terminated, truncated, info = step_ret
            done = bool(terminated or truncated)
        else:
            _, reward, done, info = step_ret
        info = info if isinstance(info, dict) else {}
        scaled_reward = float(reward) / reward_scale
        ep_reward += scaled_reward
        ep_reward_raw += float(reward)

        ep_reward_t[step_idx] = scaled_reward
        ep_svc[step_idx] = float(info.get('svc', np.nan))
        ep_rld[step_idx] = float(info.get('rld', np.nan))
        ep_srd[step_idx] = float(info.get('srd', np.nan))
        ep_total_demand[step_idx] = float(info.get('total_demand', np.nan))
        ep_served[step_idx] = float(info.get('served_orders', np.nan))
        ep_ri_steps.append(float(info.get('rld', np.nan)))

        step_metrics = compute_all_metrics(env, info)
        ce_steps.append(float(step_metrics.get('charging_efficiency', np.nan)))
        qd_steps.append(float(step_metrics.get('charger_queue_delay', np.nan)))
        ens_steps.append(float(step_metrics.get('energy_not_served', np.nan)))
        socv_steps.append(float(step_metrics.get('safety_soc_violation_count', np.nan)))
        step_idx += 1

    return {
        'reward': ep_reward,
        'raw_reward': ep_reward_raw,
        'ri_steps': ep_ri_steps,
        'svc_by_timestep': ep_svc,
        'rld_by_timestep': ep_rld,
        'srd_by_timestep': ep_srd,
        'reward_by_timestep': ep_reward_t,
        'total_demand_by_timestep': ep_total_demand,
        'served_orders_by_timestep': ep_served,
        'charging_efficiency_history': float(np.nanmean(ce_steps)) if ce_steps else np.nan,
        'charger_queue_delay_history': float(np.nanmean(qd_steps)) if qd_steps else np.nan,
        'energy_not_served_history': float(np.nanmean(ens_steps)) if ens_steps else np.nan,
        'safety_soc_violation_count_history': float(np.nanmean(socv_steps)) if socv_steps else np.nan,
    }


def main():
    cfg = get_args()
    env, _ = env_agent_config(cfg)
    if hasattr(env, 'set_training_mode'):
        env.set_training_mode(1)
    if hasattr(env, 'set_experiment_variant'):
        env.set_experiment_variant(cfg.get('exp_variant', 'a1'))
    if hasattr(env, 'set_reward_mode'):
        env.set_reward_mode(cfg.get('reward_mode', 'shaping'))

    curr_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = build_sa_run_dir(train_eps=cfg['train_eps'], curr_time=curr_time)
    os.makedirs(run_dir, exist_ok=True)

    solver = SimulatedAnnealingSolver(
        zone_num=int(getattr(env, 'ZONE_NUM', env.action_space.shape[0] // 3)),
        init_temp=cfg['sa_init_temp'],
        final_temp=cfg['sa_final_temp'],
        cooling_rate=cfg['sa_cooling_rate'],
        iterations=cfg['sa_iters_per_step'],
        seed=cfg.get('seed', None),
        seed_action=cfg.get('sa_seed_action', 'demand_charge'),
        weight_dispatch_support=cfg.get('sa_weight_dispatch_support', 0.4),
        weight_charging_bonus=cfg.get('sa_weight_charging_bonus', 0.7),
        weight_dispatch_cost=cfg.get('sa_weight_dispatch_cost', 0.15),
        weight_low_soc_penalty=cfg.get('sa_weight_low_soc_penalty', 3.0),
        weight_unmet_penalty=cfg.get('sa_weight_unmet_penalty', 0.1),
    )

    reward_scale = float(cfg.get('reward_scale', 1.0)) or 1.0
    time_num = int(getattr(env, 'TIME_NUM', 26))

    rewards = []
    raw_rewards = []
    rld_history, svc_history, srd_history = [], [], []
    charging_efficiency_history = []
    charger_queue_delay_history = []
    energy_not_served_history = []
    safety_soc_violation_count_history = []
    episode_ri_steps = []

    svc_by_timestep = []
    rld_by_timestep = []
    srd_by_timestep = []
    reward_by_timestep = []
    total_demand_by_timestep = []
    served_orders_by_timestep = []

    for _ in range(int(cfg['train_eps'])):
        episode = _run_episode(env, solver, reward_scale, time_num)
        rewards.append(episode['reward'])
        raw_rewards.append(episode['raw_reward'])
        svc_by_timestep.append(episode['svc_by_timestep'])
        rld_by_timestep.append(episode['rld_by_timestep'])
        srd_by_timestep.append(episode['srd_by_timestep'])
        reward_by_timestep.append(episode['reward_by_timestep'])
        total_demand_by_timestep.append(episode['total_demand_by_timestep'])
        served_orders_by_timestep.append(episode['served_orders_by_timestep'])
        episode_ri_steps.append(episode['ri_steps'])

        svc_vals = episode['svc_by_timestep']
        rld_vals = episode['rld_by_timestep']
        srd_vals = episode['srd_by_timestep']
        svc_history.append(float(np.nanmean(svc_vals)) if np.any(np.isfinite(svc_vals)) else 0.0)
        rld_history.append(float(np.nanmean(rld_vals)) if np.any(np.isfinite(rld_vals)) else 0.0)
        srd_history.append(float(np.nanmean(srd_vals)) if np.any(np.isfinite(srd_vals)) else 0.0)
        charging_efficiency_history.append(episode['charging_efficiency_history'])
        charger_queue_delay_history.append(episode['charger_queue_delay_history'])
        energy_not_served_history.append(episode['energy_not_served_history'])
        safety_soc_violation_count_history.append(episode['safety_soc_violation_count_history'])

    svc_arr = np.asarray(svc_by_timestep, dtype=float) if svc_by_timestep else np.empty((0, time_num), dtype=float)
    rld_arr = np.asarray(rld_by_timestep, dtype=float) if rld_by_timestep else np.empty((0, time_num), dtype=float)
    srd_arr = np.asarray(srd_by_timestep, dtype=float) if srd_by_timestep else np.empty((0, time_num), dtype=float)
    reward_arr = np.asarray(reward_by_timestep, dtype=float) if reward_by_timestep else np.empty((0, time_num), dtype=float)
    demand_arr = np.asarray(total_demand_by_timestep, dtype=float) if total_demand_by_timestep else np.empty((0, time_num), dtype=float)
    served_arr = np.asarray(served_orders_by_timestep, dtype=float) if served_orders_by_timestep else np.empty((0, time_num), dtype=float)

    ri_exports = build_ri_exports(
        episode_step_metrics=episode_ri_steps,
        by_timestep=rld_arr,
        timestep_mean=_safe_nanmean(rld_arr, axis=0, fallback_shape=(time_num,)),
    )

    _save_array(run_dir, 'rewards.npy', rewards)
    _save_array(run_dir, 'raw_rewards.npy', raw_rewards)
    _save_array(run_dir, 'rld_history.npy', rld_history)
    _save_array(run_dir, 'svc_history.npy', svc_history)
    _save_array(run_dir, 'srd_history.npy', srd_history)
    _save_array(run_dir, 'ri_history.npy', ri_exports['ri_history'])
    _save_array(run_dir, 'charging_efficiency_history.npy', charging_efficiency_history)
    _save_array(run_dir, 'charger_queue_delay_history.npy', charger_queue_delay_history)
    _save_array(run_dir, 'energy_not_served_history.npy', energy_not_served_history)
    _save_array(run_dir, 'safety_soc_violation_count_history.npy', safety_soc_violation_count_history)
    _save_array(run_dir, 'svc_by_timestep.npy', svc_arr)
    _save_array(run_dir, 'rld_by_timestep.npy', rld_arr)
    _save_array(run_dir, 'ri_by_timestep.npy', ri_exports['ri_by_timestep'])
    _save_array(run_dir, 'srd_by_timestep.npy', srd_arr)
    _save_array(run_dir, 'reward_by_timestep.npy', reward_arr)
    _save_array(run_dir, 'total_demand_by_timestep.npy', demand_arr)
    _save_array(run_dir, 'served_orders_by_timestep.npy', served_arr)
    _save_array(run_dir, 'ri_timestep_mean.npy', ri_exports['ri_timestep_mean'])

    with open(os.path.join(run_dir, 'training_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(build_sa_metadata(cfg, rewards, run_dir), f, indent=2)

    plot_rewards(rewards, cfg, path=os.path.join(run_dir, 'train_rewards.png'), tag='train')
    print(f"SA outputs saved to: {run_dir}")


if __name__ == '__main__':
    main()
