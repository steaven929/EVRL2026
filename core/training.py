# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 19:24:55 2024

@author: TLab
"""

import numpy as np
import random
from collections import deque
from core.marl_metrics import compute_all_metrics


def _charge_slice_from_action(action):
    action_dim = int(np.asarray(action).shape[0])
    if action_dim % 3 != 0:
        raise ValueError(f"action dim {action_dim} is not divisible by 3")
    zone_num = action_dim // 3
    return slice(2 * zone_num, 3 * zone_num)


def _evaluate_policy(cfg, env, agent, num_episodes):
    rewards = []
    reward_scale = float(cfg.get('reward_scale', 1.0))
    if reward_scale <= 0:
        reward_scale = 1.0
    for _ in range(num_episodes):
        env.set_training_mode(0)
        reset_ret = env.reset()
        state = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
        ep_reward = 0.0
        for _ in range(cfg['max_steps']):
            action = agent.predict_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward / reward_scale
            state = next_state
            if done:
                break
        rewards.append(ep_reward)
    return float(np.mean(rewards)) if rewards else 0.0


def _episode_nanmean(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return np.nan
    with np.errstate(invalid='ignore'):
        return float(np.nanmean(arr))


def _safe_nanmean(arr, axis, fallback_shape):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return np.full(fallback_shape, np.nan, dtype=float)
    finite = np.isfinite(arr)
    counts = np.sum(finite, axis=axis)
    summed = np.sum(np.where(finite, arr, 0.0), axis=axis)
    denom = np.where(counts > 0, counts, 1)
    mean = summed / denom
    if np.isscalar(mean):
        return float(mean) if counts > 0 else np.nan
    mean = np.asarray(mean, dtype=float)
    mean[np.asarray(counts) == 0] = np.nan
    return mean

class OUNoise(object):
    '''Ornstein–Uhlenbeck噪声
    '''
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.05, min_sigma=0.005, decay_period=10000):
        self.mu           = mu # OU噪声的参数
        self.theta        = theta # OU噪声的参数
        self.sigma        = max_sigma # OU噪声的参数
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.n_actions    = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
    def reset(self):
        self.obs = np.ones(self.n_actions) * self.mu
    def evolve_obs(self):
        x  = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.n_actions)
        self.obs = x + dx
        return self.obs
    def get_action(self, action, t=0):
        ou_obs = self.evolve_obs()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period) # sigma会逐渐衰减
        if random.random()>0.99:
            print(ou_obs)
        return np.clip(action + ou_obs, self.low, self.high) # 动作加上噪声后进行剪切
def encode_custom_action(zone_to_counts, zone_from_counts=None, charge_pref=None, zone_num=4):
    """
    按照车辆调度分布生成动作向量
    :param zone_to_counts: 每个区域作为目的地的车辆数量 [zone0, zone1, ...]
    :param zone_from_counts: 每个区域作为出发地的车辆数量 [zone0, zone1, ...]，如果为空，则均匀分配
    :param charge_pref: 充电倾向分布（zone级别），如果为空，默认等于目的地分布
    :return: 长度为 3 * zone_num 的动作向量
    """
    def normalize(arr):
        total = sum(arr)
        return [x / total if total > 0 else 0 for x in arr]

    if zone_from_counts is None:
        zone_from_counts = [1] * zone_num  # 均匀分布
    if charge_pref is None:
        charge_pref = zone_to_counts  # 默认充电倾向 = 去哪充哪

    vec = normalize(zone_from_counts) + normalize(zone_to_counts) + normalize(charge_pref)
    return np.array(vec, dtype=np.float32)



def train(cfg, env, agent):
    print("开始训练！")
    algo_name = str(cfg.get('algo_name', '')).upper()
    is_sac = algo_name == 'SAC'
    is_td3 = algo_name == 'TD3'
    is_pure_ddpg = algo_name == 'DDPG_PURE'
    exp_variant = str(cfg.get('exp_variant', 'a1')).lower()
    if hasattr(env, 'set_experiment_variant'):
        env.set_experiment_variant(exp_variant)
    if hasattr(env, 'set_reward_mode'):
        env.set_reward_mode(cfg.get('reward_mode', 'base'))
    ou_noise = None if (is_sac or is_td3) else OUNoise(env.action_space)  # SAC/TD3不叠加OU噪声
    reward_scale = float(cfg.get('reward_scale', 1.0))
    if reward_scale <= 0:
        reward_scale = 1.0
    eval_every = int(cfg.get('eval_every', 0))
    eval_eps = max(1, int(cfg.get('eval_eps', 1)))
    rewards = [] # 记录所有回合的奖励
    critic1_losses, critic2_losses, actor_losses = [], [], []
    q_values, target_q_values = [], []  # 修改：记录Q值和目标Q值用于诊断
    rld_history, svc_history, srd_history = [], [], []  # 修改
    eval_rewards, eval_episodes = [], []
    raw_rewards = []

    env_time_num = int(getattr(env, 'TIME_NUM', 26))
    if env_time_num <= 0:
        env_time_num = 26

    max_steps = int(cfg.get('max_steps', 0))
    if max_steps <= 0:
        max_steps = env_time_num

    # 日志时间维度应跟随环境真实时域，而不是训练循环上限(max_steps)
    time_num = int(cfg.get('log_time_num', env_time_num))
    if time_num <= 0:
        time_num = env_time_num
    time_num = min(time_num, max_steps)
    zone_num = int(getattr(env, 'ZONE_NUM', max(1, int(env.action_space.shape[0] // 3))))

    svc_by_timestep, rld_by_timestep, srd_by_timestep = [], [], []
    reward_by_timestep, total_demand_by_timestep, served_orders_by_timestep = [], [], []
    charging_efficiency_by_timestep, charger_queue_delay_by_timestep = [], []
    energy_not_served_by_timestep, soc_violation_by_timestep = [], []

    demand_by_timestep_zone, served_by_timestep_zone = [], []
    service_rate_by_timestep_zone, charging_actions_by_timestep_zone = [], []
    dispatch_by_timestep_zone = []
    parking_available_by_timestep_zone, parking_used_by_timestep_zone = [], []
    charger_available_by_timestep_zone, charger_used_by_timestep_zone = [], []

    charging_efficiency_history = []
    charger_queue_delay_history = []
    energy_not_served_history = []
    safety_soc_violation_count_history = []

    info_missing_counts = {'rld': 0, 'svc': 0, 'srd': 0}

    battery_log_keep_steps = int(cfg.get('battery_log_keep_steps', 2000))
    if battery_log_keep_steps <= 0:
        battery_log_keep_steps = 1
    battery_log_stride = int(cfg.get('battery_log_stride', 1))
    if battery_log_stride <= 0:
        battery_log_stride = 1
    battery_log = deque(maxlen=battery_log_keep_steps)

    # =========2026.2.7修改：保存探索期动作==========
    exploration_actions_saved = False  # 标记是否已保存探索期动作
    # ======修改结束

    for i_ep in range(cfg['train_eps']):
        env.set_training_mode(1)
        # =========2026.2.7修改，方案D2：设置当前epoch==========
        env.set_epoch(i_ep)  # 设置当前轮数，用于探索策略衰减

        # 在探索期（第10轮）保存详细动作
        if i_ep == 10 and not exploration_actions_saved:
            env.set_training_mode(2)  # 设置为探索期记录模式
            exploration_actions_saved = True
        # ======修改结束
        reset_ret = env.reset()
        state = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
        if ou_noise is not None:
            ou_noise.reset()
        ep_reward = 0.0
        ep_reward_raw = 0.0
        current_ep_battery_log = []  # 修改：每个episode初始化当前episode的电池日志  # 修改

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

        for i_step in range(max_steps):
            action = agent.sample_action(state)
            if ou_noise is not None:
                action = ou_noise.get_action(action, i_step+1)

            # =========2026.2.14修改，实验C1：需求感知探索噪声==========
            if is_sac and exp_variant == 'c1':
                demand_now = 0.0
                if hasattr(env, 'od_demand_now') and getattr(env, 'od_demand_now', np.array([])).size > 0:
                    demand_now = float(np.sum(env.od_demand_now[0]))
                demand_factor = demand_now / (demand_now + 12.0)
                noise_sigma = 0.12 - 0.09 * demand_factor  # 低需求高噪声，高需求低噪声
                noise_sigma = max(0.03, min(0.12, noise_sigma))
                action_noise = np.random.normal(0.0, noise_sigma, size=action.shape[0])
                action = np.clip(action + action_noise, 0.0, 1.0)
            # ======修改结束

            # =========2026.2.7修改，方案D2：充电维度噪声增强==========
            # 前20轮训练，对充电维度添加额外噪声（按动作维度动态切片）
            if (not is_sac) and (not is_td3) and (not is_pure_ddpg) and i_ep < 20:
                charge_slice = _charge_slice_from_action(action)
                charge_dim = charge_slice.stop - charge_slice.start
                charging_noise = np.random.uniform(-0.3, 0.3, size=charge_dim)
                action[charge_slice] = np.clip(action[charge_slice] + charging_noise, 0, 1)
            # ======修改结束

            next_state, reward, done, info = env.step(action)  # 修改：移除epoch参数
            ep_reward_raw += reward
            scaled_reward = reward / reward_scale
            ep_reward += scaled_reward

            step_idx = min(i_step, time_num - 1)
            ep_reward_by_t[step_idx] = float(scaled_reward)
            info_dict = info if isinstance(info, dict) else {}

            if 'svc' in info_dict:
                ep_svc_by_t[step_idx] = float(info_dict['svc'])
            if 'rld' in info_dict:
                ep_rld_by_t[step_idx] = float(info_dict['rld'])
            if 'srd' in info_dict:
                ep_srd_by_t[step_idx] = float(info_dict['srd'])
            if 'total_demand' in info_dict:
                ep_total_demand_by_t[step_idx] = float(info_dict['total_demand'])
            if 'served_orders' in info_dict:
                ep_served_orders_by_t[step_idx] = float(info_dict['served_orders'])

            if 'demand_matrix' in info_dict:
                dm = np.asarray(info_dict['demand_matrix'], dtype=float)
                if dm.ndim == 2:
                    z = min(zone_num, dm.shape[0])
                    ep_demand_by_tz[step_idx, :z] = np.sum(dm[:z, :], axis=1)

            if 'demand_orders' in info_dict:
                sm = np.asarray(info_dict['demand_orders'], dtype=float)
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

            if 'charging_actions' in info_dict and isinstance(info_dict['charging_actions'], dict):
                carr = np.zeros(zone_num, dtype=float)
                for k, v in info_dict['charging_actions'].items():
                    try:
                        zi = int(k)
                        if 0 <= zi < zone_num:
                            carr[zi] = float(v)
                    except Exception:
                        continue
                ep_charging_by_tz[step_idx] = carr

            if 'action_table_true' in info_dict:
                am = np.asarray(info_dict['action_table_true'], dtype=float)
                if am.ndim == 2:
                    z = min(zone_num, am.shape[0])
                    ep_dispatch_by_tz[step_idx, :z] = np.sum(am[:z, :], axis=1)

            if 'parking_available_by_zone' in info_dict:
                pa = np.asarray(info_dict['parking_available_by_zone'], dtype=float)
                z = min(zone_num, pa.shape[0])
                ep_parking_available_by_tz[step_idx, :z] = pa[:z]

            if 'parking_used_by_zone' in info_dict:
                pu = np.asarray(info_dict['parking_used_by_zone'], dtype=float)
                z = min(zone_num, pu.shape[0])
                ep_parking_used_by_tz[step_idx, :z] = pu[:z]

            if 'charger_available_by_zone' in info_dict:
                ca = np.asarray(info_dict['charger_available_by_zone'], dtype=float)
                z = min(zone_num, ca.shape[0])
                ep_charger_available_by_tz[step_idx, :z] = ca[:z]

            if 'charger_used_by_zone' in info_dict:
                cu = np.asarray(info_dict['charger_used_by_zone'], dtype=float)
                z = min(zone_num, cu.shape[0])
                ep_charger_used_by_tz[step_idx, :z] = cu[:z]

            step_metrics = compute_all_metrics(env, info_dict)
            ep_ce_by_t[step_idx] = float(step_metrics.get('charging_efficiency', np.nan))
            ep_qd_by_t[step_idx] = float(step_metrics.get('charger_queue_delay', np.nan))
            ep_ens_by_t[step_idx] = float(step_metrics.get('energy_not_served', np.nan))
            ep_socv_by_t[step_idx] = float(step_metrics.get('safety_soc_violation_count', np.nan))

            # 记录环境指标  # 修改
            if 'rld' in info:  # 修改
                rld_history.append(info['rld'])  # 修改
            else:
                info_missing_counts['rld'] += 1
            if 'svc' in info:  # 修改
                svc_history.append(info['svc'])  # 修改
            else:
                info_missing_counts['svc'] += 1
            if 'srd' in info:  # 修改
                srd_history.append(info['srd'])  # 修改
            else:
                info_missing_counts['srd'] += 1

            if (i_step % battery_log_stride) == 0:
                battery_snapshot = np.asarray([car[1] for car in env.car], dtype=np.float32)
                current_ep_battery_log.append(battery_snapshot)

            agent.memory.push((state, action, scaled_reward, next_state, bool(done)))
            #agent.update()  #2026.1.22修改
            loss_info = agent.update()  # 让它返回 dict
            if loss_info is not None:
                if 'critic1_loss' in loss_info: critic1_losses.append(loss_info['critic1_loss'])
                if 'critic2_loss' in loss_info: critic2_losses.append(loss_info['critic2_loss'])
                if 'actor_loss' in loss_info and loss_info['actor_loss'] is not None:
                    actor_losses.append(loss_info['actor_loss'])
                if 'q_value' in loss_info:      q_values.append(loss_info['q_value'])  # 修改
                if 'target_q' in loss_info:     target_q_values.append(loss_info['target_q'])  # 修改
            state = next_state

            if done:
                break
        if 1: #(i_ep+1)%1 == 0:
            print(f"回合：{i_ep+1}/{cfg['train_eps']}，奖励(raw/scaled)：{ep_reward_raw:.2f}/{ep_reward:.2f}")
        rewards.append(ep_reward)
        raw_rewards.append(ep_reward_raw)

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

        charging_efficiency_history.append(_episode_nanmean(ep_ce_by_t))
        charger_queue_delay_history.append(_episode_nanmean(ep_qd_by_t))
        energy_not_served_history.append(_episode_nanmean(ep_ens_by_t))
        safety_soc_violation_count_history.append(_episode_nanmean(ep_socv_by_t))

        if eval_every > 0 and (i_ep + 1) % eval_every == 0:
            eval_reward = _evaluate_policy(cfg, env, agent, eval_eps)
            eval_rewards.append(eval_reward)
            eval_episodes.append(i_ep + 1)
            print(f"  [Eval] Episode {i_ep+1}: mean reward={eval_reward:.2f} over {eval_eps} eps")
        battery_log.extend(current_ep_battery_log)
    print(f"[InfoCoverage] missing counts: {info_missing_counts}")
    print("完成训练！")

    if svc_by_timestep:
        svc_by_timestep_arr = np.asarray(svc_by_timestep, dtype=float)
        rld_by_timestep_arr = np.asarray(rld_by_timestep, dtype=float)
        srd_by_timestep_arr = np.asarray(srd_by_timestep, dtype=float)
        reward_by_timestep_arr = np.asarray(reward_by_timestep, dtype=float)
        total_demand_by_timestep_arr = np.asarray(total_demand_by_timestep, dtype=float)
        served_orders_by_timestep_arr = np.asarray(served_orders_by_timestep, dtype=float)
        charging_efficiency_by_timestep_arr = np.asarray(charging_efficiency_by_timestep, dtype=float)
        charger_queue_delay_by_timestep_arr = np.asarray(charger_queue_delay_by_timestep, dtype=float)
        energy_not_served_by_timestep_arr = np.asarray(energy_not_served_by_timestep, dtype=float)
        soc_violation_by_timestep_arr = np.asarray(soc_violation_by_timestep, dtype=float)
    else:
        svc_by_timestep_arr = np.empty((0, time_num), dtype=float)
        rld_by_timestep_arr = np.empty((0, time_num), dtype=float)
        srd_by_timestep_arr = np.empty((0, time_num), dtype=float)
        reward_by_timestep_arr = np.empty((0, time_num), dtype=float)
        total_demand_by_timestep_arr = np.empty((0, time_num), dtype=float)
        served_orders_by_timestep_arr = np.empty((0, time_num), dtype=float)
        charging_efficiency_by_timestep_arr = np.empty((0, time_num), dtype=float)
        charger_queue_delay_by_timestep_arr = np.empty((0, time_num), dtype=float)
        energy_not_served_by_timestep_arr = np.empty((0, time_num), dtype=float)
        soc_violation_by_timestep_arr = np.empty((0, time_num), dtype=float)

    if demand_by_timestep_zone:
        demand_by_timestep_zone_arr = np.asarray(demand_by_timestep_zone, dtype=float)
        served_by_timestep_zone_arr = np.asarray(served_by_timestep_zone, dtype=float)
        service_rate_by_timestep_zone_arr = np.asarray(service_rate_by_timestep_zone, dtype=float)
        charging_actions_by_timestep_zone_arr = np.asarray(charging_actions_by_timestep_zone, dtype=float)
        dispatch_by_timestep_zone_arr = np.asarray(dispatch_by_timestep_zone, dtype=float)
        parking_available_by_timestep_zone_arr = np.asarray(parking_available_by_timestep_zone, dtype=float)
        parking_used_by_timestep_zone_arr = np.asarray(parking_used_by_timestep_zone, dtype=float)
        charger_available_by_timestep_zone_arr = np.asarray(charger_available_by_timestep_zone, dtype=float)
        charger_used_by_timestep_zone_arr = np.asarray(charger_used_by_timestep_zone, dtype=float)
    else:
        demand_by_timestep_zone_arr = np.empty((0, time_num, zone_num), dtype=float)
        served_by_timestep_zone_arr = np.empty((0, time_num, zone_num), dtype=float)
        service_rate_by_timestep_zone_arr = np.empty((0, time_num, zone_num), dtype=float)
        charging_actions_by_timestep_zone_arr = np.empty((0, time_num, zone_num), dtype=float)
        dispatch_by_timestep_zone_arr = np.empty((0, time_num, zone_num), dtype=float)
        parking_available_by_timestep_zone_arr = np.empty((0, time_num, zone_num), dtype=float)
        parking_used_by_timestep_zone_arr = np.empty((0, time_num, zone_num), dtype=float)
        charger_available_by_timestep_zone_arr = np.empty((0, time_num, zone_num), dtype=float)
        charger_used_by_timestep_zone_arr = np.empty((0, time_num, zone_num), dtype=float)

    if len(battery_log) > 0:
        battery_log_arr = np.asarray(list(battery_log), dtype=np.float32)
    else:
        battery_log_arr = np.empty((0, int(getattr(env, 'CAR_NUM', 0))), dtype=np.float32)

    return {'rewards': rewards,
    'raw_rewards': raw_rewards,
    'eval_rewards': eval_rewards,
    'eval_episodes': eval_episodes,
    'battery_log': battery_log_arr,
    'critic1_losses': critic1_losses,
    'critic2_losses': critic2_losses,
    'actor_losses': actor_losses,
    'q_values': q_values,  # 修改
    'target_q_values': target_q_values,  # 修改
    'rld_history': rld_history,  # 修改
    'svc_history': svc_history,  # 修改
    'srd_history': srd_history,
    'info_missing_counts': info_missing_counts,
    'charging_efficiency_history': charging_efficiency_history,
    'charger_queue_delay_history': charger_queue_delay_history,
    'energy_not_served_history': energy_not_served_history,
    'safety_soc_violation_count_history': safety_soc_violation_count_history,

    'svc_by_timestep': svc_by_timestep_arr,
    'rld_by_timestep': rld_by_timestep_arr,
    'srd_by_timestep': srd_by_timestep_arr,
    'reward_by_timestep': reward_by_timestep_arr,
    'total_demand_by_timestep': total_demand_by_timestep_arr,
    'served_orders_by_timestep': served_orders_by_timestep_arr,
    'charging_efficiency_by_timestep': charging_efficiency_by_timestep_arr,
    'charger_queue_delay_by_timestep': charger_queue_delay_by_timestep_arr,
    'energy_not_served_by_timestep': energy_not_served_by_timestep_arr,
    'soc_violation_by_timestep': soc_violation_by_timestep_arr,

    'svc_timestep_mean': _safe_nanmean(svc_by_timestep_arr, axis=0, fallback_shape=(time_num,)),
    'rld_timestep_mean': _safe_nanmean(rld_by_timestep_arr, axis=0, fallback_shape=(time_num,)),
    'srd_timestep_mean': _safe_nanmean(srd_by_timestep_arr, axis=0, fallback_shape=(time_num,)),
    'reward_timestep_mean': _safe_nanmean(reward_by_timestep_arr, axis=0, fallback_shape=(time_num,)),
    'total_demand_timestep_mean': _safe_nanmean(total_demand_by_timestep_arr, axis=0, fallback_shape=(time_num,)),
    'served_orders_timestep_mean': _safe_nanmean(served_orders_by_timestep_arr, axis=0, fallback_shape=(time_num,)),
    'charging_efficiency_timestep_mean': _safe_nanmean(charging_efficiency_by_timestep_arr, axis=0, fallback_shape=(time_num,)),
    'charger_queue_delay_timestep_mean': _safe_nanmean(charger_queue_delay_by_timestep_arr, axis=0, fallback_shape=(time_num,)),
    'energy_not_served_timestep_mean': _safe_nanmean(energy_not_served_by_timestep_arr, axis=0, fallback_shape=(time_num,)),
    'soc_violation_timestep_mean': _safe_nanmean(soc_violation_by_timestep_arr, axis=0, fallback_shape=(time_num,)),

    'zone_ids': np.arange(zone_num, dtype=int),
    'timestep_ids': np.arange(time_num, dtype=int),
    'demand_by_timestep_zone': demand_by_timestep_zone_arr,
    'served_by_timestep_zone': served_by_timestep_zone_arr,
    'service_rate_by_timestep_zone': service_rate_by_timestep_zone_arr,
    'charging_actions_by_timestep_zone': charging_actions_by_timestep_zone_arr,
    'dispatch_by_timestep_zone': dispatch_by_timestep_zone_arr,
    'parking_available_by_timestep_zone': parking_available_by_timestep_zone_arr,
    'parking_used_by_timestep_zone': parking_used_by_timestep_zone_arr,
    'charger_available_by_timestep_zone': charger_available_by_timestep_zone_arr,
    'charger_used_by_timestep_zone': charger_used_by_timestep_zone_arr,

    'demand_timestep_zone_mean': _safe_nanmean(demand_by_timestep_zone_arr, axis=0, fallback_shape=(time_num, zone_num)),
    'served_timestep_zone_mean': _safe_nanmean(served_by_timestep_zone_arr, axis=0, fallback_shape=(time_num, zone_num)),
    'service_rate_timestep_zone_mean': _safe_nanmean(service_rate_by_timestep_zone_arr, axis=0, fallback_shape=(time_num, zone_num)),
    'charging_actions_timestep_zone_mean': _safe_nanmean(charging_actions_by_timestep_zone_arr, axis=0, fallback_shape=(time_num, zone_num)),
    'dispatch_timestep_zone_mean': _safe_nanmean(dispatch_by_timestep_zone_arr, axis=0, fallback_shape=(time_num, zone_num)),
    'parking_available_timestep_zone_mean': _safe_nanmean(parking_available_by_timestep_zone_arr, axis=0, fallback_shape=(time_num, zone_num)),
    'parking_used_timestep_zone_mean': _safe_nanmean(parking_used_by_timestep_zone_arr, axis=0, fallback_shape=(time_num, zone_num)),
    'charger_available_timestep_zone_mean': _safe_nanmean(charger_available_by_timestep_zone_arr, axis=0, fallback_shape=(time_num, zone_num)),
    'charger_used_timestep_zone_mean': _safe_nanmean(charger_used_by_timestep_zone_arr, axis=0, fallback_shape=(time_num, zone_num)),
    }  # 修改

def test(cfg, env, agent):
    print("开始测试！")
    exp_variant = str(cfg.get('exp_variant', 'a1')).lower()
    if hasattr(env, 'set_experiment_variant'):
        env.set_experiment_variant(exp_variant)
    if hasattr(env, 'set_reward_mode'):
        env.set_reward_mode(cfg.get('reward_mode', 'base'))
    rewards = [] # 记录所有回合的奖励
    reward_scale = float(cfg.get('reward_scale', 1.0))
    if reward_scale <= 0:
        reward_scale = 1.0
    for i_ep in range(cfg['test_eps']):
        env.set_training_mode(0)
        reset_ret = env.reset()
        state = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
        ep_reward = 0
        for i_step in range(cfg['max_steps']):
            action = agent.predict_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward / reward_scale
            state = next_state
            if done:
                break
        rewards.append(ep_reward)
        print(f"回合：{i_ep+1}/{cfg['test_eps']}，奖励：{ep_reward:.2f}")
    print("完成测试！")
    return {'rewards':rewards}
