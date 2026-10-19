# -*- coding: utf-8 -*-
"""Non-RL optimization baseline with main.py-compatible artifacts."""

import io
import json
import os
import shutil
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import torch
import gym
from gym.envs.registration import register

from core.util import cp, get_args, plot_critic_losses, plot_rewards, plot_vehicle_battery_heatmap, smooth
from core.output_paths import RESULTS_ROOT
from core.run_provenance import write_run_provenance
import core.environment as dclilyenv


if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


register(id='dclily', entry_point='specific.dclily_elec:dclily')


def _normalize(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, 0.0, None)
    s = float(np.sum(arr))
    if s <= 1e-8:
        return np.ones_like(arr, dtype=np.float32) / max(1, arr.shape[0])
    return arr / s


def _optimization_action(env):
    z = int(env.action_space.shape[0] // 3)

    demand = getattr(env, 'od_demand_now', None)
    if isinstance(demand, np.ndarray) and demand.size >= z * z:
        mat = np.asarray(demand, dtype=np.float32)
        if mat.ndim == 1:
            mat = np.resize(mat, (z, z))
        else:
            mat = mat[:z, :z]
        origin_pref = np.sum(mat, axis=1)
        dest_pref = np.sum(mat, axis=0)
    else:
        origin_pref = np.ones(z, dtype=np.float32)
        dest_pref = np.ones(z, dtype=np.float32)

    avail = getattr(env, 'available', None)
    if isinstance(avail, np.ndarray) and avail.size >= z:
        avail_zone = np.asarray(avail[1][:z], dtype=np.float32) if avail.ndim >= 2 else np.asarray(avail[:z], dtype=np.float32)
    else:
        avail_zone = np.zeros(z, dtype=np.float32)

    cars = getattr(env, 'car', None)
    low_soc = np.zeros(z, dtype=np.float32)
    if isinstance(cars, (list, np.ndarray)):
        for item in cars:
            try:
                zone = int(item[0])
                soc = float(item[1])
                if 0 <= zone < z and soc < 900.0:
                    low_soc[zone] += 1.0
            except Exception:
                continue

    from_pref = 0.7 * origin_pref + 0.3 * avail_zone
    charge_pref = np.maximum(low_soc, 0.25 * dest_pref)

    action = np.concatenate([
        _normalize(from_pref),
        _normalize(dest_pref),
        _normalize(charge_pref),
    ]).astype(np.float32)
    return np.clip(action, 0.0, 1.0)


def _evaluate_policy(cfg, env, num_episodes):
    rewards = []
    reward_scale = max(1.0, float(cfg.get('reward_scale', 1.0)))
    for _ in range(num_episodes):
        if hasattr(env, 'set_training_mode'):
            env.set_training_mode(0)
        reset_ret = env.reset()
        state = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
        ep_reward = 0.0
        for _ in range(cfg['max_steps']):
            action = _optimization_action(env)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward / reward_scale
            state = next_state
            if done:
                break
        rewards.append(ep_reward)
    return float(np.mean(rewards)) if rewards else 0.0


def optimization_train(cfg, env):
    reward_scale = max(1.0, float(cfg.get('reward_scale', 1.0)))
    eval_every = int(cfg.get('eval_every', 0))
    eval_eps = max(1, int(cfg.get('eval_eps', 1)))

    rewards = []
    raw_rewards = []
    eval_rewards = []
    eval_episodes = []
    rld_history = []
    svc_history = []
    srd_history = []
    battery_log = []
    info_missing_counts = {'rld': 0, 'svc': 0, 'srd': 0}

    for i_ep in range(cfg['train_eps']):
        if hasattr(env, 'set_training_mode'):
            env.set_training_mode(1)
        if hasattr(env, 'set_epoch'):
            env.set_epoch(i_ep)

        reset_ret = env.reset()
        state = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
        ep_reward = 0.0
        ep_reward_raw = 0.0
        ep_battery_log = []

        for _ in range(cfg['max_steps']):
            action = _optimization_action(env)
            next_state, reward, done, info = env.step(action)
            ep_reward_raw += reward
            ep_reward += reward / reward_scale

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

            cars = getattr(env, 'car', None)
            if isinstance(cars, (list, np.ndarray)):
                try:
                    ep_battery_log.append([car[1] for car in cars])
                except Exception:
                    pass

            state = next_state
            if done:
                break

        rewards.append(ep_reward)
        raw_rewards.append(ep_reward_raw)
        battery_log.extend(ep_battery_log)
        print(f"回合：{i_ep + 1}/{cfg['train_eps']}，奖励(raw/scaled)：{ep_reward_raw:.2f}/{ep_reward:.2f}")

        if eval_every > 0 and (i_ep + 1) % eval_every == 0:
            eval_reward = _evaluate_policy(cfg, env, eval_eps)
            eval_rewards.append(eval_reward)
            eval_episodes.append(i_ep + 1)
            print(f"  [Eval] Episode {i_ep + 1}: mean reward={eval_reward:.2f} over {eval_eps} eps")

    return {
        'rewards': rewards,
        'raw_rewards': raw_rewards,
        'eval_rewards': eval_rewards,
        'eval_episodes': eval_episodes,
        'battery_log': battery_log,
        'critic1_losses': [],
        'critic2_losses': [],
        'actor_losses': [],
        'q_values': [],
        'target_q_values': [],
        'rld_history': rld_history,
        'svc_history': svc_history,
        'srd_history': srd_history,
        'info_missing_counts': info_missing_counts,
    }


def optimization_test(cfg, env):
    rewards = []
    reward_scale = max(1.0, float(cfg.get('reward_scale', 1.0)))
    for i_ep in range(cfg['test_eps']):
        if hasattr(env, 'set_training_mode'):
            env.set_training_mode(0)
        reset_ret = env.reset()
        state = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
        ep_reward = 0.0
        for _ in range(cfg['max_steps']):
            action = _optimization_action(env)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward / reward_scale
            state = next_state
            if done:
                break
        rewards.append(ep_reward)
        print(f"测试回合：{i_ep + 1}/{cfg['test_eps']}，奖励：{ep_reward:.2f}")
    return {'rewards': rewards}


def _plot_final_metrics(res_dic, out_file):
    fig, axes = plt.subplots(8, 1, figsize=(10, 20))

    axes[0].plot(res_dic['critic1_losses'], label='Critic Loss', color='blue', linewidth=1.5)
    axes[0].set_xlabel('Update Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Critic Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(res_dic['actor_losses'], label='Actor Loss (-Q)', color='red', linewidth=1.5)
    axes[1].set_xlabel('Update Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Actor Loss (-Q Value, should decrease)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(res_dic['q_values'], label='Q Value', color='cyan', linewidth=1.5)
    axes[2].set_xlabel('Update Step')
    axes[2].set_ylabel('Q Value')
    axes[2].set_title('Q Value (N/A for optimization baseline)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(res_dic['target_q_values'], label='Target Q', color='magenta', linewidth=1.5)
    axes[3].set_xlabel('Update Step')
    axes[3].set_ylabel('Target Q Value')
    axes[3].set_title('Target Q Value (N/A for optimization baseline)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(res_dic['rld_history'], label='RLD', color='green', linewidth=1.5)
    axes[4].set_xlabel('Step')
    axes[4].set_ylabel('RLD')
    axes[4].set_title('Repositioning Loss Distance (RLD)')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)

    axes[5].plot(res_dic['svc_history'], label='SVC', color='purple', linewidth=1.5)
    axes[5].set_xlabel('Step')
    axes[5].set_ylabel('SVC')
    axes[5].set_title('Service Rate (SVC)')
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)

    axes[6].plot(res_dic['srd_history'], label='SRD', color='orange', linewidth=1.5)
    axes[6].set_xlabel('Step')
    axes[6].set_ylabel('SRD')
    axes[6].set_title('Service Reward Ratio (SRD)')
    axes[6].legend()
    axes[6].grid(True, alpha=0.3)

    rewards = np.asarray(res_dic.get('rewards', []), dtype=float)
    if rewards.size > 0:
        axes[7].plot(rewards, label='Reward', linewidth=2.0, alpha=0.15, color='gray')
        if len(rewards) > 1:
            axes[7].plot(smooth(rewards), label='smoothed', linewidth=2.0)
        eval_rewards = np.asarray(res_dic.get('eval_rewards', []), dtype=float)
        eval_episodes = np.asarray(res_dic.get('eval_episodes', []), dtype=float)
        if eval_rewards.size > 0 and eval_episodes.size == eval_rewards.size:
            axes[7].plot(eval_episodes - 1, eval_rewards, 'o-', label='eval_reward', linewidth=1.5, markersize=4)
    axes[7].set_xlabel('Episode')
    axes[7].set_ylabel('Reward')
    axes[7].set_title('Episode Reward')
    axes[7].legend(loc='upper right')
    axes[7].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_file('final_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"综合指标图已保存: {out_file('final_metrics.png')}")


def main():
    cfg = get_args()
    cfg['algo_name'] = 'OPTIMIZED'

    if str(cfg.get('data_folder', '')).lower().startswith('data\\toy_network'):
        cfg['data_folder'] = 'data\\SIP\\'

    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(current_dir, 'specific')
    data_dir = os.path.join(current_dir, cfg['data_folder'])
    sip_cfg = os.path.join(data_dir, 'config_sip.json')
    plain_cfg = os.path.join(data_dir, 'config.json')
    if os.path.exists(sip_cfg) and not os.path.exists(plain_cfg):
        shutil.copy2(sip_cfg, plain_cfg)
        print(f"Prepared SIP config: {plain_cfg}")

    os.chdir(target_dir)
    cp("..\\" + cfg['data_folder'])
    os.chdir('..')

    output_root = os.fspath(RESULTS_ROOT)
    os.makedirs(output_root, exist_ok=True)
    run_stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(output_root, f"{run_stamp}_{str(cfg['algo_name']).upper()}_{int(cfg['train_eps'])}ep")
    os.makedirs(run_dir, exist_ok=True)
    cfg['run_output_dir'] = run_dir
    print(f"运行输出目录: {run_dir}")
    write_run_provenance(run_dir, cfg)
    print(f"完整参数与Git版本已保存到: {run_dir}")

    def out_file(filename):
        return os.path.join(run_dir, filename)

    env = dclilyenv.NormalizedActions(gym.make(cfg['env_name']))
    if cfg['seed'] != 0:
        dclilyenv.all_seed(env, seed=cfg['seed'])

    obs_shape = tuple(getattr(env.observation_space, 'shape', ()) or ())
    act_shape = tuple(getattr(env.action_space, 'shape', ()) or ())
    if len(obs_shape) == 0:
        raise ValueError('Observation space shape is unavailable for OPTIMIZED baseline')
    if len(act_shape) == 0:
        raise ValueError('Action space shape is unavailable for OPTIMIZED baseline')
    cfg['n_states'] = int(obs_shape[0])
    cfg['n_actions'] = int(act_shape[0])

    res_dic = optimization_train(cfg, env)

    np.save(out_file('critic1_losses.npy'), res_dic['critic1_losses'])
    np.save(out_file('critic2_losses.npy'), res_dic['critic2_losses'])
    np.save(out_file('actor_losses.npy'), res_dic['actor_losses'])
    np.save(out_file('q_values.npy'), res_dic['q_values'])
    np.save(out_file('target_q_values.npy'), res_dic['target_q_values'])
    np.save(out_file('rewards.npy'), res_dic['rewards'])
    np.save(out_file('raw_rewards.npy'), res_dic['raw_rewards'])
    np.save(out_file('eval_rewards.npy'), res_dic.get('eval_rewards', []))
    np.save(out_file('eval_episodes.npy'), res_dic.get('eval_episodes', []))
    np.save(out_file('rld_history.npy'), res_dic['rld_history'])
    np.save(out_file('svc_history.npy'), res_dic['svc_history'])
    np.save(out_file('srd_history.npy'), res_dic['srd_history'])
    print('Loss数据和指标已保存')
    print(f"Info字段缺失计数: {res_dic.get('info_missing_counts', {})}")

    metadata = {
        'algo_name': cfg['algo_name'],
        'seed': cfg['seed'],
        'train_eps': cfg['train_eps'],
        'batch_size': cfg['batch_size'],
        'actor_lr': cfg.get('actor_lr', cfg.get('lr', 'N/A')),
        'critic_lr': cfg.get('critic_lr', cfg.get('lr', 'N/A')),
        'gamma': cfg['gamma'],
        'tau': cfg.get('tau', 'N/A'),
        'memory_capacity': cfg['memory_capacity'],
        'n_states': cfg['n_states'],
        'n_actions': cfg['n_actions'],
        'timestamp': time.strftime('%Y%m%d-%H%M%S'),
        'total_episodes': len(res_dic['rewards']),
        'mean_reward': float(np.mean(res_dic['rewards'])) if len(res_dic['rewards']) > 0 else 0.0,
        'max_reward': float(np.max(res_dic['rewards'])) if len(res_dic['rewards']) > 0 else 0.0,
        'final_reward': float(res_dic['rewards'][-1]) if len(res_dic['rewards']) > 0 else 0.0,
        'std_reward': float(np.std(res_dic['rewards'])) if len(res_dic['rewards']) > 0 else 0.0,
    }
    with open(out_file('training_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"训练元数据已保存到: {out_file('training_metadata.json')}")

    torch.save({'policy': 'non_rl_optimization', 'config': cfg}, out_file('trained_model.pth'))
    print(f"训练完成，模型已保存到: {out_file('trained_model.pth')}")

    plot_rewards(res_dic['rewards'], cfg, tag='train')
    plot_critic_losses(res_dic['critic1_losses'], res_dic.get('critic2_losses', None), tag='train', save_path=out_file('critic_loss_curve.png'))
    _plot_final_metrics(res_dic, out_file)

    plot_vehicle_battery_heatmap(res_dic['battery_log'], title='Battery Heatmap', save_path=out_file('battery_heatmap.png'))
    print(f"电池热力图已保存: {out_file('battery_heatmap.png')}")

    test_dic = optimization_test(cfg, env)
    plot_rewards(test_dic['rewards'], cfg, tag='test')
    np.save(out_file('test_rewards.npy'), test_dic.get('rewards', []))


if __name__ == '__main__':
    main()
