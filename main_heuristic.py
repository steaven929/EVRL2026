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
    return f"./outputs/heuristic_{train_eps}ep_{timestamp}/"


def _run_episode(env, cfg):
    reward_scale = float(cfg.get("reward_scale", 1.0))
    if reward_scale <= 0:
        reward_scale = 1.0

    _ = env.reset()
    done = False
    flat_action = np.zeros(env.action_space.shape[0], dtype=np.float32)

    ep_reward = 0.0
    ep_reward_raw = 0.0
    svc_steps = []
    rld_steps = []
    srd_steps = []

    while not done:
        _, reward, done, info = env.step(flat_action)
        scaled_reward = reward / reward_scale
        ep_reward += scaled_reward
        ep_reward_raw += reward
        if isinstance(info, dict):
            svc_steps.append(float(info.get("svc", 0.0)))
            rld_steps.append(float(info.get("rld", 0.0)))
            srd_steps.append(float(info.get("srd", 0.0)))

    return {
        "reward": ep_reward,
        "raw_reward": ep_reward_raw,
        "svc": float(np.mean(svc_steps)) if svc_steps else 0.0,
        "rld": float(np.mean(rld_steps)) if rld_steps else 0.0,
        "srd": float(np.mean(srd_steps)) if srd_steps else 0.0,
    }


def run_heuristic(env, cfg):
    rewards = []
    raw_rewards = []
    ma_rewards = []
    svc_history = []
    rld_history = []
    srd_history = []

    for i_ep in range(int(cfg["train_eps"])):
        episode_stats = _run_episode(env, cfg)
        rewards.append(episode_stats["reward"])
        raw_rewards.append(episode_stats["raw_reward"])
        svc_history.append(episode_stats["svc"])
        rld_history.append(episode_stats["rld"])
        srd_history.append(episode_stats["srd"])
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
    }


def run_heuristic_test(env, cfg, path=None):
    test_rewards = []
    original_eps = int(cfg.get("test_eps", 1))
    for _ in range(original_eps):
        episode_stats = _run_episode(env, cfg)
        test_rewards.append(episode_stats["reward"])
    if path is not None:
        np.save(os.path.join(path, "marl_test_rewards.npy"), np.asarray(test_rewards, dtype=float))
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
