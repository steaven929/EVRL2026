# -*- coding: utf-8 -*-
"""Independent entrypoint for the 1000-car heuristic baseline."""

import io
import json
import os
import sys
import time
from datetime import datetime
from unittest.mock import patch

import gym
from gym.envs.registration import register, registry
import numpy as np

import core.environment as dclilyenv
import core.training as dclilytrain
from core.output_naming import build_outputs_dir
from core.util import cp, get_args, plot_rewards


ENV_ID = "dclily_heuristic_1000"
ENTRY_POINT = "specific.dclily_elec_heuristic_1000:dclily"


if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


class _SimpleReplayBuffer:
    def __init__(self):
        self._data = []

    def push(self, transition):
        self._data.append(transition)

    def __len__(self):
        return len(self._data)


class HeuristicAgent:
    """Compatibility agent so the heuristic env can reuse the existing train loop."""

    def __init__(self, action_dim):
        self.action_dim = int(action_dim)
        self.memory = _SimpleReplayBuffer()

    def sample_action(self, state):
        return np.zeros(self.action_dim, dtype=np.float32)

    def predict_action(self, state):
        return np.zeros(self.action_dim, dtype=np.float32)

    def update(self):
        return None


def register_heuristic_1000_env():
    try:
        already_registered = ENV_ID in registry
    except Exception:
        already_registered = ENV_ID in getattr(registry, "env_specs", {})
    if already_registered:
        return
    register(id=ENV_ID, entry_point=ENTRY_POINT)


def build_cfg(argv=None):
    argv = [] if argv is None else list(argv)
    with patch.object(sys, "argv", [sys.argv[0], *argv]):
        cfg = get_args()
    cfg["env_name"] = ENV_ID
    cfg["algo_name"] = "HEURISTIC_1000"
    return cfg


def _copy_data_config(cfg):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(current_dir, "specific")
    os.chdir(target_dir)
    try:
        cp("..\\" + cfg["data_folder"])
    finally:
        os.chdir(current_dir)


def _save_results(cfg, run_dir, res_dic):
    os.makedirs(run_dir, exist_ok=True)

    np.save(os.path.join(run_dir, "rewards.npy"), res_dic["rewards"])
    np.save(os.path.join(run_dir, "raw_rewards.npy"), res_dic.get("raw_rewards", []))
    np.save(os.path.join(run_dir, "eval_rewards.npy"), res_dic.get("eval_rewards", []))
    np.save(os.path.join(run_dir, "eval_episodes.npy"), res_dic.get("eval_episodes", []))
    np.save(os.path.join(run_dir, "rld_history.npy"), res_dic.get("rld_history", []))
    np.save(os.path.join(run_dir, "svc_history.npy"), res_dic.get("svc_history", []))
    np.save(os.path.join(run_dir, "srd_history.npy"), res_dic.get("srd_history", []))
    np.save(os.path.join(run_dir, "charging_efficiency_history.npy"), res_dic.get("charging_efficiency_history", []))
    np.save(os.path.join(run_dir, "charger_queue_delay_history.npy"), res_dic.get("charger_queue_delay_history", []))
    np.save(os.path.join(run_dir, "energy_not_served_history.npy"), res_dic.get("energy_not_served_history", []))
    np.save(
        os.path.join(run_dir, "safety_soc_violation_count_history.npy"),
        res_dic.get("safety_soc_violation_count_history", []),
    )

    optional_keys = [
        "zone_ids",
        "timestep_ids",
        "svc_by_timestep",
        "rld_by_timestep",
        "srd_by_timestep",
        "reward_by_timestep",
        "total_demand_by_timestep",
        "served_orders_by_timestep",
        "charging_efficiency_by_timestep",
        "charger_queue_delay_by_timestep",
        "energy_not_served_by_timestep",
        "soc_violation_by_timestep",
        "svc_timestep_mean",
        "rld_timestep_mean",
        "srd_timestep_mean",
        "reward_timestep_mean",
        "total_demand_timestep_mean",
        "served_orders_timestep_mean",
        "charging_efficiency_timestep_mean",
        "charger_queue_delay_timestep_mean",
        "energy_not_served_timestep_mean",
        "soc_violation_timestep_mean",
        "demand_by_timestep_zone",
        "served_by_timestep_zone",
        "service_rate_by_timestep_zone",
        "charging_actions_by_timestep_zone",
        "dispatch_by_timestep_zone",
        "demand_timestep_zone_mean",
        "served_timestep_zone_mean",
        "service_rate_timestep_zone_mean",
        "charging_actions_timestep_zone_mean",
        "dispatch_timestep_zone_mean",
        "parking_available_by_timestep_zone",
        "parking_used_by_timestep_zone",
        "charger_available_by_timestep_zone",
        "charger_used_by_timestep_zone",
        "parking_available_timestep_zone_mean",
        "parking_used_timestep_zone_mean",
        "charger_available_timestep_zone_mean",
        "charger_used_timestep_zone_mean",
    ]
    for key in optional_keys:
        if key in res_dic:
            np.save(os.path.join(run_dir, f"{key}.npy"), res_dic[key])

    plot_rewards(res_dic["rewards"], cfg, path=os.path.join(run_dir, "train_rewards.png"), tag="train")

    metadata = {
        "algo_name": cfg["algo_name"],
        "env_name": cfg["env_name"],
        "seed": cfg["seed"],
        "train_eps": cfg["train_eps"],
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "total_episodes": len(res_dic["rewards"]),
        "mean_reward": float(np.mean(res_dic["rewards"])) if res_dic["rewards"] else 0.0,
        "max_reward": float(np.max(res_dic["rewards"])) if res_dic["rewards"] else 0.0,
        "final_reward": float(res_dic["rewards"][-1]) if res_dic["rewards"] else 0.0,
    }
    with open(os.path.join(run_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main(argv=None):
    register_heuristic_1000_env()
    cfg = build_cfg(sys.argv[1:] if argv is None else argv)
    _copy_data_config(cfg)

    env = dclilyenv.NormalizedActions(gym.make(cfg["env_name"]))
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env

    env_time_num = int(getattr(base_env, "TIME_NUM", 26))
    if env_time_num <= 0:
        env_time_num = 26
    cfg["log_time_num"] = env_time_num
    if int(cfg.get("max_steps", env_time_num)) > env_time_num:
        cfg["max_steps"] = env_time_num

    agent = HeuristicAgent(env.action_space.shape[0])
    run_dir = build_outputs_dir("heuristic", cfg["train_eps"], datetime.now().strftime("%Y%m%d-%H%M%S"))
    res_dic = dclilytrain.train(cfg, base_env, agent)
    _save_results(cfg, run_dir, res_dic)
    return run_dir


if __name__ == "__main__":
    main()
