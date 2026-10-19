# -*- coding: utf-8 -*-
"""MARL no-optim entrypoint with MARL-compatible artifact output."""

import datetime
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gym.envs.registration import register, registry

from core.environment import env_agent_config
from core.output_paths import result_run_directory
from core.run_provenance import write_run_provenance
from core.marl_training import marl_test, marl_train
from core.reward_components import REWARD_COMPONENT_KEYS, write_reward_components_csv
from core.util import get_args, plot_critic_losses, plot_rewards, smooth


NOOPTIM_ENV_ID = "dclily_nooptim"
NOOPTIM_ENTRY_POINT = "specific.dclily_elec_nooptim:dclily"


def register_nooptim_env():
    if NOOPTIM_ENV_ID not in registry:
        register(id=NOOPTIM_ENV_ID, entry_point=NOOPTIM_ENTRY_POINT)


def build_cfg():
    cfg = get_args()
    cfg["env_name"] = NOOPTIM_ENV_ID
    cfg["marl_mode"] = "multi"
    cfg["n_agents"] = 50
    cfg["algo_name"] = "MARL_NOOPTIM"
    cfg["reward_mode"] = "shaping"
    return cfg


def build_output_path(train_eps, timestamp):
    return result_run_directory(f"marl_nooptim_{train_eps}ep_{timestamp}")


def prepare_output_path(train_eps, timestamp):
    path = build_output_path(train_eps, timestamp)
    os.makedirs(path, exist_ok=True)
    print(f"Output directory: {path}")
    return path


def build_training_path(path):
    path_str = os.fspath(path)
    if not path_str.endswith((os.sep, "/")):
        path_str += os.sep
    return path_str


def apply_nooptim_reward_weights(env):
    base_env = getattr(env, "unwrapped", env)
    reward_weights = getattr(base_env, "reward_weights", None)
    if not isinstance(reward_weights, dict):
        return
    reward_weights["charging_bonus"] = 0.5
    reward_weights["charging_maintenance"] = 0.5


EPISODE_HISTORY_KEYS = [
    "critic_losses",
    "actor_losses",
    "rewards",
    "raw_rewards",
    "ma_rewards",
    "rld_history",
    "svc_history",
    "srd_history",
    "charging_efficiency_history",
    "vehicle_utilization_history",
    "fleet_health_history",
    "zone_coordination_history",
    "response_readiness_history",
    "charger_occupancy_rate_history",
    "charger_queue_delay_history",
    "energy_not_served_history",
    "safety_soc_violation_count_history",
    "charge_shift_success_rate_history",
    "weighted_service_rate_history",
    "effective_service_rate_history",
    "service_quality_index_history",
    "dispatch_efficiency_history",
    "effective_dispatch_rate_history",
    "dispatch_roi_history",
    *(f"{key}_history" for key in REWARD_COMPONENT_KEYS),
]

TIMESTEP_KEYS = [
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
    *(f"{key}_by_timestep" for key in REWARD_COMPONENT_KEYS),
]


def _save_available_arrays(path, res_dic, keys):
    for key in keys:
        if key in res_dic:
            np.save(os.path.join(path, f"{key}.npy"), res_dic[key])


def _save_training_metadata(path, res_dic, cfg):
    metadata = {
        "algo_name": cfg.get("algo_name", "MARL_NOOPTIM"),
        "seed": cfg["seed"],
        "train_eps": cfg["train_eps"],
        "n_agents": cfg.get("n_agents", 50),
        "batch_size": cfg["batch_size"],
        "actor_lr": cfg.get("actor_lr", "N/A"),
        "critic_lr": cfg.get("critic_lr", "N/A"),
        "gamma": cfg["gamma"],
        "tau": cfg.get("tau", "N/A"),
        "memory_capacity": cfg["memory_capacity"],
        "obs_dim": cfg.get("obs_dim", 6),
        "action_dim": cfg.get("action_dim", 3),
        "global_state_dim": cfg.get("global_state_dim", 251),
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "total_episodes": len(res_dic["rewards"]),
        "mean_reward": float(np.mean(res_dic["rewards"])),
        "max_reward": float(np.max(res_dic["rewards"])),
        "final_reward": float(res_dic["rewards"][-1]),
        "std_reward": float(np.std(res_dic["rewards"])),
        "mean_zone_coordination": float(np.mean(res_dic.get("zone_coordination_history", [0]))),
        "mean_response_readiness": float(np.mean(res_dic.get("response_readiness_history", [0]))),
        "mean_fleet_health": float(np.mean(res_dic.get("fleet_health_history", [0]))),
    }
    with open(os.path.join(path, "training_metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def _plot_final_metrics(path, res_dic):
    fig, axes = plt.subplots(8, 1, figsize=(10, 20))

    axes[0].plot(res_dic["critic_losses"], label="Critic Loss", color="blue", linewidth=1.5)
    axes[0].set_xlabel("Update Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Critic Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(res_dic["actor_losses"], label="Actor Loss (-Q)", color="red", linewidth=1.5)
    axes[1].set_xlabel("Update Step")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Actor Loss (-Q Value, should decrease)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].text(0.5, 0.5, "Q Value tracking not available in MARL", ha="center", va="center", transform=axes[2].transAxes)
    axes[2].set_title("Q Value (N/A for MARL)")

    axes[3].text(0.5, 0.5, "Target Q tracking not available in MARL", ha="center", va="center", transform=axes[3].transAxes)
    axes[3].set_title("Target Q Value (N/A for MARL)")

    if "rld_history" in res_dic and len(res_dic["rld_history"]) > 0:
        axes[4].plot(res_dic["rld_history"], label="RLD", color="green", linewidth=1.5)
        axes[4].set_xlabel("Step")
        axes[4].set_ylabel("RLD")
        axes[4].set_title("Repositioning Loss Distance (RLD)")
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)

    if "svc_history" in res_dic and len(res_dic["svc_history"]) > 0:
        axes[5].plot(res_dic["svc_history"], label="SVC", color="purple", linewidth=1.5)
        axes[5].set_xlabel("Step")
        axes[5].set_ylabel("SVC")
        axes[5].set_title("Service Rate (SVC)")
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)

    if "srd_history" in res_dic and len(res_dic["srd_history"]) > 0:
        axes[6].plot(res_dic["srd_history"], label="SRD", color="orange", linewidth=1.5)
        axes[6].set_xlabel("Step")
        axes[6].set_ylabel("SRD")
        axes[6].set_title("Service Reward Ratio (SRD)")
        axes[6].legend()
        axes[6].grid(True, alpha=0.3)

    axes[7].plot(res_dic["rewards"], label="Reward", color="gray", alpha=0.3, linewidth=1.0)
    axes[7].plot(smooth(res_dic["rewards"]), label="Smoothed", color="blue", linewidth=2.0)
    axes[7].set_xlabel("Episode")
    axes[7].set_ylabel("Reward")
    axes[7].set_title("Episode Reward")
    axes[7].legend()
    axes[7].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(path, "final_metrics.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_marl_comprehensive_metrics(path, res_dic):
    fig, axes = plt.subplots(6, 2, figsize=(16, 24))
    axes = axes.flatten()

    axes[0].plot(res_dic["critic_losses"], label="Critic Loss", color="blue", linewidth=1.5)
    axes[0].set_xlabel("Update Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Critic Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(res_dic["actor_losses"], label="Actor Loss (-Q)", color="red", linewidth=1.5)
    axes[1].set_xlabel("Update Step")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Actor Loss (-Q Value, should decrease)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(res_dic["rewards"], label="Episode Reward", color="green", linewidth=1.5, alpha=0.5)
    axes[2].plot(res_dic["ma_rewards"], label="MA Reward", color="darkgreen", linewidth=2.0)
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Reward")
    axes[2].set_title("Episode Reward (Scaled)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(res_dic["svc_history"], label="SVC (Mean)", color="purple", linewidth=1.5)
    axes[3].set_xlabel("Episode")
    axes[3].set_ylabel("Service Rate (%)")
    axes[3].set_title("Service Rate (SVC) - Traditional Metric")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    if "charging_efficiency_history" in res_dic:
        axes[4].plot(res_dic["charging_efficiency_history"], label="Charging Efficiency", color="#FF6B6B", linewidth=2.0)
        axes[4].set_xlabel("Episode")
        axes[4].set_ylabel("Efficiency (0-1)")
        axes[4].set_title("Charging Efficiency (Effective/Total) - Higher is Better")
        axes[4].axhline(y=0.9, color="green", linestyle="--", alpha=0.5, label="Target: 0.9")
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)
        axes[4].set_ylim([0, 1.05])

    if "vehicle_utilization_history" in res_dic:
        axes[5].plot(res_dic["vehicle_utilization_history"], label="Vehicle Utilization", color="#4ECDC4", linewidth=2.0)
        axes[5].set_xlabel("Episode")
        axes[5].set_ylabel("Utilization (0-1)")
        axes[5].set_title("Vehicle Utilization (Active/Total) - Higher is Better")
        axes[5].axhline(y=0.5, color="green", linestyle="--", alpha=0.5, label="Target: 0.5")
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)
        axes[5].set_ylim([0, 1.05])

    if "fleet_health_history" in res_dic:
        axes[6].plot(res_dic["fleet_health_history"], label="Fleet Health", color="#95E1D3", linewidth=2.0)
        axes[6].set_xlabel("Episode")
        axes[6].set_ylabel("Health (0-1)")
        axes[6].set_title("Fleet Health (Avg SOC/Capacity) - Higher is Better")
        axes[6].axhline(y=0.6, color="green", linestyle="--", alpha=0.5, label="Target: 0.6")
        axes[6].legend()
        axes[6].grid(True, alpha=0.3)
        axes[6].set_ylim([0, 1.05])

    if "zone_coordination_history" in res_dic:
        axes[7].plot(res_dic["zone_coordination_history"], label="Zone Coordination", color="#F38181", linewidth=2.0)
        axes[7].set_xlabel("Episode")
        axes[7].set_ylabel("Coordination (0-1)")
        axes[7].set_title("Zone Coordination (Vehicle Distribution Balance) - Higher is Better")
        axes[7].axhline(y=0.7, color="green", linestyle="--", alpha=0.5, label="Target: 0.7")
        axes[7].legend()
        axes[7].grid(True, alpha=0.3)
        axes[7].set_ylim([0, 1.05])

    if "response_readiness_history" in res_dic:
        axes[8].plot(res_dic["response_readiness_history"], label="Response Readiness", color="#AA96DA", linewidth=2.0)
        axes[8].set_xlabel("Episode")
        axes[8].set_ylabel("Readiness (0-1)")
        axes[8].set_title("Response Readiness (High-SOC Idle Vehicles) - Higher is Better")
        axes[8].axhline(y=0.4, color="green", linestyle="--", alpha=0.5, label="Target: 0.4")
        axes[8].legend()
        axes[8].grid(True, alpha=0.3)
        axes[8].set_ylim([0, 1.05])

    axes[9].plot(res_dic["rld_history"], label="RLD", color="orange", linewidth=1.5)
    axes[9].set_xlabel("Episode")
    axes[9].set_ylabel("Distance")
    axes[9].set_title("Relocation Distance (RLD)")
    axes[9].legend()
    axes[9].grid(True, alpha=0.3)

    last_n = min(10, len(res_dic["rewards"]))
    if all(key in res_dic for key in ["charging_efficiency_history", "vehicle_utilization_history", "fleet_health_history", "zone_coordination_history", "response_readiness_history"]):
        metrics_avg = {
            "Charging\nEfficiency": np.mean(res_dic["charging_efficiency_history"][-last_n:]),
            "Vehicle\nUtilization": np.mean(res_dic["vehicle_utilization_history"][-last_n:]),
            "Fleet\nHealth": np.mean(res_dic["fleet_health_history"][-last_n:]),
            "Zone\nCoordination": np.mean(res_dic["zone_coordination_history"][-last_n:]),
            "Response\nReadiness": np.mean(res_dic["response_readiness_history"][-last_n:]),
        }
        axes[10].bar(range(len(metrics_avg)), list(metrics_avg.values()), color=["#FF6B6B", "#4ECDC4", "#95E1D3", "#F38181", "#AA96DA"])
        axes[10].set_xticks(range(len(metrics_avg)))
        axes[10].set_xticklabels(list(metrics_avg.keys()), rotation=15, ha="right")
        axes[10].set_ylabel("Score (0-1)")
        axes[10].set_title(f"MARL Overall Performance (Last {last_n} Episodes Avg)")
        axes[10].axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Baseline: 0.5")
        axes[10].legend()
        axes[10].grid(True, alpha=0.3, axis="y")
        axes[10].set_ylim([0, 1.05])
        for idx, value in enumerate(metrics_avg.values()):
            axes[10].text(idx, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    axes[11].axis("off")
    summary_text = f"""
    MARL No-Optim Training Summary
    Episodes: {len(res_dic["rewards"])}
    Final Reward: {res_dic["rewards"][-1]:.2f}
    Best Reward: {max(res_dic["rewards"]):.2f}
    Service Rate: {np.mean(res_dic["svc_history"][-last_n:]):.2f}%
    RLD: {np.mean(res_dic["rld_history"][-last_n:]):.2f}
    Charging Efficiency: {np.mean(res_dic.get("charging_efficiency_history", [0])[-last_n:]):.3f}
    Vehicle Utilization: {np.mean(res_dic.get("vehicle_utilization_history", [0])[-last_n:]):.3f}
    Fleet Health: {np.mean(res_dic.get("fleet_health_history", [0])[-last_n:]):.3f}
    Zone Coordination: {np.mean(res_dic.get("zone_coordination_history", [0])[-last_n:]):.3f}
    Response Readiness: {np.mean(res_dic.get("response_readiness_history", [0])[-last_n:]):.3f}
    """
    axes[11].text(0.1, 0.5, summary_text, fontsize=10, family="monospace", verticalalignment="center")

    plt.tight_layout()
    plt.savefig(os.path.join(path, "marl_comprehensive_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_metrics_comparison(path, res_dic):
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.flatten()

    axes[0].plot(res_dic["svc_history"], label="SVC (Old)", color="gray", linewidth=2.0, alpha=0.6)
    if "weighted_service_rate_history" in res_dic:
        axes[0].plot(res_dic["weighted_service_rate_history"], label="Weighted Service Rate (New)", color="#FF6B6B", linewidth=2.0)
    if "effective_service_rate_history" in res_dic:
        axes[0].plot(res_dic["effective_service_rate_history"], label="Effective Service Rate (New)", color="#4ECDC4", linewidth=2.0)
    if "service_quality_index_history" in res_dic:
        axes[0].plot(res_dic["service_quality_index_history"], label="Service Quality Index (New)", color="#95E1D3", linewidth=2.0)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Rate (0-1)")
    axes[0].set_title("SVC Metrics Comparison: Old vs New")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(res_dic["rld_history"], label="RLD (Old)", color="gray", linewidth=2.0, alpha=0.6)
    ax1_twin = axes[1].twinx()
    if "dispatch_efficiency_history" in res_dic:
        ax1_twin.plot(res_dic["dispatch_efficiency_history"], label="Dispatch Efficiency (New)", color="#F38181", linewidth=2.0)
    if "effective_dispatch_rate_history" in res_dic:
        ax1_twin.plot(res_dic["effective_dispatch_rate_history"], label="Effective Dispatch Rate (New)", color="#AA96DA", linewidth=2.0)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("RLD Distance (Old)", color="gray")
    ax1_twin.set_ylabel("Efficiency/Rate (New)", color="#F38181")
    axes[1].set_title("RLD Metrics Comparison: Old vs New")
    axes[1].legend(loc="upper left")
    ax1_twin.legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    if "weighted_service_rate_history" in res_dic:
        axes[2].plot(res_dic["weighted_service_rate_history"], label="Weighted Service Rate", color="#FF6B6B", linewidth=2.0)
        axes[2].fill_between(range(len(res_dic["weighted_service_rate_history"])), res_dic["weighted_service_rate_history"], alpha=0.3, color="#FF6B6B")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Weighted Service Rate")
        axes[2].set_title("Weighted Service Rate (Distance/Value Weighted)")
        axes[2].axhline(y=0.5, color="green", linestyle="--", alpha=0.5, label="Target: 0.5")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 1.05])

    if "service_quality_index_history" in res_dic:
        axes[3].plot(res_dic["service_quality_index_history"], label="Service Quality Index", color="#95E1D3", linewidth=2.0)
        axes[3].fill_between(range(len(res_dic["service_quality_index_history"])), res_dic["service_quality_index_history"], alpha=0.3, color="#95E1D3")
        axes[3].set_xlabel("Episode")
        axes[3].set_ylabel("Service Quality Index")
        axes[3].set_title("Service Quality Index (Rate+SOC+Readiness)")
        axes[3].axhline(y=0.6, color="green", linestyle="--", alpha=0.5, label="Target: 0.6")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylim([0, 1.05])

    if "dispatch_efficiency_history" in res_dic:
        axes[4].plot(res_dic["dispatch_efficiency_history"], label="Dispatch Efficiency", color="#F38181", linewidth=2.0)
        axes[4].fill_between(range(len(res_dic["dispatch_efficiency_history"])), res_dic["dispatch_efficiency_history"], alpha=0.3, color="#F38181")
        axes[4].set_xlabel("Episode")
        axes[4].set_ylabel("Dispatch Efficiency (orders/distance)")
        axes[4].set_title("Dispatch Efficiency (Orders/Distance)")
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)

    if "dispatch_roi_history" in res_dic:
        axes[5].plot(res_dic["dispatch_roi_history"], label="Dispatch ROI", color="#AA96DA", linewidth=2.0)
        axes[5].fill_between(range(len(res_dic["dispatch_roi_history"])), res_dic["dispatch_roi_history"], alpha=0.3, color="#AA96DA")
        axes[5].axhline(y=0, color="red", linestyle="-", alpha=0.5, label="Break-even")
        axes[5].set_xlabel("Episode")
        axes[5].set_ylabel("Dispatch ROI")
        axes[5].set_title("Dispatch ROI (Return on Investment)")
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(path, "metrics_comparison_old_vs_new.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_charging_metrics(path, res_dic):
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    axes = axes.flatten()

    if "charger_occupancy_rate_history" in res_dic:
        axes[0].plot(res_dic["charger_occupancy_rate_history"], color="#1f77b4", linewidth=2.0, label="Charger Occupancy Rate")
        axes[0].set_title("Charger Occupancy Rate")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Rate (0-1)")
        axes[0].set_ylim([0, 1.05])
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

    if "charger_queue_delay_history" in res_dic:
        axes[1].plot(res_dic["charger_queue_delay_history"], color="#ff7f0e", linewidth=2.0, label="Charger Queue Delay")
        axes[1].set_title("Charger Queue Delay (minutes, proxy)")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Minutes")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

    if "energy_not_served_history" in res_dic:
        axes[2].plot(res_dic["energy_not_served_history"], color="#d62728", linewidth=2.0, label="Energy Not Served")
        axes[2].set_title("Energy Not Served (orders, proxy)")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Orders")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

    if "safety_soc_violation_count_history" in res_dic:
        axes[3].plot(res_dic["safety_soc_violation_count_history"], color="#9467bd", linewidth=2.0, label="Safety SOC Violation Count")
        axes[3].set_title("Safety SOC Violation Count (SOC < 20%)")
        axes[3].set_xlabel("Episode")
        axes[3].set_ylabel("Vehicle Count")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()

    if "charge_shift_success_rate_history" in res_dic:
        axes[4].plot(res_dic["charge_shift_success_rate_history"], color="#2ca02c", linewidth=2.0, label="Charge Shift Success Rate")
        axes[4].set_title("Charge Shift Success Rate")
        axes[4].set_xlabel("Episode")
        axes[4].set_ylabel("Rate (0-1)")
        axes[4].set_ylim([0, 1.05])
        axes[4].grid(True, alpha=0.3)
        axes[4].legend()

    axes[5].axis("off")
    summary = f"""
    Charging Scheduling Metrics Summary (Last 10 Eps Avg)
    -----------------------------------------------------
    Charger Occupancy Rate: {np.nanmean(res_dic.get("charger_occupancy_rate_history", [np.nan])[-10:]):.3f}
    Charger Queue Delay: {np.nanmean(res_dic.get("charger_queue_delay_history", [np.nan])[-10:]):.3f}
    Energy Not Served: {np.nanmean(res_dic.get("energy_not_served_history", [np.nan])[-10:]):.3f}
    Safety SOC Violations: {np.nanmean(res_dic.get("safety_soc_violation_count_history", [np.nan])[-10:]):.3f}
    Charge Shift Success Rate: {np.nanmean(res_dic.get("charge_shift_success_rate_history", [np.nan])[-10:]):.3f}
    """
    axes[5].text(0.05, 0.5, summary, fontsize=10, family="monospace", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(path, "charging_scheduling_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_training_artifacts(path, res_dic, cfg):
    write_run_provenance(path, cfg)
    _save_available_arrays(path, res_dic, EPISODE_HISTORY_KEYS)
    _save_available_arrays(path, res_dic, TIMESTEP_KEYS)
    write_reward_components_csv(
        os.path.join(path, 'reward_components_by_timestep.csv'),
        {key: res_dic[f'{key}_by_timestep'] for key in REWARD_COMPONENT_KEYS},
    )
    _save_training_metadata(path, res_dic, cfg)
    plot_rewards(res_dic["rewards"], cfg, path=os.path.join(path, "train_rewards.png"), tag="train")
    plt.savefig(os.path.join(path, "train_rewards.png"), dpi=300, bbox_inches="tight")
    plt.close()
    plot_critic_losses(res_dic["critic_losses"], None, tag="train", save_path=os.path.join(path, "critic_loss_curve.png"))
    _plot_final_metrics(path, res_dic)
    _plot_marl_comprehensive_metrics(path, res_dic)
    _plot_metrics_comparison(path, res_dic)
    _plot_charging_metrics(path, res_dic)


def main():
    register_nooptim_env()
    cfg = build_cfg()
    env, agent = env_agent_config(cfg)
    apply_nooptim_reward_weights(env)

    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_eps = cfg.get("train_eps", 0)
    path = prepare_output_path(train_eps, curr_time)
    train_path = build_training_path(path)

    if cfg.get("run_mode") == "test":
        return marl_test(env, agent, cfg, path=train_path)

    cfg["curr_time"] = curr_time
    cfg.setdefault("algo_name", "MARL_NOOPTIM")
    res_dic = marl_train(env, agent, cfg, path=train_path)
    save_training_artifacts(path, res_dic, cfg)
    agent.save_model(os.path.join(path, "marl_model_final.pth"))
    marl_test(env, agent, cfg, path=train_path)
    return res_dic


if __name__ == "__main__":
    main()



