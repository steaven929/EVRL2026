"""Reward-component calculation and export helpers."""

import csv
from pathlib import Path

import numpy as np


REWARD_COMPONENT_KEYS = (
    "service_revenue",
    "relocation_cost",
    "NOR",
    "P_soc",
    "P_svr",
    "B_chg",
    "B_soc",
    "PFT",
    "environment_reward",
    "scaled_training_reward",
)


def format_reward_breakdown(component_totals, reward_scale):
    """Format the episode reward and paper-profit identities for console logs."""
    values = {key: float(component_totals[key]) for key in REWARD_COMPONENT_KEYS}
    reward_scale = float(reward_scale)
    return (
        "NOR = "
        f"service_revenue({values['service_revenue']:.2f}) - "
        f"relocation_cost({values['relocation_cost']:.2f}) = {values['NOR']:.2f}",
        "PFT = "
        f"NOR({values['NOR']:.2f}) - P_soc({values['P_soc']:.2f}) - "
        f"P_svr({values['P_svr']:.2f}) + B_chg({values['B_chg']:.2f}) + "
        f"B_soc({values['B_soc']:.2f}) = {values['PFT']:.2f}",
        "scaled_training_reward = "
        f"environment_reward({values['environment_reward']:.2f}) / "
        f"reward_scale({reward_scale:.2f}) = {values['scaled_training_reward']:.6f}",
        "reward_minus_PFT = "
        f"environment_reward({values['environment_reward']:.2f}) - "
        f"PFT({values['PFT']:.2f}) = "
        f"{values['environment_reward'] - values['PFT']:.2f}",
    )


def build_reward_components(
    *, service_revenue, relocation_cost, p_soc, p_svr, b_chg, b_soc,
    environment_reward, reward_scale,
):
    """Return paper metrics and the two reward values using explicit units."""
    reward_scale = float(reward_scale)
    if reward_scale <= 0:
        raise ValueError("reward_scale must be positive")
    service_revenue = float(service_revenue)
    relocation_cost = float(relocation_cost)
    p_soc = float(p_soc)
    p_svr = float(p_svr)
    b_chg = float(b_chg)
    b_soc = float(b_soc)
    environment_reward = float(environment_reward)
    nor = service_revenue - relocation_cost
    pft = nor - p_soc - p_svr + b_chg + b_soc
    return {
        "service_revenue": service_revenue,
        "relocation_cost": relocation_cost,
        "NOR": nor,
        "P_soc": p_soc,
        "P_svr": p_svr,
        "B_chg": b_chg,
        "B_soc": b_soc,
        "PFT": pft,
        "environment_reward": environment_reward,
        "scaled_training_reward": environment_reward / reward_scale,
    }


def write_reward_components_csv(output_path, component_arrays):
    """Write [episode, timestep] component arrays to one tidy CSV file."""
    arrays = {key: np.asarray(component_arrays[key], dtype=float) for key in REWARD_COMPONENT_KEYS}
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"reward component arrays must share one shape, got {sorted(shapes)}")
    shape = next(iter(shapes))
    if len(shape) != 2:
        raise ValueError(f"reward component arrays must be 2-D, got {shape}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("episode", "timestep", *REWARD_COMPONENT_KEYS))
        for episode_idx in range(shape[0]):
            for timestep_idx in range(shape[1]):
                writer.writerow((episode_idx + 1, timestep_idx + 1, *(
                    arrays[key][episode_idx, timestep_idx] for key in REWARD_COMPONENT_KEYS
                )))


def write_reward_formula_summary(output_path, component_arrays, reward_scale):
    """Write one auditable reward/PFT formula row for each training episode."""
    arrays = {key: np.asarray(component_arrays[key], dtype=float) for key in REWARD_COMPONENT_KEYS}
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"reward component arrays must share one shape, got {sorted(shapes)}")
    shape = next(iter(shapes))
    if len(shape) != 2:
        raise ValueError(f"reward component arrays must be 2-D, got {shape}")
    reward_scale = float(reward_scale)
    if reward_scale <= 0:
        raise ValueError("reward_scale must be positive")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "episode", *REWARD_COMPONENT_KEYS, "reward_scale", "reward_minus_PFT",
        "pft_identity_error", "reward_scaling_error",
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for episode_idx in range(shape[0]):
            totals = {}
            for key, array in arrays.items():
                values = array[episode_idx]
                finite = values[np.isfinite(values)]
                totals[key] = float(np.sum(finite)) if finite.size else np.nan
            expected_pft = (
                totals["NOR"] - totals["P_soc"] - totals["P_svr"]
                + totals["B_chg"] + totals["B_soc"]
            )
            expected_scaled_reward = totals["environment_reward"] / reward_scale
            writer.writerow({
                "episode": episode_idx + 1,
                **totals,
                "reward_scale": reward_scale,
                "reward_minus_PFT": totals["environment_reward"] - totals["PFT"],
                "pft_identity_error": totals["PFT"] - expected_pft,
                "reward_scaling_error": (
                    totals["scaled_training_reward"] - expected_scaled_reward
                ),
            })
