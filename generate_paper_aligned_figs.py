from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class AlgoData:
    name: str
    seeds: list[int]
    rewards: list[np.ndarray]


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "output2" / "compare_baseline2"
RUN_150 = BASE / "run_20260304-012548"
RUN_10 = BASE / "run_20260304-173148"
MARL_MECH = ROOT / "output2" / "marl_20260305-204126"


def ensure_results_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "results" / f"paper_aligned_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def read_run_meta(run_dir: Path) -> dict:
    meta_file = run_dir / "run_meta.json"
    if not meta_file.exists():
        return {}
    return json.loads(meta_file.read_text(encoding="utf-8"))


def load_algo_group(run_dir: Path, prefix: str, label: str, keep_failed: bool = False) -> AlgoData:
    rewards: list[np.ndarray] = []
    seeds: list[int] = []

    for sub in sorted(run_dir.glob(f"{prefix}_seed*")):
        if not sub.is_dir():
            continue
        meta = read_run_meta(sub)
        rc = meta.get("returncode", 0)
        rewards_file = sub / "rewards.npy"
        if not rewards_file.exists():
            continue
        if (not keep_failed) and rc != 0:
            continue
        rewards.append(np.load(rewards_file).astype(float))
        seed = int(meta.get("seed", -1))
        seeds.append(seed)

    return AlgoData(name=label, seeds=seeds, rewards=rewards)


def align_rewards(arrays: list[np.ndarray]) -> np.ndarray:
    min_len = min(len(a) for a in arrays)
    return np.stack([a[:min_len] for a in arrays], axis=0)


def plot_learning_curve(datas: list[AlgoData], title: str, output_file: Path) -> dict:
    valid = [d for d in datas if d.rewards]
    all_arrays = [arr for d in valid for arr in d.rewards]
    aligned_len = min(len(a) for a in all_arrays)

    plt.figure(figsize=(10, 6))
    x = np.arange(1, aligned_len + 1)
    summary = {"aligned_episodes": int(aligned_len), "algorithms": {}}

    for d in valid:
        stack = np.stack([r[:aligned_len] for r in d.rewards], axis=0)
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        plt.plot(x, mean, label=f"{d.name} (n={len(d.rewards)})", linewidth=2)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
        summary["algorithms"][d.name] = {
            "n_seeds": len(d.rewards),
            "seeds": d.seeds,
            "mean_reward": float(stack.mean()),
            "final10_mean": float(stack[:, -10:].mean()) if aligned_len >= 10 else float(stack.mean()),
        }

    plt.xlabel("Training Episodes (Aligned)")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close()
    return summary


def plot_auc_and_final10(datas: list[AlgoData], title: str, output_file: Path) -> dict:
    valid = [d for d in datas if d.rewards]
    all_arrays = [arr for d in valid for arr in d.rewards]
    aligned_len = min(len(a) for a in all_arrays)

    labels = []
    auc_means = []
    auc_stds = []
    f10_means = []
    f10_stds = []
    summary = {"aligned_episodes": int(aligned_len), "algorithms": {}}

    for d in valid:
        stack = np.stack([r[:aligned_len] for r in d.rewards], axis=0)
        auc = stack.mean(axis=1)
        f10 = stack[:, -10:].mean(axis=1) if aligned_len >= 10 else stack.mean(axis=1)
        labels.append(d.name)
        auc_means.append(float(auc.mean()))
        auc_stds.append(float(auc.std()))
        f10_means.append(float(f10.mean()))
        f10_stds.append(float(f10.std()))
        summary["algorithms"][d.name] = {
            "n_seeds": len(d.rewards),
            "auc_mean": float(auc.mean()),
            "auc_std": float(auc.std()),
            "final10_mean": float(f10.mean()),
            "final10_std": float(f10.std()),
        }

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(x, auc_means, yerr=auc_stds, capsize=4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title("AUC over Aligned Episodes")
    axes[0].set_ylabel("Mean Reward")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, f10_means, yerr=f10_stds, capsize=4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Final-10 Episodes")
    axes[1].set_ylabel("Mean Reward")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary


def plot_success_rate(base_dir: Path, output_file: Path) -> dict:
    stats = {
        "MARL": {"ok": 0, "total": 0},
        "SAC": {"ok": 0, "total": 0},
        "TD3": {"ok": 0, "total": 0},
        "DDPG": {"ok": 0, "total": 0},
    }

    for meta_file in base_dir.rglob("run_meta.json"):
        parent = meta_file.parent.name
        algo = None
        if parent.startswith("marl_marl_seed"):
            algo = "MARL"
        elif parent.startswith("single_sac_seed"):
            algo = "SAC"
        elif parent.startswith("single_td3_seed"):
            algo = "TD3"
        elif parent.startswith("single_ddpg_seed"):
            algo = "DDPG"
        if algo is None:
            continue
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        stats[algo]["total"] += 1
        if meta.get("returncode", 1) == 0:
            stats[algo]["ok"] += 1

    algos = list(stats.keys())
    rates = [100.0 * stats[a]["ok"] / stats[a]["total"] if stats[a]["total"] else 0.0 for a in algos]
    totals = [stats[a]["total"] for a in algos]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(algos, rates)
    plt.ylabel("Successful Runs (%)")
    plt.title("Run Success Rate by Algorithm")
    plt.ylim(0, 105)
    plt.grid(axis="y", alpha=0.3)
    for bar, rate, total in zip(bars, rates, totals):
        plt.text(bar.get_x() + bar.get_width() / 2, rate + 2, f"{rate:.1f}%\n(n={total})", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close()
    return stats


def plot_marl_mechanism(run_dir: Path, output_file: Path) -> dict:
    metrics = {
        "Service Quality Index": "service_quality_index_history.npy",
        "Zone Coordination": "zone_coordination_history.npy",
        "Response Readiness": "response_readiness_history.npy",
        "Energy Not Served": "energy_not_served_history.npy",
    }

    loaded = {}
    for k, fn in metrics.items():
        p = run_dir / fn
        if p.exists():
            loaded[k] = np.load(p).astype(float)

    if len(loaded) < 2:
        raise RuntimeError(f"Not enough MARL mechanism metrics in {run_dir}")

    n = len(loaded)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    summary = {}
    for idx, (name, arr) in enumerate(loaded.items()):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        x = np.arange(1, len(arr) + 1)
        ax.plot(x, arr, linewidth=2)
        ax.set_title(name)
        ax.set_xlabel("Training Episodes")
        ax.grid(alpha=0.3)
        summary[name] = {
            "episodes": int(len(arr)),
            "mean": float(np.mean(arr)),
            "final10": float(np.mean(arr[-10:])) if len(arr) >= 10 else float(np.mean(arr)),
        }

    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    fig.suptitle(f"MARL Mechanism Metrics ({run_dir.name})")
    fig.tight_layout()
    fig.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> None:
    out_dir = ensure_results_dir()

    marl_150 = load_algo_group(RUN_150, "marl_marl", "MARL", keep_failed=False)
    sac_150 = load_algo_group(RUN_150, "single_sac", "SAC", keep_failed=False)

    marl_10 = load_algo_group(RUN_10, "marl_marl", "MARL", keep_failed=False)
    sac_10 = load_algo_group(RUN_10, "single_sac", "SAC", keep_failed=False)
    td3_10 = load_algo_group(RUN_10, "single_td3", "TD3", keep_failed=False)

    fig1_summary = plot_learning_curve(
        [marl_150, sac_150],
        "Aligned Learning Curves (150 episodes): MARL vs SAC",
        out_dir / "fig1_learning_curve_150ep_marl_vs_sac.png",
    )

    fig2_summary = plot_auc_and_final10(
        [marl_150, sac_150],
        "Aligned Performance (150 episodes): MARL vs SAC",
        out_dir / "fig2_auc_final10_150ep_marl_vs_sac.png",
    )

    fig3_summary = plot_learning_curve(
        [marl_10, sac_10, td3_10],
        "Aligned Learning Curves (10 episodes): MARL vs SAC vs TD3",
        out_dir / "fig3_learning_curve_10ep_marl_sac_td3.png",
    )

    fig4_summary = plot_success_rate(
        BASE,
        out_dir / "fig4_run_success_rate_by_algorithm.png",
    )

    fig5_summary = plot_marl_mechanism(
        MARL_MECH,
        out_dir / "fig5_marl_mechanism_metrics.png",
    )

    report = {
        "result_dir": str(out_dir),
        "alignment_rule": "All comparison plots are truncated to the shortest available episode length among compared algorithms/seeds.",
        "fig1": fig1_summary,
        "fig2": fig2_summary,
        "fig3": fig3_summary,
        "fig4": fig4_summary,
        "fig5": fig5_summary,
        "source_runs": {
            "run_150": str(RUN_150),
            "run_10": str(RUN_10),
            "marl_mechanism": str(MARL_MECH),
        },
    }

    json_path = out_dir / "analysis_summary.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Paper Figure Analysis (Aligned Episodes)",
        "",
        f"- Output folder: `{out_dir}`",
        "- Episode alignment rule: all compared algorithms are truncated to a unified episode count.",
        "",
        "## Figure Notes",
        "- Fig1: 150-episode aligned MARL vs SAC learning curves (mean ± std over seeds).",
        "- Fig2: 150-episode aligned AUC and Final-10 comparison with error bars.",
        "- Fig3: 10-episode aligned MARL/SAC/TD3 early-training comparison.",
        "- Fig4: Success rate by algorithm from run_meta return codes in compare_baseline2.",
        "- Fig5: MARL mechanism metrics showing coordination and service quality behavior.",
        "",
        "## Key Interpretation",
    ]

    fig2_alg = fig2_summary.get("algorithms", {})
    if "MARL" in fig2_alg and "SAC" in fig2_alg:
        m_auc = fig2_alg["MARL"]["auc_mean"]
        s_auc = fig2_alg["SAC"]["auc_mean"]
        m_f10 = fig2_alg["MARL"]["final10_mean"]
        s_f10 = fig2_alg["SAC"]["final10_mean"]
        lines.append(f"- In 150-episode aligned setting, AUC: MARL={m_auc:.2f}, SAC={s_auc:.2f}.")
        lines.append(f"- In 150-episode aligned setting, Final-10: MARL={m_f10:.2f}, SAC={s_f10:.2f}.")

    fig3_alg = fig3_summary.get("algorithms", {})
    if fig3_alg:
        short_text = ", ".join(
            [f"{k} final10={v['final10_mean']:.2f}" for k, v in fig3_alg.items()]
        )
        lines.append(f"- In 10-episode aligned setting: {short_text}.")

    ddpg_stats = fig4_summary.get("DDPG", {})
    if ddpg_stats:
        lines.append(
            f"- DDPG run success rate is low ({ddpg_stats.get('ok', 0)}/{ddpg_stats.get('total', 0)}), so DDPG should be reported with caveats."
        )

    lines.extend(
        [
            "",
            "## Writing Guidance",
            "- Use aligned-episode curves/tables for fairness whenever x-axis is training episodes.",
            "- Report seed count and successful-run count together.",
            "- Separate MARL-specific mechanism metrics from direct baseline reward comparison.",
        ]
    )

    md_path = out_dir / "analysis_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved figure package to: {out_dir}")
    print(f"- Summary JSON: {json_path}")
    print(f"- Report MD: {md_path}")


if __name__ == "__main__":
    main()
