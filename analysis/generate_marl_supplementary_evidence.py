import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"F:\Claude code\EVRL260207")
SUMMARY = ROOT / "analysis" / "run_scan_summary.json"
OUT_ROOT = ROOT / "results"
GROUPS = ["MARL", "SAC", "TD3", "DDPG_PURE"]


def load_rows():
    rows = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = [r for r in rows if (r.get("train_eps") or -1) >= 100 and r.get("group") in GROUPS]
    dedup = {}
    for r in rows:
        k = (r["group"], r["run_name"])
        score = sum(
            1
            for f in [
                "rewards.npy",
                "raw_rewards.npy",
                "svc_history.npy",
                "rld_history.npy",
                "srd_history.npy",
                "charging_efficiency_history.npy",
            ]
            if os.path.exists(os.path.join(r["dir"], f))
        )
        if k not in dedup or score > dedup[k][0]:
            dedup[k] = (score, r)
    return [v[1] for v in dedup.values()]


def metric_values(rows, group, metric):
    return [r[metric] for r in rows if r["group"] == group and isinstance(r.get(metric), (int, float))]


def set_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )


def fig_shared_distribution(rows, out_file):
    metrics = [
        ("reward_tail10", "Reward (tail-10)", True),
        ("svc_mean", "SVC mean", True),
        ("rld_mean", "RLD mean", False),
        ("srd_mean", "SRD mean", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.flatten()
    colors = {"MARL": "#2a9d8f", "SAC": "#457b9d", "TD3": "#e76f51", "DDPG_PURE": "#6c757d"}

    for i, (metric, title, _) in enumerate(metrics):
        ax = axes[i]
        pos = []
        vals = []
        labels = []
        for j, g in enumerate(GROUPS, start=1):
            v = metric_values(rows, g, metric)
            if not v:
                continue
            pos.append(j)
            vals.append(v)
            labels.append(f"{g}\n(n={len(v)})")

        if vals:
            bp = ax.boxplot(vals, positions=pos, widths=0.5, patch_artist=True, showmeans=True)
            for patch, p in zip(bp["boxes"], pos):
                g = GROUPS[p - 1]
                patch.set_facecolor(colors[g])
                patch.set_alpha(0.35)
            for p, v in zip(pos, vals):
                rng = np.random.default_rng(42 + p)
                jitter = (rng.random(len(v)) - 0.5) * 0.18
                ax.scatter(np.full(len(v), p) + jitter, v, s=14, alpha=0.75, color=colors[GROUPS[p - 1]])
            ax.set_xticks(pos)
            ax.set_xticklabels(labels)

        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Fig A1: Cross-Algorithm Shared Metrics (train_eps >= 100)")
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_coverage_matrix(rows, out_file):
    metric_files = [
        "rewards.npy",
        "raw_rewards.npy",
        "svc_history.npy",
        "rld_history.npy",
        "srd_history.npy",
        "charging_efficiency_history.npy",
        "charger_queue_delay_history.npy",
        "energy_not_served_history.npy",
        "safety_soc_violation_count_history.npy",
        "dispatch_roi_history.npy",
    ]
    mat = np.zeros((len(GROUPS), len(metric_files)), dtype=float)
    n_runs = []

    for i, g in enumerate(GROUPS):
        grows = [r for r in rows if r["group"] == g]
        n = max(1, len(grows))
        n_runs.append(len(grows))
        for j, mf in enumerate(metric_files):
            c = sum(1 for r in grows if (Path(r["dir"]) / mf).exists())
            mat[i, j] = c / n

    fig, ax = plt.subplots(figsize=(13, 3.8))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(GROUPS)))
    ax.set_yticklabels([f"{g} (n={n_runs[i]})" for i, g in enumerate(GROUPS)])
    ax.set_xticks(np.arange(len(metric_files)))
    ax.set_xticklabels([m.replace("_history.npy", "").replace(".npy", "") for m in metric_files], rotation=25, ha="right")

    for i in range(len(GROUPS)):
        for j in range(len(metric_files)):
            ax.text(j, i, f"{mat[i, j]:.0%}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Coverage ratio")
    ax.set_title("Fig A2: Metric Coverage Matrix")
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_efficiency_service_scatter(rows, out_file):
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    colors = {"MARL": "#2a9d8f", "SAC": "#457b9d", "TD3": "#e76f51", "DDPG_PURE": "#6c757d"}
    markers = {"MARL": "o", "SAC": "s", "TD3": "^", "DDPG_PURE": "D"}

    for g in GROUPS:
        x = [r["rld_mean"] for r in rows if r["group"] == g and isinstance(r.get("rld_mean"), (int, float)) and isinstance(r.get("svc_mean"), (int, float))]
        y = [r["svc_mean"] for r in rows if r["group"] == g and isinstance(r.get("rld_mean"), (int, float)) and isinstance(r.get("svc_mean"), (int, float))]
        if not x:
            continue
        ax.scatter(x, y, s=40, alpha=0.8, label=f"{g} (n={len(x)})", color=colors[g], marker=markers[g])

        mx, my = float(np.median(x)), float(np.median(y))
        ax.scatter([mx], [my], s=100, color=colors[g], marker="X", edgecolors="black", linewidths=0.6)

    ax.set_xlabel("RLD mean (lower better)")
    ax.set_ylabel("SVC mean (higher better)")
    ax.set_title("Fig A3: Efficiency-Service Tradeoff (Pareto-style)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _last10(path):
    if not path.exists():
        return None
    arr = np.asarray(np.load(path), dtype=float)
    if arr.size == 0:
        return None
    return float(np.mean(arr[-min(10, arr.size) :]))


def fig_marl_charging_mechanism(rows, out_file):
    marl_rows = [r for r in rows if r["group"] == "MARL"]
    metrics = [
        ("charging_efficiency_history.npy", "Charging Eff.", True),
        ("charger_queue_delay_history.npy", "Queue Delay", False),
        ("energy_not_served_history.npy", "Energy Not Served", False),
        ("safety_soc_violation_count_history.npy", "SOC Violations", False),
        ("dispatch_efficiency_history.npy", "Dispatch Eff.", True),
        ("dispatch_roi_history.npy", "Dispatch ROI", True),
    ]

    vals = []
    labels = []
    dirs = []
    for mf, label, higher in metrics:
        v = []
        for r in marl_rows:
            x = _last10(Path(r["dir"]) / mf)
            if isinstance(x, float) and np.isfinite(x):
                v.append(x)
        vals.append(v)
        labels.append(label + (" (up)" if higher else " (down)"))
        dirs.append(higher)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.5))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i >= len(vals):
            ax.axis("off")
            continue
        v = vals[i]
        if not v:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(labels[i])
            continue
        ax.boxplot([v], positions=[1], widths=0.5, patch_artist=True, showmeans=True)
        rng = np.random.default_rng(100 + i)
        jitter = (rng.random(len(v)) - 0.5) * 0.2
        ax.scatter(np.ones(len(v)) + jitter, v, s=16, alpha=0.75, color="#2a9d8f")
        ax.set_xticks([1])
        ax.set_xticklabels([f"MARL\n(n={len(v)})"])
        ax.set_title(labels[i])
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Fig A4: MARL Charging-Scheduling Mechanism Metrics")
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_report(rows, out_dir):
    n_map = {g: len([r for r in rows if r["group"] == g]) for g in GROUPS}

    def med(g, m):
        v = metric_values(rows, g, m)
        return None if not v else float(np.median(np.asarray(v, dtype=float)))

    lines = []
    lines.append("# Supplementary Evidence Analysis\n")
    lines.append("\n## Data Coverage\n")
    lines.append("- Filter: `train_eps >= 100`\n")
    lines.append("- Run counts: " + ", ".join([f"{g}={n_map[g]}" for g in GROUPS]) + "\n")

    lines.append("\n## Figure Captions (paper-ready)\n")
    lines.append("- **Fig A1**: Shared metrics across algorithms (box + seed points). This is the primary cross-algorithm evidence panel because metrics are broadly comparable.\n")
    lines.append("- **Fig A2**: Metric coverage matrix. It explicitly shows which claims are cross-algorithm vs MARL-only mechanism claims.\n")
    lines.append("- **Fig A3**: RLD-SVC tradeoff scatter. Points closer to lower-left-high-service region indicate better dispatch efficiency under service constraints.\n")
    lines.append("- **Fig A4**: MARL charging mechanism metrics. This supports the charging-scheduling story (efficiency, queue, safety, energy-not-served, ROI).\n")

    lines.append("\n## Suggested Claim Wording\n")
    lines.append("- Primary claim: MARL improves dispatch efficiency (lower RLD) while maintaining competitive service levels on shared metrics.\n")
    lines.append("- Secondary claim: MARL exhibits favorable charging-scheduling mechanisms (queue delay/safety/energy-not-served related indicators) in runs with detailed charging logs.\n")
    lines.append("- Limitation statement: Sample sizes and metric coverage are imbalanced; charging-specific cross-algorithm claims should be phrased cautiously.\n")

    lines.append("\n## Quick Median Snapshot (shared metrics)\n")
    for g in GROUPS:
        lines.append(
            f"- {g}: median SVC={med(g,'svc_mean')}, median RLD={med(g,'rld_mean')}, median SRD={med(g,'srd_mean')}, median RewardTail10={med(g,'reward_tail10')}\n"
        )

    (out_dir / "analysis.md").write_text("".join(lines), encoding="utf-8")


def main():
    set_style()
    rows = load_rows()

    out_dir = OUT_ROOT / f"marl_supplementary_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_shared_distribution(rows, out_dir / "figA1_shared_distribution.png")
    fig_coverage_matrix(rows, out_dir / "figA2_metric_coverage_matrix.png")
    fig_efficiency_service_scatter(rows, out_dir / "figA3_rld_svc_scatter.png")
    fig_marl_charging_mechanism(rows, out_dir / "figA4_marl_charging_mechanism.png")

    (out_dir / "rows_used.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    build_report(rows, out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
