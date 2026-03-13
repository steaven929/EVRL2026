import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"F:\Claude code\EVRL260207")
SUMMARY_PATH = ROOT / "analysis" / "run_scan_summary.json"
RESULTS_ROOT = ROOT / "results"


def load_rows():
    rows = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    rows = [r for r in rows if (r.get("train_eps") or -1) >= 100]
    return dedupe_rows(rows)


def existing_file_count(run_dir: str):
    p = Path(run_dir)
    if not p.exists():
        return 0
    return sum(1 for x in p.iterdir() if x.is_file())


def dedupe_rows(rows):
    grouped = {}
    for r in rows:
        key = (r.get("group"), r.get("run_name"))
        grouped.setdefault(key, []).append(r)
    out = []
    for _, items in grouped.items():
        items.sort(key=lambda x: existing_file_count(x["dir"]), reverse=True)
        out.append(items[0])
    return out


def select_groups(rows):
    keep = {"MARL", "SAC", "TD3", "DDPG_PURE"}
    return [r for r in rows if r.get("group") in keep]


def top_by_metric(rows, metric, higher_better=True):
    cand = [r for r in rows if isinstance(r.get(metric), (int, float)) and np.isfinite(r.get(metric))]
    if not cand:
        return None
    cand.sort(key=lambda x: x[metric], reverse=higher_better)
    return cand[0]


def collect_best_of_n(rows):
    groups = ["MARL", "SAC", "TD3", "DDPG_PURE"]
    shared = []
    for g in groups:
        grows = [r for r in rows if r["group"] == g]
        if not grows:
            continue
        b_reward = top_by_metric(grows, "reward_tail10", True)
        b_svc = top_by_metric(grows, "svc_mean", True)
        b_rld = top_by_metric(grows, "rld_mean", False)
        b_srd = top_by_metric(grows, "srd_mean", True)
        shared.append(
            {
                "group": g,
                "N": len(grows),
                "best_reward_tail10": None if b_reward is None else b_reward["reward_tail10"],
                "best_reward_run": None if b_reward is None else b_reward["run_name"],
                "best_svc_mean": None if b_svc is None else b_svc["svc_mean"],
                "best_svc_run": None if b_svc is None else b_svc["run_name"],
                "best_rld_mean": None if b_rld is None else b_rld["rld_mean"],
                "best_rld_run": None if b_rld is None else b_rld["run_name"],
                "best_srd_mean": None if b_srd is None else b_srd["srd_mean"],
                "best_srd_run": None if b_srd is None else b_srd["run_name"],
            }
        )
    return shared


def np_last10_mean(path):
    if not path.exists():
        return None
    arr = np.load(path)
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return None
    return float(np.mean(arr[-min(10, arr.size) :]))


def collect_marl_charging(rows):
    marl_rows = [r for r in rows if r["group"] == "MARL"]
    metrics = {
        "charging_efficiency": ("charging_efficiency_history.npy", True),
        "response_readiness": ("response_readiness_history.npy", True),
        "charger_occupancy_rate": ("charger_occupancy_rate_history.npy", True),
        "charger_queue_delay": ("charger_queue_delay_history.npy", False),
        "energy_not_served": ("energy_not_served_history.npy", False),
        "safety_soc_violation_count": ("safety_soc_violation_count_history.npy", False),
        "dispatch_efficiency": ("dispatch_efficiency_history.npy", True),
        "dispatch_roi": ("dispatch_roi_history.npy", True),
    }

    table = []
    for r in marl_rows:
        item = {
            "run_name": r["run_name"],
            "dir": r["dir"],
            "train_eps": r["train_eps"],
            "svc_mean": r.get("svc_mean"),
            "rld_mean": r.get("rld_mean"),
        }
        for m, (fname, _) in metrics.items():
            item[m] = np_last10_mean(Path(r["dir"]) / fname)
        table.append(item)

    valid = []
    score_metrics = list(metrics.keys())
    for t in table:
        if all(isinstance(t.get(m), (int, float)) and np.isfinite(t[m]) for m in score_metrics):
            valid.append(t)

    if valid:
        # z-score composite
        mat = np.array([[v[m] for m in score_metrics] for v in valid], dtype=float)
        mu = np.mean(mat, axis=0)
        sd = np.std(mat, axis=0) + 1e-8
        z = (mat - mu) / sd
        for j, m in enumerate(score_metrics):
            _, higher = metrics[m]
            if not higher:
                z[:, j] = -z[:, j]
        comp = np.mean(z, axis=1)
        for i, v in enumerate(valid):
            v["composite_score"] = float(comp[i])
        valid.sort(key=lambda x: x["composite_score"], reverse=True)
        best = valid[0]
    else:
        best = None

    return table, best


def save_json_csv(data, json_path, csv_path):
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if not data:
        csv_path.write_text("", encoding="utf-8")
        return
    keys = list(data[0].keys())
    lines = [",".join(keys)]
    for row in data:
        vals = []
        for k in keys:
            v = row.get(k, "")
            s = "" if v is None else str(v)
            s = s.replace('"', '""')
            if "," in s:
                s = f'"{s}"'
            vals.append(s)
        lines.append(",".join(vals))
    csv_path.write_text("\n".join(lines), encoding="utf-8")


def plot_best_of_n(best_rows, out_file):
    groups = [r["group"] for r in best_rows]
    svc = [r["best_svc_mean"] for r in best_rows]
    rld = [r["best_rld_mean"] for r in best_rows]
    srd = [r["best_srd_mean"] for r in best_rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].bar(groups, svc, color="#2a9d8f")
    axes[0].set_title("Best SVC (higher better)")
    axes[0].set_ylim(0.94, 0.98)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(groups, rld, color="#e76f51")
    axes[1].set_title("Best RLD (lower better)")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(groups, srd, color="#457b9d")
    axes[2].set_title("Best SRD (higher better)")
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("Best-of-N Comparison (train_eps >= 100)")
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_distribution(rows, out_file):
    groups = ["MARL", "SAC", "TD3", "DDPG_PURE"]
    svc_data = []
    rld_data = []
    labels = []
    for g in groups:
        grows = [r for r in rows if r["group"] == g]
        s = [x["svc_mean"] for x in grows if isinstance(x.get("svc_mean"), (int, float))]
        d = [x["rld_mean"] for x in grows if isinstance(x.get("rld_mean"), (int, float))]
        if s and d:
            labels.append(f"{g}\n(n={len(grows)})")
            svc_data.append(s)
            rld_data.append(d)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].boxplot(svc_data, labels=labels, showmeans=True)
    axes[0].set_title("SVC distribution")
    axes[0].set_ylim(0.90, 1.00)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].boxplot(rld_data, labels=labels, showmeans=True)
    axes[1].set_title("RLD distribution")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Cross-Algorithm Distribution (train_eps >= 100)")
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metric_coverage(rows, out_file):
    groups = ["MARL", "SAC", "TD3", "DDPG_PURE"]
    metric_files = [
        "charging_efficiency_history.npy",
        "charger_queue_delay_history.npy",
        "energy_not_served_history.npy",
        "safety_soc_violation_count_history.npy",
        "dispatch_efficiency_history.npy",
        "dispatch_roi_history.npy",
    ]
    x = np.arange(len(metric_files))
    width = 0.18

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, g in enumerate(groups):
        grows = [r for r in rows if r["group"] == g]
        vals = []
        for mf in metric_files:
            c = sum(1 for r in grows if (Path(r["dir"]) / mf).exists())
            vals.append(c)
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=f"{g} (n={len(grows)})")

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_history.npy", "") for m in metric_files], rotation=20, ha="right")
    ax.set_ylabel("Run count with metric file")
    ax.set_title("Charging-Related Metric Coverage (train_eps >= 100)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_best_marl_profile(best_run, out_file):
    run_dir = Path(best_run["dir"])
    metric_defs = [
        ("charging_efficiency_history.npy", "Charging Eff.", True),
        ("response_readiness_history.npy", "Response Ready", True),
        ("charger_queue_delay_history.npy", "Queue Delay", False),
        ("energy_not_served_history.npy", "Energy Not Served", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.flatten()

    for i, (fname, title, _) in enumerate(metric_defs):
        p = run_dir / fname
        if not p.exists():
            axes[i].text(0.5, 0.5, "missing", ha="center", va="center")
            axes[i].set_title(title)
            continue
        arr = np.asarray(np.load(p), dtype=float)
        axes[i].plot(arr, linewidth=1.8)
        if arr.size > 10:
            k = min(10, arr.size)
            tail = float(np.mean(arr[-k:]))
            axes[i].axhline(tail, linestyle="--", alpha=0.6)
            axes[i].text(0.02, 0.9, f"last10={tail:.4f}", transform=axes[i].transAxes)
        axes[i].set_title(title)
        axes[i].grid(alpha=0.3)

    fig.suptitle(f"Best MARL Charging Profile: {best_run['run_name']}")
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(out_dir, shared_best, marl_best, marl_table):
    report = out_dir / "analysis_report.md"
    lines = []
    lines.append("# MARL Evidence Pack\n")
    lines.append("## Scope\n")
    lines.append("- Data source: `analysis/run_scan_summary.json`\n")
    lines.append("- Filter: `train_eps >= 100`\n")
    lines.append("- Groups: MARL, SAC, TD3, DDPG_PURE\n")
    lines.append("\n## Key Findings\n")
    lines.append("- MARL strongest cross-algorithm edge appears on `RLD` (lower better), with SVC competitive near SAC.\n")
    lines.append("- Charging-centric evidence is currently strongest in MARL-only metric set (partial coverage in non-MARL runs).\n")
    lines.append("\n## Best-of-N Shared Metrics\n")
    for row in shared_best:
        lines.append(
            f"- {row['group']} (N={row['N']}): best SVC={row['best_svc_mean']}, best RLD={row['best_rld_mean']}, best SRD={row['best_srd_mean']}\n"
        )

    lines.append("\n## MARL Charging Composite Best Run\n")
    if marl_best is None:
        lines.append("- No MARL run had full charging metric coverage for composite scoring.\n")
    else:
        lines.append(
            f"- Run: `{marl_best['run_name']}` (eps={marl_best['train_eps']}) composite_score={marl_best.get('composite_score')}\n"
        )
        lines.append(
            f"- charging_efficiency={marl_best.get('charging_efficiency')}, queue_delay={marl_best.get('charger_queue_delay')}, energy_not_served={marl_best.get('energy_not_served')}\n"
        )

    lines.append("\n## Files\n")
    lines.append("- fig1_best_of_n_shared.png\n")
    lines.append("- fig2_shared_metric_distribution.png\n")
    lines.append("- fig3_charging_metric_coverage.png\n")
    lines.append("- fig4_best_marl_charging_profile.png\n")
    lines.append("- shared_best_of_n.csv/json\n")
    lines.append("- marl_charging_table.csv/json\n")
    lines.append("\n## Caveats\n")
    lines.append("- Sample sizes are imbalanced across algorithms; best-of-N is not equivalent to typical performance.\n")
    lines.append("- Several charging-specific metrics are MARL-heavy in coverage; cross-algorithm claims should focus on shared metrics.\n")

    report.write_text("".join(lines), encoding="utf-8")


def main():
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = RESULTS_ROOT / f"marl_evidence_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = select_groups(load_rows())
    shared_best = collect_best_of_n(rows)
    marl_table, marl_best = collect_marl_charging(rows)

    save_json_csv(shared_best, out_dir / "shared_best_of_n.json", out_dir / "shared_best_of_n.csv")
    save_json_csv(marl_table, out_dir / "marl_charging_table.json", out_dir / "marl_charging_table.csv")

    plot_best_of_n(shared_best, out_dir / "fig1_best_of_n_shared.png")
    plot_distribution(rows, out_dir / "fig2_shared_metric_distribution.png")
    plot_metric_coverage(rows, out_dir / "fig3_charging_metric_coverage.png")
    if marl_best is not None:
        plot_best_marl_profile(marl_best, out_dir / "fig4_best_marl_charging_profile.png")

    write_report(out_dir, shared_best, marl_best, marl_table)
    print(out_dir)


if __name__ == "__main__":
    main()
