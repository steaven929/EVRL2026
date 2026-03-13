import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TARGET = Path(r"F:\Claude code\EVRL260207\results\marl_two_dimension_20260312-220209")
CSV_PATH = TARGET / "marl_two_dimension_ranking.csv"


def load_rows():
    rows = []
    with CSV_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "run_name": r["run_name"],
                    "train_eps": int(float(r["train_eps"])),
                    "svc_tail20": float(r["svc_tail20"]),
                    "ens_tail20": float(r["ens_tail20"]),
                    "ce_tail20": float(r["ce_tail20"]),
                    "qd_tail20": float(r["qd_tail20"]),
                    "sv_tail20": float(r["sv_tail20"]),
                    "demand_score": float(r["demand_satisfaction_score"]),
                    "charging_score": float(r["charging_satisfaction_score"]),
                    "balanced_score": float(r["balanced_score"]),
                    "dir": r["dir"],
                }
            )
    return rows


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def pareto_front(rows):
    pts = np.array([[r["demand_score"], r["charging_score"]] for r in rows], dtype=float)
    keep = np.ones(len(rows), dtype=bool)
    for i in range(len(rows)):
        for j in range(len(rows)):
            if i == j:
                continue
            # j dominates i if j>=i on both and strictly > one
            if (pts[j, 0] >= pts[i, 0] and pts[j, 1] >= pts[i, 1]) and (pts[j, 0] > pts[i, 0] or pts[j, 1] > pts[i, 1]):
                keep[i] = False
                break
    return [rows[i] for i in range(len(rows)) if keep[i]]


def quadrant_counts(rows):
    q = {"Q1(++ )": 0, "Q2(-,+)": 0, "Q3(--)": 0, "Q4(+,-)": 0}
    for r in rows:
        d, c = r["demand_score"], r["charging_score"]
        if d >= 0 and c >= 0:
            q["Q1(++ )"] += 1
        elif d < 0 <= c:
            q["Q2(-,+)"] += 1
        elif d < 0 and c < 0:
            q["Q3(--)"] += 1
        else:
            q["Q4(+,-)"] += 1
    return q


def plot_pareto(rows, frontier, out_file):
    x = np.array([r["demand_score"] for r in rows], dtype=float)
    y = np.array([r["charging_score"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    ax.scatter(x, y, s=45, alpha=0.5, color="#8d99ae", label="All MARL runs")

    fx = np.array([r["demand_score"] for r in frontier], dtype=float)
    fy = np.array([r["charging_score"] for r in frontier], dtype=float)
    order = np.argsort(fx)
    ax.plot(fx[order], fy[order], color="#2a9d8f", linewidth=2.0, label="Pareto frontier")
    ax.scatter(fx, fy, s=75, color="#2a9d8f", edgecolors="black", linewidths=0.4)

    for r in sorted(frontier, key=lambda z: z["balanced_score"], reverse=True)[:5]:
        ax.annotate(r["run_name"], (r["demand_score"], r["charging_score"]), fontsize=8, xytext=(4, 3), textcoords="offset points")

    ax.axhline(0, linestyle="--", color="#adb5bd", alpha=0.6)
    ax.axvline(0, linestyle="--", color="#adb5bd", alpha=0.6)
    ax.set_xlabel("Demand satisfaction score")
    ax.set_ylabel("Charging satisfaction score")
    ax.set_title("Fig D3: Pareto Frontier in Two-Dimension Space")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_eps_tradeoff(rows, out_file):
    # bucket by train eps
    bins = [(100, 199), (200, 399), (400, 799), (800, 5000)]
    labels = ["100-199", "200-399", "400-799", "800+"]
    d_means, c_means, n_vals = [], [], []
    for lo, hi in bins:
        group = [r for r in rows if lo <= r["train_eps"] <= hi]
        if group:
            d_means.append(float(np.mean([r["demand_score"] for r in group])))
            c_means.append(float(np.mean([r["charging_score"] for r in group])))
            n_vals.append(len(group))
        else:
            d_means.append(np.nan)
            c_means.append(np.nan)
            n_vals.append(0)

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.bar(x - width / 2, d_means, width=width, color="#457b9d", label="Demand score")
    ax.bar(x + width / 2, c_means, width=width, color="#e76f51", label="Charging score")
    for i, n in enumerate(n_vals):
        ax.text(i, max((d_means[i] if np.isfinite(d_means[i]) else 0), (c_means[i] if np.isfinite(c_means[i]) else 0)) + 0.05, f"n={n}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Fig D4: Mean Two-Dimension Scores by Training-Episode Range")
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_tables(rows, frontier):
    # frontier table
    fpath = TARGET / "marl_two_dimension_pareto.csv"
    cols = ["run_name", "train_eps", "demand_score", "charging_score", "balanced_score", "dir"]
    lines = [",".join(cols)]
    for r in sorted(frontier, key=lambda z: z["balanced_score"], reverse=True):
        vals = [r[c] for c in cols]
        out = []
        for v in vals:
            s = f"{v:.6f}" if isinstance(v, float) else str(v)
            if "," in s:
                s = f'"{s}"'
            out.append(s)
        lines.append(",".join(out))
    fpath.write_text("\n".join(lines), encoding="utf-8")


def write_report(rows, frontier):
    corr = pearson([r["demand_score"] for r in rows], [r["charging_score"] for r in rows])
    q = quadrant_counts(rows)
    top_bal = sorted(rows, key=lambda z: z["balanced_score"], reverse=True)[:5]
    top_d = sorted(rows, key=lambda z: z["demand_score"], reverse=True)[:5]
    top_c = sorted(rows, key=lambda z: z["charging_score"], reverse=True)[:5]

    lines = []
    lines.append("# MARL Two-Dimension Deep-Dive\n")
    lines.append(f"- Runs analyzed: {len(rows)}\n")
    lines.append(f"- Demand-Charging Pearson correlation: {corr:.4f}\n")
    lines.append(f"- Quadrant counts: {q}\n")
    lines.append(f"- Pareto frontier size: {len(frontier)}\n")
    lines.append("\n## Top-5 by balanced score\n")
    for r in top_bal:
        lines.append(f"- {r['run_name']} (eps={r['train_eps']}): D={r['demand_score']:.4f}, C={r['charging_score']:.4f}, B={r['balanced_score']:.4f}\n")
    lines.append("\n## Top-5 demand-focused runs\n")
    for r in top_d:
        lines.append(f"- {r['run_name']} (eps={r['train_eps']}): D={r['demand_score']:.4f}, C={r['charging_score']:.4f}\n")
    lines.append("\n## Top-5 charging-focused runs\n")
    for r in top_c:
        lines.append(f"- {r['run_name']} (eps={r['train_eps']}): D={r['demand_score']:.4f}, C={r['charging_score']:.4f}\n")

    lines.append("\n## Interpretation\n")
    lines.append("- If correlation is negative or near zero, demand and charging goals compete; balanced policy design is necessary.\n")
    lines.append("- Pareto frontier runs are recommended as candidate checkpoints for deployment or further ablation.\n")
    lines.append("- Episode-range chart (Fig D4) indicates whether longer training improves both dimensions or introduces trade-offs.\n")

    (TARGET / "marl_two_dimension_deepdive.md").write_text("".join(lines), encoding="utf-8")


def main():
    rows = load_rows()
    frontier = pareto_front(rows)

    plot_pareto(rows, frontier, TARGET / "figD3_pareto_frontier.png")
    plot_eps_tradeoff(rows, TARGET / "figD4_episode_range_tradeoff.png")
    save_tables(rows, frontier)
    write_report(rows, frontier)

    print(TARGET)


if __name__ == "__main__":
    main()
