import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"F:\Claude code\EVRL260207")
SUMMARY = ROOT / "analysis" / "run_scan_summary.json"
OUT_ROOT = ROOT / "results"


def load_np(path: Path):
    if not path.exists():
        return None
    return np.asarray(np.load(path), dtype=float)


def tail_mean(a, k=20):
    if a is None or a.size == 0:
        return None
    return float(np.mean(a[-min(k, a.size) :]))


def zscore(v):
    x = np.asarray(v, dtype=float)
    m = np.mean(x)
    s = np.std(x) + 1e-8
    return (x - m) / s


def collect_runs():
    rows = json.loads(SUMMARY.read_text(encoding="utf-8"))
    marl = [r for r in rows if r.get("group") == "MARL" and (r.get("train_eps") or -1) >= 100]
    dedup = {}
    for r in marl:
        # skip duplicated nested export directories
        if "marl_outputs" in str(r.get("dir", "")):
            continue
        k = r.get("dir")
        dedup[k] = r
    marl = list(dedup.values())

    records = []
    for r in marl:
        p = Path(r["dir"])
        svc = load_np(p / "svc_history.npy")
        ens = load_np(p / "energy_not_served_history.npy")
        ce = load_np(p / "charging_efficiency_history.npy")
        qd = load_np(p / "charger_queue_delay_history.npy")
        sv = load_np(p / "safety_soc_violation_count_history.npy")
        if any(x is None for x in [svc, ens, ce, qd, sv]):
            continue
        rec = {
            "run_name": r["run_name"],
            "dir": r["dir"],
            "train_eps": int(r.get("train_eps") or len(svc)),
            "svc_tail20": tail_mean(svc, 20),
            "ens_tail20": tail_mean(ens, 20),
            "ce_tail20": tail_mean(ce, 20),
            "qd_tail20": tail_mean(qd, 20),
            "sv_tail20": tail_mean(sv, 20),
        }
        records.append(rec)
    return records


def compute_scores(records):
    svc = [r["svc_tail20"] for r in records]
    ens = [r["ens_tail20"] for r in records]
    ce = [r["ce_tail20"] for r in records]
    qd = [r["qd_tail20"] for r in records]
    sv = [r["sv_tail20"] for r in records]

    z_svc = zscore(svc)
    z_ens = -zscore(ens)
    z_ce = zscore(ce)
    z_qd = -zscore(qd)
    z_sv = -zscore(sv)

    for i, r in enumerate(records):
        r["demand_satisfaction_score"] = float(0.6 * z_svc[i] + 0.4 * z_ens[i])
        r["charging_satisfaction_score"] = float(0.5 * z_ce[i] + 0.3 * z_qd[i] + 0.2 * z_sv[i])
        # balanced score favors both dimensions
        r["balanced_score"] = float(
            0.5 * (r["demand_satisfaction_score"] + r["charging_satisfaction_score"])
            - 0.25 * abs(r["demand_satisfaction_score"] - r["charging_satisfaction_score"])
        )


def plot_run_scatter(records, out_file):
    xs = np.array([r["demand_satisfaction_score"] for r in records], dtype=float)
    ys = np.array([r["charging_satisfaction_score"] for r in records], dtype=float)
    bs = np.array([r["balanced_score"] for r in records], dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    sc = ax.scatter(xs, ys, c=bs, cmap="viridis", s=55, alpha=0.85)
    ax.axhline(0, linestyle="--", color="#8d99ae", alpha=0.6)
    ax.axvline(0, linestyle="--", color="#8d99ae", alpha=0.6)
    ax.set_xlabel("Demand satisfaction score")
    ax.set_ylabel("Charging satisfaction score")
    ax.set_title("MARL Runs: Demand vs Charging Satisfaction")
    ax.grid(alpha=0.25)

    top = sorted(records, key=lambda r: r["balanced_score"], reverse=True)[:5]
    for r in top:
        ax.annotate(r["run_name"], (r["demand_satisfaction_score"], r["charging_satisfaction_score"]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Balanced score")
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def moving(a, w=15):
    w = max(1, min(w, a.size))
    p = np.pad(a, (w // 2, w - 1 - w // 2), mode="edge")
    k = np.ones(w, dtype=float) / w
    return np.convolve(p, k, mode="valid")


def norm01(a):
    mn = float(np.nanmin(a))
    mx = float(np.nanmax(a))
    if mx - mn < 1e-12:
        return np.full_like(a, 0.5, dtype=float)
    return (a - mn) / (mx - mn)


def plot_best_trajectory(best, out_file):
    p = Path(best["dir"])
    svc = moving(load_np(p / "svc_history.npy"), 11)
    ens = moving(load_np(p / "energy_not_served_history.npy"), 11)
    ce = moving(load_np(p / "charging_efficiency_history.npy"), 11)
    qd = moving(load_np(p / "charger_queue_delay_history.npy"), 11)
    sv = moving(load_np(p / "safety_soc_violation_count_history.npy"), 11)

    n = min(len(svc), len(ens), len(ce), len(qd), len(sv))
    svc, ens, ce, qd, sv = svc[:n], ens[:n], ce[:n], qd[:n], sv[:n]

    demand_idx = 0.6 * norm01(svc) + 0.4 * (1.0 - norm01(ens))
    charging_idx = 0.5 * norm01(ce) + 0.3 * (1.0 - norm01(qd)) + 0.2 * (1.0 - norm01(sv))

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    c = np.arange(n)
    sc = ax.scatter(demand_idx, charging_idx, c=c, cmap="plasma", s=22, alpha=0.8)
    ax.plot(demand_idx, charging_idx, alpha=0.35, linewidth=1.0, color="#555")
    ax.scatter([demand_idx[0]], [charging_idx[0]], color="#e76f51", s=70, label="start")
    ax.scatter([demand_idx[-1]], [charging_idx[-1]], color="#2a9d8f", s=80, marker="X", label="end")
    ax.set_xlabel("Demand satisfaction index (normalized)")
    ax.set_ylabel("Charging satisfaction index (normalized)")
    ax.set_title(f"Best MARL trajectory: {best['run_name']}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Episode")
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_tables(records, out_dir):
    cols = [
        "run_name",
        "train_eps",
        "svc_tail20",
        "ens_tail20",
        "ce_tail20",
        "qd_tail20",
        "sv_tail20",
        "demand_satisfaction_score",
        "charging_satisfaction_score",
        "balanced_score",
        "dir",
    ]
    records = sorted(records, key=lambda r: r["balanced_score"], reverse=True)

    # csv
    lines = [",".join(cols)]
    for r in records:
        vals = []
        for c in cols:
            v = r.get(c)
            s = f"{v:.6f}" if isinstance(v, float) else str(v)
            if "," in s:
                s = f'"{s}"'
            vals.append(s)
        lines.append(",".join(vals))
    (out_dir / "marl_two_dimension_ranking.csv").write_text("\n".join(lines), encoding="utf-8")

    # markdown
    md = []
    md.append("# MARL Two-Dimension Analysis\n")
    md.append("\n## Definition\n")
    md.append("- Demand satisfaction score = 0.6*z(SVC_tail20) + 0.4*z(-EnergyNotServed_tail20)\n")
    md.append("- Charging satisfaction score = 0.5*z(ChargeEff_tail20) + 0.3*z(-QueueDelay_tail20) + 0.2*z(-SOCViolation_tail20)\n")
    md.append("\n## Top Runs\n")
    md.append("| Run | Eps | DemandScore | ChargingScore | Balanced |\n")
    md.append("|---|---:|---:|---:|---:|\n")
    for r in records[:10]:
        md.append(
            f"| {r['run_name']} | {r['train_eps']} | {r['demand_satisfaction_score']:.4f} | {r['charging_satisfaction_score']:.4f} | {r['balanced_score']:.4f} |\n"
        )
    (out_dir / "marl_two_dimension_analysis.md").write_text("".join(md), encoding="utf-8")


def main():
    records = collect_runs()
    if not records:
        raise RuntimeError("No MARL runs with complete demand/charging metrics found")

    compute_scores(records)
    best = sorted(records, key=lambda r: r["balanced_score"], reverse=True)[0]

    out_dir = OUT_ROOT / f"marl_two_dimension_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_run_scatter(records, out_dir / "figD1_marl_demand_vs_charging_scatter.png")
    plot_best_trajectory(best, out_dir / "figD2_best_marl_demand_charging_trajectory.png")
    save_tables(records, out_dir)

    (out_dir / "best_run.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
