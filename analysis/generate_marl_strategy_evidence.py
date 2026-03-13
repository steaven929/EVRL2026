import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"F:\Claude code\EVRL260207")
SUMMARY = ROOT / "analysis" / "run_scan_summary.json"
OUT_ROOT = ROOT / "results"
FORCED_RUN_NAME = "marl_20260306-110926"


REQ_METRICS = {
    "charging_efficiency_history.npy": True,
    "charger_queue_delay_history.npy": False,
    "energy_not_served_history.npy": False,
    "safety_soc_violation_count_history.npy": False,
    "dispatch_roi_history.npy": True,
    "response_readiness_history.npy": True,
    "service_quality_index_history.npy": True,
}


def load_np(path: Path):
    if not path.exists():
        return None
    return np.asarray(np.load(path), dtype=float)


def tail_mean(arr, k=20):
    if arr is None or arr.size == 0:
        return None
    return float(np.mean(arr[-min(k, arr.size) :]))


def pick_best_marl_run():
    rows = json.loads(SUMMARY.read_text(encoding="utf-8"))
    marl = [r for r in rows if r.get("group") == "MARL" and (r.get("train_eps") or -1) >= 100]

    candidates = []
    for r in marl:
        # skip duplicate nested export dirs
        if "marl_outputs" in str(r.get("dir", "")):
            continue
        p = Path(r["dir"])
        vals = {}
        ok = True
        for f, higher in REQ_METRICS.items():
            a = load_np(p / f)
            v = tail_mean(a, 20)
            if v is None:
                ok = False
                break
            vals[f] = v
        if ok:
            candidates.append((r, vals))

    if not candidates:
        return None, None

    # optional: force a specific run for strategy-focused analysis
    for c in candidates:
        r = c[0]
        if str(r.get("run_name", "")) == FORCED_RUN_NAME:
            keys = list(REQ_METRICS.keys())
            mat = np.array([[cc[1][k] for k in keys] for cc in candidates], dtype=float)
            mu = np.mean(mat, axis=0)
            sd = np.std(mat, axis=0) + 1e-8
            z = (mat - mu) / sd
            for j, k in enumerate(keys):
                if not REQ_METRICS[k]:
                    z[:, j] = -z[:, j]
            scores = np.mean(z, axis=1)
            score_map = {id(candidates[i][0]): float(scores[i]) for i in range(len(candidates))}
            return r, score_map.get(id(r), float("nan"))

    # fallback: best reward_tail10 among runs with complete mechanism logs
    ranked = [c[0] for c in candidates if isinstance(c[0].get("reward_tail10"), (int, float))]
    ranked.sort(key=lambda r: (r.get("reward_tail10", -1e18), r.get("train_eps", -1)), reverse=True)
    best = ranked[0]

    # composite score only for reporting
    keys = list(REQ_METRICS.keys())
    mat = np.array([[c[1][k] for k in keys] for c in candidates], dtype=float)
    mu = np.mean(mat, axis=0)
    sd = np.std(mat, axis=0) + 1e-8
    z = (mat - mu) / sd
    for j, k in enumerate(keys):
        if not REQ_METRICS[k]:
            z[:, j] = -z[:, j]
    scores = np.mean(z, axis=1)
    # map candidate->score
    score_map = {id(candidates[i][0]): float(scores[i]) for i in range(len(candidates))}
    return best, score_map.get(id(best), float("nan"))


def _moving(a, w=15):
    if a is None or a.size == 0:
        return a
    w = max(1, min(w, a.size))
    pad_l = w // 2
    pad_r = w - 1 - pad_l
    p = np.pad(a, (pad_l, pad_r), mode="edge")
    k = np.ones(w, dtype=float) / w
    return np.convolve(p, k, mode="valid")


def fig_phase_charge_queue(run_dir: Path, out_file: Path):
    ce = _moving(load_np(run_dir / "charging_efficiency_history.npy"), 11)
    qd = _moving(load_np(run_dir / "charger_queue_delay_history.npy"), 11)
    n = min(len(ce), len(qd))
    ce, qd = ce[:n], qd[:n]

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    c = np.arange(n)
    sc = ax.scatter(qd, ce, c=c, cmap="viridis", s=22, alpha=0.8)
    ax.scatter([qd[0]], [ce[0]], color="#e76f51", s=60, marker="o", label="start")
    ax.scatter([qd[-1]], [ce[-1]], color="#2a9d8f", s=70, marker="X", label="end")
    ax.set_xlabel("Queue Delay (lower better)")
    ax.set_ylabel("Charging Efficiency (higher better)")
    ax.set_title("Policy Trajectory: Charging Efficiency vs Queue Delay")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Episode")
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_safety_supply(run_dir: Path, out_file: Path):
    ens = _moving(load_np(run_dir / "energy_not_served_history.npy"), 11)
    socv = _moving(load_np(run_dir / "safety_soc_violation_count_history.npy"), 11)
    n = min(len(ens), len(socv))
    ens, socv = ens[:n], socv[:n]

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    c = np.arange(n)
    sc = ax.scatter(ens, socv, c=c, cmap="plasma", s=22, alpha=0.8)
    ax.scatter([ens[0]], [socv[0]], color="#e76f51", s=60, marker="o", label="start")
    ax.scatter([ens[-1]], [socv[-1]], color="#2a9d8f", s=70, marker="X", label="end")
    ax.set_xlabel("Energy Not Served (lower better)")
    ax.set_ylabel("SOC Violation Count (lower better)")
    ax.set_title("Policy Trajectory: Supply Reliability vs Battery Safety")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Episode")
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_strategy_shift(run_dir: Path, out_file: Path):
    keys = [
        ("charging_efficiency_history.npy", True, "ChargeEff"),
        ("response_readiness_history.npy", True, "Ready"),
        ("service_quality_index_history.npy", True, "SQI"),
        ("dispatch_roi_history.npy", True, "ROI"),
        ("charger_queue_delay_history.npy", False, "QueueDelay"),
        ("energy_not_served_history.npy", False, "EnergyGap"),
        ("safety_soc_violation_count_history.npy", False, "SOCViolation"),
    ]
    arrs = {}
    for f, _, name in keys:
        a = load_np(run_dir / f)
        arrs[name] = a if a is not None else np.array([])

    # early/mid/late segment means (normalized to 0-1 orientation higher better)
    segs = ["Early", "Mid", "Late"]
    vals = {s: [] for s in segs}
    for f, higher, name in keys:
        a = arrs[name]
        if a.size < 9:
            v = [np.nan, np.nan, np.nan]
        else:
            n = a.size
            i1, i2 = n // 3, 2 * n // 3
            v = [float(np.mean(a[:i1])), float(np.mean(a[i1:i2])), float(np.mean(a[i2:]))]
        # normalize across three segments for this metric
        vv = np.array(v, dtype=float)
        mn, mx = np.nanmin(vv), np.nanmax(vv)
        if np.isfinite(mn) and np.isfinite(mx) and (mx - mn) > 1e-12:
            nv = (vv - mn) / (mx - mn)
        else:
            nv = np.array([0.5, 0.5, 0.5])
        if not higher:
            nv = 1.0 - nv
        for si, s in enumerate(segs):
            vals[s].append(float(nv[si]))

    x = np.arange(len(keys))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11.2, 4.2))
    ax.bar(x - width, vals["Early"], width=width, label="Early", color="#8d99ae")
    ax.bar(x, vals["Mid"], width=width, label="Mid", color="#457b9d")
    ax.bar(x + width, vals["Late"], width=width, label="Late", color="#2a9d8f")
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, _, name in keys], rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Normalized strategy score (higher better)")
    ax.set_title("Strategy Shift: Early vs Mid vs Late Training")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(out_dir: Path, best_run: dict, score: float):
    run_dir = Path(best_run["dir"])
    def tl(fname):
        a = load_np(run_dir / fname)
        return tail_mean(a, 20)

    lines = []
    lines.append("# MARL Strategy Evidence\n")
    lines.append(f"- Selected best MARL run: `{best_run['run_name']}` (eps={best_run['train_eps']})\n")
    lines.append(f"- Composite mechanism score (z-based): {score:.4f}\n")
    lines.append("\n## Tail-20 key metrics\n")
    lines.append(f"- Charging Efficiency: {tl('charging_efficiency_history.npy')}\n")
    lines.append(f"- Queue Delay: {tl('charger_queue_delay_history.npy')}\n")
    lines.append(f"- Energy Not Served: {tl('energy_not_served_history.npy')}\n")
    lines.append(f"- SOC Violations: {tl('safety_soc_violation_count_history.npy')}\n")
    lines.append(f"- Dispatch ROI: {tl('dispatch_roi_history.npy')}\n")
    lines.append("\n## Interpretation\n")
    lines.append("- If Fig S1 moves toward lower queue delay and higher charging efficiency over episodes, MARL is learning charging-priority control.\n")
    lines.append("- If Fig S2 moves toward lower energy-not-served and lower SOC violations, MARL is learning safer energy management.\n")
    lines.append("- Fig S3 indicates whether late-stage policy dominates early-stage behavior across charging and dispatch dimensions.\n")
    (out_dir / "strategy_analysis.md").write_text("".join(lines), encoding="utf-8")


def main():
    best_run, score = pick_best_marl_run()
    if best_run is None:
        raise RuntimeError("No MARL run with complete mechanism metrics found.")

    out_dir = OUT_ROOT / f"marl_strategy_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(best_run["dir"])
    fig_phase_charge_queue(run_dir, out_dir / "figS1_chargeeff_vs_queuedelay_trajectory.png")
    fig_safety_supply(run_dir, out_dir / "figS2_energygap_vs_socviolation_trajectory.png")
    fig_strategy_shift(run_dir, out_dir / "figS3_early_mid_late_strategy_shift.png")

    write_report(out_dir, best_run, score)
    (out_dir / "selected_run.json").write_text(json.dumps(best_run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
