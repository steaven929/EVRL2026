import json
import os
from pathlib import Path

import numpy as np


TARGET_DIR = Path(r"F:\Claude code\EVRL260207\results\marl_supplementary_20260308-212225")
ROWS_PATH = TARGET_DIR / "rows_used.json"
GROUPS = ["MARL", "SAC", "TD3", "DDPG_PURE"]


def fmt(v):
    if v is None:
        return "NA"
    return f"{v:.6f}"


def stats(vals):
    if not vals:
        return None
    a = np.asarray(vals, dtype=float)
    return {
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "q1": float(np.percentile(a, 25)),
        "q3": float(np.percentile(a, 75)),
        "n": int(a.size),
    }


def main():
    rows = json.loads(ROWS_PATH.read_text(encoding="utf-8"))

    # Table 1: shared metrics summary
    shared_metrics = [
        ("reward_tail10", "RewardTail10", "higher"),
        ("svc_mean", "SVC", "higher"),
        ("rld_mean", "RLD", "lower"),
        ("srd_mean", "SRD", "higher"),
    ]

    t1 = []
    for g in GROUPS:
        grows = [r for r in rows if r["group"] == g]
        row = {"Algorithm": g, "N": len(grows)}
        for key, alias, _ in shared_metrics:
            vals = [r[key] for r in grows if isinstance(r.get(key), (int, float))]
            s = stats(vals)
            row[f"{alias}_median"] = None if s is None else s["median"]
            row[f"{alias}_mean"] = None if s is None else s["mean"]
            row[f"{alias}_std"] = None if s is None else s["std"]
        t1.append(row)

    # Table 2: metric coverage
    coverage_metrics = [
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

    t2 = []
    for mf in coverage_metrics:
        row = {"MetricFile": mf}
        for g in GROUPS:
            grows = [r for r in rows if r["group"] == g]
            n = len(grows)
            c = sum(1 for r in grows if (Path(r["dir"]) / mf).exists())
            row[g] = f"{c}/{n}" if n > 0 else "0/0"
        t2.append(row)

    # Table 3: MARL charging mechanism metrics
    marl_rows = [r for r in rows if r["group"] == "MARL"]
    mech_metrics = [
        ("charging_efficiency_history.npy", "ChargingEfficiency", "higher"),
        ("charger_queue_delay_history.npy", "QueueDelay", "lower"),
        ("energy_not_served_history.npy", "EnergyNotServed", "lower"),
        ("safety_soc_violation_count_history.npy", "SocViolationCount", "lower"),
        ("dispatch_efficiency_history.npy", "DispatchEfficiency", "higher"),
        ("dispatch_roi_history.npy", "DispatchROI", "higher"),
    ]

    t3 = []
    for mf, alias, direction in mech_metrics:
        vals = []
        for r in marl_rows:
            p = Path(r["dir"]) / mf
            if not p.exists():
                continue
            arr = np.asarray(np.load(p), dtype=float)
            if arr.size == 0:
                continue
            vals.append(float(np.mean(arr[-min(10, arr.size) :])))
        s = stats(vals)
        t3.append(
            {
                "Metric": alias,
                "Direction": direction,
                "N": 0 if s is None else s["n"],
                "Median": None if s is None else s["median"],
                "Q1": None if s is None else s["q1"],
                "Q3": None if s is None else s["q3"],
                "Mean": None if s is None else s["mean"],
                "Std": None if s is None else s["std"],
            }
        )

    def write_csv(path, rows_data):
        if not rows_data:
            path.write_text("", encoding="utf-8")
            return
        keys = list(rows_data[0].keys())
        lines = [",".join(keys)]
        for r in rows_data:
            vals = []
            for k in keys:
                v = r.get(k)
                if isinstance(v, float):
                    s = f"{v:.6f}"
                else:
                    s = "" if v is None else str(v)
                if "," in s:
                    s = f'"{s}"'
                vals.append(s)
            lines.append(",".join(vals))
        path.write_text("\n".join(lines), encoding="utf-8")

    write_csv(TARGET_DIR / "table1_shared_metrics_summary.csv", t1)
    write_csv(TARGET_DIR / "table2_metric_coverage.csv", t2)
    write_csv(TARGET_DIR / "table3_marl_mechanism_summary.csv", t3)

    # markdown bundle
    md = []
    md.append("# Supplementary Tables\n\n")
    md.append("## Table 1: Shared Metrics Summary\n\n")
    md.append("| Algorithm | N | RewardTail10_median | RewardTail10_mean | SVC_median | SVC_mean | RLD_median | RLD_mean | SRD_median | SRD_mean |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in t1:
        md.append(
            f"| {r['Algorithm']} | {r['N']} | {fmt(r['RewardTail10_median'])} | {fmt(r['RewardTail10_mean'])} | {fmt(r['SVC_median'])} | {fmt(r['SVC_mean'])} | {fmt(r['RLD_median'])} | {fmt(r['RLD_mean'])} | {fmt(r['SRD_median'])} | {fmt(r['SRD_mean'])} |\n"
        )

    md.append("\n## Table 2: Metric Coverage\n\n")
    md.append("| Metric File | MARL | SAC | TD3 | DDPG_PURE |\n")
    md.append("|---|---:|---:|---:|---:|\n")
    for r in t2:
        md.append(f"| {r['MetricFile']} | {r['MARL']} | {r['SAC']} | {r['TD3']} | {r['DDPG_PURE']} |\n")

    md.append("\n## Table 3: MARL Charging Mechanism Summary (last-10 mean per run)\n\n")
    md.append("| Metric | Direction | N | Median | Q1 | Q3 | Mean | Std |\n")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|\n")
    for r in t3:
        md.append(
            f"| {r['Metric']} | {r['Direction']} | {r['N']} | {fmt(r['Median'])} | {fmt(r['Q1'])} | {fmt(r['Q3'])} | {fmt(r['Mean'])} | {fmt(r['Std'])} |\n"
        )

    (TARGET_DIR / "tables.md").write_text("".join(md), encoding="utf-8")
    print(TARGET_DIR)


if __name__ == "__main__":
    main()
