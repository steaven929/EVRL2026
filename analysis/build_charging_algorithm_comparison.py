import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(r"F:\Claude code\EVRL260207")
SUMMARY = ROOT / "analysis" / "run_scan_summary.json"
OUT = ROOT / "results" / "marl_supplementary_20260308-212225"
ALGOS = ["MARL", "SAC", "TD3", "DDPG_PURE"]

METRICS = {
    "charging_efficiency_history.npy": "ChargeEff",
    "charger_queue_delay_history.npy": "QueueDelay",
    "energy_not_served_history.npy": "EnergyNotServed",
    "safety_soc_violation_count_history.npy": "SOCViolation",
    "dispatch_roi_history.npy": "DispatchROI",
}


def tail20(path: Path):
    if not path.exists():
        return None
    a = np.asarray(np.load(path), dtype=float)
    if a.size == 0:
        return None
    return float(np.mean(a[-min(20, a.size) :]))


def fmt(v):
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def stat(vals):
    if not vals:
        return None
    a = np.asarray(vals, dtype=float)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
    }


def main():
    rows = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = [r for r in rows if (r.get("train_eps") or -1) >= 100 and r.get("group") in ALGOS]
    rows = [r for r in rows if "marl_outputs" not in str(r.get("dir", ""))]

    table = []
    for algo in ALGOS:
        g = [r for r in rows if r.get("group") == algo]
        rec = {
            "Algorithm": "DDPG" if algo == "DDPG_PURE" else algo,
            "Runs_total": len(g),
        }
        for fname, key in METRICS.items():
            vals = []
            for r in g:
                v = tail20(Path(r["dir"]) / fname)
                if isinstance(v, float):
                    vals.append(v)
            s = stat(vals)
            rec[f"{key}_N"] = 0 if s is None else s["n"]
            rec[f"{key}_median"] = None if s is None else s["median"]
            rec[f"{key}_mean"] = None if s is None else s["mean"]
            rec[f"{key}_IQR"] = None if s is None else s["iqr"]
        table.append(rec)

    keys = list(table[0].keys())
    csv_lines = [",".join(keys)]
    for r in table:
        vals = []
        for k in keys:
            s = fmt(r[k])
            if "," in s:
                s = f'"{s}"'
            vals.append(s)
        csv_lines.append(",".join(vals))
    (OUT / "charging_metrics_algorithm_comparison.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    md = []
    md.append("# Charging Metrics: Cross-Algorithm Comparison\n\n")
    md.append("Source: train_eps>=100 runs from run_scan_summary.json.\n\n")
    md.append("| Algo | Runs | CE_N | CE_med | QD_N | QD_med | ENS_N | ENS_med | SOCV_N | SOCV_med | ROI_N | ROI_med |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in table:
        md.append(
            f"| {r['Algorithm']} | {r['Runs_total']} | {r['ChargeEff_N']} | {fmt(r['ChargeEff_median'])} | {r['QueueDelay_N']} | {fmt(r['QueueDelay_median'])} | {r['EnergyNotServed_N']} | {fmt(r['EnergyNotServed_median'])} | {r['SOCViolation_N']} | {fmt(r['SOCViolation_median'])} | {r['DispatchROI_N']} | {fmt(r['DispatchROI_median'])} |\n"
        )

    md.append("\n## Notes\n")
    md.append("- NA means the algorithm logs do not contain that charging metric file.\n")
    md.append("- Therefore, strict cross-alg charging-mechanism claims are currently MARL-centric unless baseline logging is extended.\n")
    (OUT / "charging_metrics_algorithm_comparison.md").write_text("".join(md), encoding="utf-8")

    print(OUT)


if __name__ == "__main__":
    main()
