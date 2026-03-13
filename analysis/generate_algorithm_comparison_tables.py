import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(r"F:\Claude code\EVRL260207")
SUMMARY = ROOT / "analysis" / "run_scan_summary.json"
OUT = ROOT / "results" / "marl_supplementary_20260308-212225"
ALGOS = ["MARL", "SAC", "TD3", "DDPG_PURE"]


def iqr(a):
    return float(np.percentile(a, 75) - np.percentile(a, 25))


def stat(vals):
    if len(vals) == 0:
        return None
    a = np.asarray(vals, dtype=float)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "iqr": iqr(a),
        "cv": float(np.std(a) / (np.mean(a) + 1e-8)),
    }


def tail_mean(path: Path, k=20):
    if not path.exists():
        return None
    a = np.asarray(np.load(path), dtype=float)
    if a.size == 0:
        return None
    return float(np.mean(a[-min(k, a.size) :]))


def fmt(v):
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for r in rows:
        vals = []
        for k in keys:
            s = fmt(r[k])
            if "," in s:
                s = f'"{s}"'
            vals.append(s)
        lines.append(",".join(vals))
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = [r for r in rows if (r.get("train_eps") or -1) >= 100 and r.get("group") in ALGOS]

    # remove nested duplicate exports
    clean = []
    seen = set()
    for r in rows:
        d = str(r.get("dir", ""))
        if "marl_outputs" in d:
            continue
        key = (r.get("group"), r.get("run_name"), d)
        if key in seen:
            continue
        seen.add(key)
        clean.append(r)
    rows = clean

    # Table 1: shared performance
    t1 = []
    for algo in ALGOS:
        g = [r for r in rows if r.get("group") == algo]
        reward = stat([r["reward_tail10"] for r in g if isinstance(r.get("reward_tail10"), (int, float))])
        svc = stat([r["svc_mean"] for r in g if isinstance(r.get("svc_mean"), (int, float))])
        rld = stat([r["rld_mean"] for r in g if isinstance(r.get("rld_mean"), (int, float))])
        srd = stat([r["srd_mean"] for r in g if isinstance(r.get("srd_mean"), (int, float))])

        t1.append(
            {
                "Algorithm": algo.replace("DDPG_PURE", "DDPG"),
                "N_runs": len(g),
                "RewardTail10_median": None if reward is None else reward["median"],
                "RewardTail10_iqr": None if reward is None else reward["iqr"],
                "SVC_median": None if svc is None else svc["median"],
                "SVC_iqr": None if svc is None else svc["iqr"],
                "RLD_median": None if rld is None else rld["median"],
                "RLD_iqr": None if rld is None else rld["iqr"],
                "SRD_median": None if srd is None else srd["median"],
                "SRD_iqr": None if srd is None else srd["iqr"],
            }
        )

    write_csv(OUT / "table_comp_1_shared_performance.csv", t1)

    # Table 2: charging mechanism by algorithm
    metric_files = {
        "ChargeEff_tail20": "charging_efficiency_history.npy",
        "QueueDelay_tail20": "charger_queue_delay_history.npy",
        "EnergyNotServed_tail20": "energy_not_served_history.npy",
        "SOCViolation_tail20": "safety_soc_violation_count_history.npy",
        "DispatchROI_tail20": "dispatch_roi_history.npy",
    }

    t2 = []
    for algo in ALGOS:
        g = [r for r in rows if r.get("group") == algo]
        rec = {"Algorithm": algo.replace("DDPG_PURE", "DDPG"), "N_runs": len(g)}
        for key, fname in metric_files.items():
            vals = []
            for r in g:
                v = tail_mean(Path(r["dir"]) / fname, 20)
                if isinstance(v, float):
                    vals.append(v)
            s = stat(vals)
            rec[f"{key}_N"] = 0 if s is None else s["n"]
            rec[f"{key}_median"] = None if s is None else s["median"]
            rec[f"{key}_iqr"] = None if s is None else s["iqr"]
        t2.append(rec)

    write_csv(OUT / "table_comp_2_charging_mechanism.csv", t2)

    # Table 3: robustness / consistency by algorithm on shared metrics
    t3 = []
    for algo in ALGOS:
        g = [r for r in rows if r.get("group") == algo]
        svc_vals = [r["svc_mean"] for r in g if isinstance(r.get("svc_mean"), (int, float))]
        rld_vals = [r["rld_mean"] for r in g if isinstance(r.get("rld_mean"), (int, float))]
        srd_vals = [r["srd_mean"] for r in g if isinstance(r.get("srd_mean"), (int, float))]
        rew_vals = [r["reward_tail10"] for r in g if isinstance(r.get("reward_tail10"), (int, float))]

        svc_s, rld_s, srd_s, rew_s = stat(svc_vals), stat(rld_vals), stat(srd_vals), stat(rew_vals)

        # simple pass rates
        svc_pass = float(np.mean(np.asarray(svc_vals) >= 0.95)) if svc_vals else None
        rld_pass = float(np.mean(np.asarray(rld_vals) <= 6.0)) if rld_vals else None
        srd_pass = float(np.mean(np.asarray(srd_vals) >= 0.019)) if srd_vals else None

        t3.append(
            {
                "Algorithm": algo.replace("DDPG_PURE", "DDPG"),
                "N_runs": len(g),
                "SVC_CV": None if svc_s is None else svc_s["cv"],
                "RLD_CV": None if rld_s is None else rld_s["cv"],
                "SRD_CV": None if srd_s is None else srd_s["cv"],
                "RewardTail10_CV": None if rew_s is None else rew_s["cv"],
                "PassRate_SVC>=0.95": svc_pass,
                "PassRate_RLD<=6": rld_pass,
                "PassRate_SRD>=0.019": srd_pass,
            }
        )

    write_csv(OUT / "table_comp_3_robustness.csv", t3)

    # markdown bundle
    md = []
    md.append("# Algorithm Comparison Tables (Multi-Dimension)\n\n")
    md.append("Data source: `analysis/run_scan_summary.json`, filtered by train_eps>=100.\n")
    md.append("\n## Table C1 Shared Performance\n\n")
    md.append("| Algo | N | RewardTail10_median | RewardTail10_iqr | SVC_median | SVC_iqr | RLD_median | RLD_iqr | SRD_median | SRD_iqr |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in t1:
        md.append(
            f"| {r['Algorithm']} | {r['N_runs']} | {fmt(r['RewardTail10_median'])} | {fmt(r['RewardTail10_iqr'])} | {fmt(r['SVC_median'])} | {fmt(r['SVC_iqr'])} | {fmt(r['RLD_median'])} | {fmt(r['RLD_iqr'])} | {fmt(r['SRD_median'])} | {fmt(r['SRD_iqr'])} |\n"
        )

    md.append("\n## Table C2 Charging Mechanism\n\n")
    md.append("| Algo | N | CE_med | CE_iqr | QD_med | QD_iqr | ENS_med | ENS_iqr | SOCV_med | SOCV_iqr | ROI_med | ROI_iqr |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in t2:
        md.append(
            f"| {r['Algorithm']} | {r['N_runs']} | {fmt(r['ChargeEff_tail20_median'])} | {fmt(r['ChargeEff_tail20_iqr'])} | {fmt(r['QueueDelay_tail20_median'])} | {fmt(r['QueueDelay_tail20_iqr'])} | {fmt(r['EnergyNotServed_tail20_median'])} | {fmt(r['EnergyNotServed_tail20_iqr'])} | {fmt(r['SOCViolation_tail20_median'])} | {fmt(r['SOCViolation_tail20_iqr'])} | {fmt(r['DispatchROI_tail20_median'])} | {fmt(r['DispatchROI_tail20_iqr'])} |\n"
        )

    md.append("\n## Table C3 Robustness\n\n")
    md.append("| Algo | N | SVC_CV | RLD_CV | SRD_CV | Reward_CV | Pass(SVC>=0.95) | Pass(RLD<=6) | Pass(SRD>=0.019) |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in t3:
        md.append(
            f"| {r['Algorithm']} | {r['N_runs']} | {fmt(r['SVC_CV'])} | {fmt(r['RLD_CV'])} | {fmt(r['SRD_CV'])} | {fmt(r['RewardTail10_CV'])} | {fmt(r['PassRate_SVC>=0.95'])} | {fmt(r['PassRate_RLD<=6'])} | {fmt(r['PassRate_SRD>=0.019'])} |\n"
        )

    (OUT / "comparison_multi_dimension_tables.md").write_text("".join(md), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
