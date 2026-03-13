import json
from pathlib import Path

import numpy as np


def last10_mean(path: Path):
    if not path.exists():
        return None
    arr = np.load(path)
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return None
    return float(np.mean(arr[-min(10, arr.size) :]))


def main():
    root = Path(r"F:\Claude code\EVRL260207")
    rows = json.loads((root / "analysis" / "run_scan_summary.json").read_text(encoding="utf-8"))
    marl_runs = [r for r in rows if r.get("group") == "MARL" and (r.get("train_eps") or -1) >= 100]

    metrics = {
        "charging_efficiency_history.npy": True,
        "vehicle_utilization_history.npy": True,
        "fleet_health_history.npy": True,
        "zone_coordination_history.npy": True,
        "response_readiness_history.npy": True,
        "charger_occupancy_rate_history.npy": True,
        "charger_queue_delay_history.npy": False,
        "energy_not_served_history.npy": False,
        "safety_soc_violation_count_history.npy": False,
        "charge_shift_success_rate_history.npy": True,
        "weighted_service_rate_history.npy": True,
        "effective_service_rate_history.npy": True,
        "service_quality_index_history.npy": True,
        "dispatch_efficiency_history.npy": True,
        "effective_dispatch_rate_history.npy": True,
        "dispatch_roi_history.npy": True,
    }

    for metric, higher_better in metrics.items():
        vals = []
        for run in marl_runs:
            v = last10_mean(Path(run["dir"]) / metric)
            if isinstance(v, float):
                vals.append((v, run))
        if not vals:
            continue
        vals.sort(key=lambda x: x[0], reverse=higher_better)
        best, run = vals[0]
        print(metric, f"best={best:.6f}", f"eps={run['train_eps']}", f"run={run['run_name']}")


if __name__ == "__main__":
    main()
