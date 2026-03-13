import json
import os
from collections import Counter
from pathlib import Path


def main():
    root = Path(r"F:\Claude code\EVRL260207")
    rows = json.loads((root / "analysis" / "run_scan_summary.json").read_text(encoding="utf-8"))
    runs = [r for r in rows if (r.get("train_eps") or -1) >= 100]

    metrics = [
        "svc_history.npy",
        "rld_history.npy",
        "srd_history.npy",
        "raw_rewards.npy",
        "marl_test_rewards.npy",
        "test_rewards.npy",
        "charging_efficiency_history.npy",
        "vehicle_utilization_history.npy",
        "fleet_health_history.npy",
        "zone_coordination_history.npy",
        "response_readiness_history.npy",
        "charger_occupancy_rate_history.npy",
        "charger_queue_delay_history.npy",
        "energy_not_served_history.npy",
        "safety_soc_violation_count_history.npy",
        "charge_shift_success_rate_history.npy",
        "weighted_service_rate_history.npy",
        "effective_service_rate_history.npy",
        "service_quality_index_history.npy",
        "dispatch_efficiency_history.npy",
        "effective_dispatch_rate_history.npy",
        "dispatch_roi_history.npy",
    ]

    print("GROUP_COUNTS", dict(Counter(r["group"] for r in runs)))
    for group in sorted(set(r["group"] for r in runs)):
        group_runs = [r for r in runs if r["group"] == group]
        print(f"\nGROUP {group} N={len(group_runs)}")
        for metric in metrics:
            cnt = sum(1 for r in group_runs if os.path.exists(os.path.join(r["dir"], metric)))
            if cnt > 0:
                print(f"{metric}: {cnt}")


if __name__ == "__main__":
    main()
