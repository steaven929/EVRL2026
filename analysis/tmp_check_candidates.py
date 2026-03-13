import numpy as np
from pathlib import Path


cands = [
    Path(r"F:\Claude code\EVRL260207\output2\marl_20260305-204126"),
    Path(r"F:\Claude code\EVRL260207\output2\marl_20260306-110926"),
    Path(r"F:\Claude code\EVRL260207\output2\compare_baseline2\run_20260304-012548\marl_marl_seed2"),
]
files = [
    "rewards.npy",
    "charging_efficiency_history.npy",
    "charger_queue_delay_history.npy",
    "energy_not_served_history.npy",
    "safety_soc_violation_count_history.npy",
    "dispatch_roi_history.npy",
    "response_readiness_history.npy",
    "service_quality_index_history.npy",
]

for p in cands:
    print("\nRUN", p, "exists", p.exists())
    if not p.exists():
        continue
    for f in files:
        fp = p / f
        if fp.exists():
            print(" ", f, "len", len(np.load(fp)))
        else:
            print(" ", f, "missing")
