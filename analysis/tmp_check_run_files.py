import numpy as np
from pathlib import Path


p = Path(r"F:\Claude code\EVRL260207\output2\marl_20260222-230455-")
print("exists", p.exists())
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
for f in files:
    fp = p / f
    if fp.exists():
        print(f, "exists", "len", len(np.load(fp)))
    else:
        print(f, "missing")
