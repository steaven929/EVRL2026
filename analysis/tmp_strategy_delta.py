import numpy as np
from pathlib import Path


runs = [
    Path(r"F:\Claude code\EVRL260207\output2\compare_baseline2\run_20260304-012548\marl_marl_seed7"),
    Path(r"F:\Claude code\EVRL260207\output2\marl_20260305-204126"),
    Path(r"F:\Claude code\EVRL260207\output2\marl_20260306-110926"),
]
metrics = [
    "charging_efficiency_history.npy",
    "charger_queue_delay_history.npy",
    "energy_not_served_history.npy",
    "safety_soc_violation_count_history.npy",
    "dispatch_roi_history.npy",
    "response_readiness_history.npy",
    "service_quality_index_history.npy",
]

for run in runs:
    print("\nRUN", run)
    for m in metrics:
        a = np.asarray(np.load(run / m), dtype=float)
        n = len(a)
        i1 = n // 3
        i2 = 2 * n // 3
        e = float(np.mean(a[:i1]))
        md = float(np.mean(a[i1:i2]))
        l = float(np.mean(a[i2:]))
        print(m, "early", e, "mid", md, "late", l, "delta_l-e", l - e)
