import json
from pathlib import Path

import numpy as np


best = json.loads(
    Path(r"F:\Claude code\EVRL260207\results\marl_two_dimension_20260312-220209\best_run.json").read_text(
        encoding="utf-8"
    )
)
p = Path(best["dir"])


def moving(a, w=11):
    w = max(1, min(w, len(a)))
    ap = np.pad(a, (w // 2, w - 1 - w // 2), mode="edge")
    k = np.ones(w, dtype=float) / w
    return np.convolve(ap, k, mode="valid")


def n01(a):
    mn = float(np.min(a))
    mx = float(np.max(a))
    if mx - mn < 1e-12:
        return np.full_like(a, 0.5, dtype=float)
    return (a - mn) / (mx - mn)


svc = moving(np.asarray(np.load(p / "svc_history.npy"), dtype=float))
ens = moving(np.asarray(np.load(p / "energy_not_served_history.npy"), dtype=float))
ce = moving(np.asarray(np.load(p / "charging_efficiency_history.npy"), dtype=float))
qd = moving(np.asarray(np.load(p / "charger_queue_delay_history.npy"), dtype=float))
sv = moving(np.asarray(np.load(p / "safety_soc_violation_count_history.npy"), dtype=float))

n = min(len(svc), len(ens), len(ce), len(qd), len(sv))
svc, ens, ce, qd, sv = svc[:n], ens[:n], ce[:n], qd[:n], sv[:n]
d = 0.6 * n01(svc) + 0.4 * (1.0 - n01(ens))
c = 0.5 * n01(ce) + 0.3 * (1.0 - n01(qd)) + 0.2 * (1.0 - n01(sv))

print("n", n)
print("start", float(d[0]), float(c[0]))
print("end", float(d[-1]), float(c[-1]))
print("delta", float(d[-1] - d[0]), float(c[-1] - c[0]))
print(
    "early/mid/late demand",
    float(np.mean(d[: n // 3])),
    float(np.mean(d[n // 3 : 2 * n // 3])),
    float(np.mean(d[2 * n // 3 :])),
)
print(
    "early/mid/late charging",
    float(np.mean(c[: n // 3])),
    float(np.mean(c[n // 3 : 2 * n // 3])),
    float(np.mean(c[2 * n // 3 :])),
)
