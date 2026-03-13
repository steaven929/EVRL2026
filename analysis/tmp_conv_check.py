import numpy as np
from pathlib import Path


def summarize(name, arr, w=100):
    tail = arr[-w:]
    prev = arr[-2 * w : -w]
    x = np.arange(w, dtype=float)
    slope = float(np.polyfit(x, tail, 1)[0])
    print(
        name,
        "len",
        len(arr),
        "tail_mean",
        float(tail.mean()),
        "prev_mean",
        float(prev.mean()),
        "delta",
        float(tail.mean() - prev.mean()),
        "tail_std",
        float(tail.std()),
        "tail_slope",
        slope,
    )


d = Path(r"F:\Claude code\EVRL260207\outputs\marl_20260222-230455")
for fname in ["rewards.npy", "svc_history.npy", "rld_history.npy", "srd_history.npy"]:
    p = d / fname
    if p.exists():
        a = np.asarray(np.load(p), dtype=float)
        summarize(fname, a)
    else:
        print(fname, "missing")
