import numpy as np
from pathlib import Path


p = Path(r"F:\Claude code\EVRL260207\output2\marl_20260222-230455-")
for f in ["svc_history.npy", "rld_history.npy", "srd_history.npy"]:
    a = np.asarray(np.load(p / f), dtype=float)
    print(f, "len", len(a), "mean", float(a.mean()), "tail20", float(a[-20:].mean()), "tail10", float(a[-10:].mean()), "min", float(a.min()), "max", float(a.max()))
    print("last5", a[-5:])
