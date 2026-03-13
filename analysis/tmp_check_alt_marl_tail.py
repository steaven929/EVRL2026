import numpy as np
from pathlib import Path


p = Path(r"F:\Claude code\EVRL260207\output2\marl_20260306-110926")
for f in ["svc_history.npy", "rld_history.npy", "srd_history.npy"]:
    a = np.asarray(np.load(p / f), dtype=float)
    print(f, "tail20", float(a[-20:].mean()), "mean", float(a.mean()), "last5", a[-5:])
