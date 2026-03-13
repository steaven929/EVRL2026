import json
from pathlib import Path

import numpy as np


rows = json.loads(
    Path(r"F:\Claude code\EVRL260207\results\reward_compare_20260308-174027\reward_rows_used.json").read_text(
        encoding="utf-8"
    )
)

for g in ["MARL", "SAC", "TD3", "DDPG_PURE"]:
    grows = [r for r in rows if r.get("group") == g]
    lens = []
    for r in grows:
        p = Path(r["dir"]) / "rewards.npy"
        if p.exists():
            lens.append(len(np.load(p)))
    if lens:
        print(g, "count", len(lens), "min", min(lens), "max", max(lens), "unique", sorted(set(lens)))
    else:
        print(g, "count", 0)
