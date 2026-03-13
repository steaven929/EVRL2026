import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


rows = json.loads(Path(r"F:\Claude code\EVRL260207\analysis\run_scan_summary.json").read_text(encoding="utf-8"))
rows = [r for r in rows if (r.get("train_eps") or -1) >= 100 and r.get("group") in ["MARL", "SAC", "TD3", "DDPG_PURE"]]

# dedupe by (group, run_name), keep row with richer file set
tmp = {}
for r in rows:
    key = (r["group"], r["run_name"])
    score = sum(
        1
        for f in ["rewards.npy", "raw_rewards.npy", "svc_history.npy", "rld_history.npy", "srd_history.npy"]
        if os.path.exists(os.path.join(r["dir"], f))
    )
    if key not in tmp or score > tmp[key][0]:
        tmp[key] = (score, r)
rows = [v[1] for v in tmp.values()]

gmap = defaultdict(list)
for r in rows:
    gmap[r["group"]].append(r)

print("N_by_group", {k: len(v) for k, v in gmap.items()})

for metric in ["svc_mean", "rld_mean", "srd_mean", "reward_tail10"]:
    print("\nmetric", metric)
    for g in ["MARL", "SAC", "TD3", "DDPG_PURE"]:
        vals = [x[metric] for x in gmap.get(g, []) if isinstance(x.get(metric), (int, float))]
        if not vals:
            print(g, "NA")
            continue
        arr = np.asarray(vals, dtype=float)
        print(g, "median", float(np.median(arr)), "mean", float(np.mean(arr)), "n", int(arr.size))

print("\nselected_best_runs")
best = {}
for g in ["MARL", "SAC", "TD3", "DDPG_PURE"]:
    cand = [x for x in gmap.get(g, []) if isinstance(x.get("reward_tail10"), (int, float))]
    if cand:
        cand.sort(key=lambda x: x["reward_tail10"], reverse=True)
        best[g] = cand[0]

for g, r in best.items():
    p = Path(r["dir"]) / "rewards.npy"
    if not p.exists():
        continue
    a = np.asarray(np.load(p), dtype=float)
    target = 0.9 * float(np.max(a))
    hit = np.where(a >= target)[0]
    eps90 = int(hit[0]) + 1 if hit.size else int(a.size)
    print(g, r["run_name"], "eps90", eps90, "len", int(a.size))
