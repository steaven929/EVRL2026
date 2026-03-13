import json
from pathlib import Path


rows = json.loads(Path(r"F:\Claude code\EVRL260207\analysis\run_scan_summary.json").read_text(encoding="utf-8"))
names = ["marl_20260306-110926", "marl_20260305-204126", "marl_marl_seed2"]

for n in names:
    m = [r for r in rows if r.get("run_name") == n or n in str(r.get("dir", ""))]
    print("\n", n, "matches", len(m))
    for r in m[:6]:
        print(
            r.get("run_name"),
            "eps",
            r.get("train_eps"),
            "reward_tail10",
            r.get("reward_tail10"),
            "svc",
            r.get("svc_mean"),
            "rld",
            r.get("rld_mean"),
            "dir",
            r.get("dir"),
        )
