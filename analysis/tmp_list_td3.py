import json
from pathlib import Path


rows = json.loads(Path(r"F:\Claude code\EVRL260207\analysis\run_scan_summary.json").read_text(encoding="utf-8"))
td3 = [r for r in rows if r.get("group") == "TD3" and (r.get("train_eps") or -1) >= 100]
print("td3_ge100_total", len(td3))
for r in td3:
    print(r.get("run_name"), r.get("train_eps"), r.get("dir"))
