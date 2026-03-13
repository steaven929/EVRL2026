import json
from pathlib import Path

import numpy as np


ROOT = Path(r"F:\Claude code\EVRL260207")
SUMMARY = ROOT / "analysis" / "run_scan_summary.json"


def tail_mean(arr, k=10):
    if arr is None or len(arr) == 0:
        return None
    return float(np.mean(arr[-min(k, len(arr)) :]))


def load_np(path: Path):
    if not path.exists():
        return None
    return np.asarray(np.load(path), dtype=float)


def main():
    rows = json.loads(SUMMARY.read_text(encoding="utf-8"))
    marl = [
        r
        for r in rows
        if r.get("group") == "MARL"
        and (r.get("train_eps") or -1) >= 100
        and isinstance(r.get("reward_tail10"), (int, float))
    ]
    marl.sort(key=lambda x: x["reward_tail10"], reverse=True)

    print("MARL_RUNS", len(marl))
    print("TOP10_BY_reward_tail10")
    for i, r in enumerate(marl[:10], 1):
        print(
            i,
            r.get("run_name"),
            "eps",
            r.get("train_eps"),
            "reward_tail10",
            r.get("reward_tail10"),
            "svc",
            r.get("svc_mean"),
            "rld",
            r.get("rld_mean"),
        )

    best = marl[0] if marl else None
    if best is None:
        return

    p = Path(best["dir"])
    print("\nBEST_RUN", best["run_name"], "DIR", p)

    keys = [
        "rewards.npy",
        "raw_rewards.npy",
        "svc_history.npy",
        "rld_history.npy",
        "srd_history.npy",
        "charging_efficiency_history.npy",
        "charger_queue_delay_history.npy",
        "energy_not_served_history.npy",
        "safety_soc_violation_count_history.npy",
        "dispatch_roi_history.npy",
    ]
    for k in keys:
        a = load_np(p / k)
        if a is None:
            print(k, "missing")
            continue
        print(k, "len", len(a), "last10", tail_mean(a, 10), "mean", float(np.mean(a)))

    # stability of rewards tail
    r = load_np(p / "rewards.npy")
    if r is not None and len(r) >= 200:
        w = 100
        tail = r[-w:]
        prev = r[-2 * w : -w]
        slope = float(np.polyfit(np.arange(w), tail, 1)[0])
        print("reward_tail100_mean", float(np.mean(tail)))
        print("reward_prev100_mean", float(np.mean(prev)))
        print("reward_delta_tail_prev", float(np.mean(tail) - np.mean(prev)))
        print("reward_tail100_std", float(np.std(tail)))
        print("reward_tail100_slope", slope)


if __name__ == "__main__":
    main()
