import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


def load_json(path: Path):
    if not path.exists():
        return {}
    for enc in ("utf-8", "gbk"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception:
            continue
    return {}


def load_npy(path: Path):
    if not path.exists():
        return None
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        return None


def mean(arr):
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return None
    return float(np.mean(a))


def tail_mean(arr, k=10):
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return None
    return float(np.mean(a[-min(k, a.size) :]))


def classify_group(run_name: str, algo_name: str, marl_mode: str):
    name = f"{run_name} {algo_name} {marl_mode}".lower()
    if "marl" in name or marl_mode == "multi":
        return "MARL"
    if "optimized" in name or "non_rl_optimization" in name:
        return "OPTIMIZED"
    if "ddpg_pure" in name or "pureddpg" in name:
        return "DDPG_PURE"
    if "td3" in name:
        return "TD3"
    if "sac" in name:
        return "SAC"
    if "ddpg" in name:
        return "DDPG"
    return "UNKNOWN"


def main():
    root = Path(r"F:\Claude code\EVRL260207")
    result_roots = [root / "output", root / "outputs", root / "output2"]

    rows = []
    for rr in result_roots:
        if not rr.exists():
            continue
        for dirpath, _, filenames in os.walk(rr):
            dp = Path(dirpath)
            files = set(filenames)
            if not (
                "training_metadata.json" in files
                or "rewards.npy" in files
                or "test_rewards.npy" in files
            ):
                continue

            md = load_json(dp / "training_metadata.json")
            rewards = load_npy(dp / "rewards.npy")
            test_rewards = load_npy(dp / "test_rewards.npy")
            eval_rewards = load_npy(dp / "eval_rewards.npy")
            svc = load_npy(dp / "svc_history.npy")
            rld = load_npy(dp / "rld_history.npy")
            srd = load_npy(dp / "srd_history.npy")

            algo = str(md.get("algo_name", "")).strip() or "UNKNOWN"
            marl_mode = str(md.get("marl_mode", "")).strip() if isinstance(md, dict) else ""
            run_name = dp.name
            if isinstance(md, dict):
                train_eps = int(md.get("train_eps", len(rewards) if rewards is not None else -1))
            else:
                train_eps = len(rewards) if rewards is not None else -1

            rows.append(
                {
                    "dir": str(dp),
                    "run_name": run_name,
                    "algo_name": algo,
                    "marl_mode": marl_mode,
                    "group": classify_group(run_name, algo, marl_mode),
                    "train_eps": train_eps,
                    "mean_reward_meta": md.get("mean_reward") if isinstance(md, dict) else None,
                    "final_reward_meta": md.get("final_reward") if isinstance(md, dict) else None,
                    "reward_tail10": tail_mean(rewards, 10),
                    "test_mean": mean(test_rewards),
                    "eval_mean": mean(eval_rewards),
                    "svc_mean": mean(svc),
                    "rld_mean": mean(rld),
                    "srd_mean": mean(srd),
                }
            )

    rows = list({r["dir"]: r for r in rows}.values())
    f100 = [r for r in rows if r["train_eps"] is not None and r["train_eps"] >= 100]

    print("TOTAL_RUNS", len(rows))
    print("RUNS_GE_100", len(f100))
    print("GROUP_COUNTS_GE_100", dict(Counter(r["group"] for r in f100)))

    for metric in ("reward_tail10", "test_mean", "svc_mean", "rld_mean"):
        cand = [r for r in f100 if isinstance(r.get(metric), (int, float))]
        if not cand:
            print("NO_METRIC", metric)
            continue
        reverse = metric != "rld_mean"
        cand.sort(key=lambda x: x[metric], reverse=reverse)
        print("\nTOP", metric)
        for i, r in enumerate(cand[:8], 1):
            print(i, r["group"], r["algo_name"], r["train_eps"], f"{r[metric]:.4f}", r["run_name"])

    print("\nBEST_BY_GROUP_reward_tail10")
    for g in sorted(set(r["group"] for r in f100)):
        cand = [r for r in f100 if r["group"] == g and isinstance(r.get("reward_tail10"), (int, float))]
        if not cand:
            continue
        cand.sort(key=lambda x: x["reward_tail10"], reverse=True)
        r = cand[0]
        print(g, r["train_eps"], f"{r['reward_tail10']:.4f}", r["run_name"])

    print("\nBEST_BY_GROUP_svc_mean")
    for g in sorted(set(r["group"] for r in f100)):
        cand = [r for r in f100 if r["group"] == g and isinstance(r.get("svc_mean"), (int, float))]
        if not cand:
            continue
        cand.sort(key=lambda x: x["svc_mean"], reverse=True)
        r = cand[0]
        print(g, r["train_eps"], f"{r['svc_mean']:.4f}", r["run_name"])

    print("\nBEST_BY_GROUP_rld_mean(lower_is_better)")
    for g in sorted(set(r["group"] for r in f100)):
        cand = [r for r in f100 if r["group"] == g and isinstance(r.get("rld_mean"), (int, float))]
        if not cand:
            continue
        cand.sort(key=lambda x: x["rld_mean"])
        r = cand[0]
        print(g, r["train_eps"], f"{r['rld_mean']:.4f}", r["run_name"])

    out = root / "analysis" / "run_scan_summary.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSAVED", out)


if __name__ == "__main__":
    main()
