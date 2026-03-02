import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def _parse_list(raw: str):
    return [x.strip() for x in raw.split(",") if x.strip()]


def _load_npy(path: Path):
    if not path.exists():
        return None
    arr = np.load(path, allow_pickle=True)
    return np.asarray(arr, dtype=float).reshape(-1)


def _safe_mean_last(arr, k):
    if arr is None or arr.size == 0:
        return np.nan
    kk = min(k, arr.size)
    return float(np.mean(arr[-kk:]))


def _safe_last(arr):
    if arr is None or arr.size == 0:
        return np.nan
    return float(arr[-1])


def _safe_max(arr):
    if arr is None or arr.size == 0:
        return np.nan
    return float(np.max(arr))


def _collect_artifacts(workdir: Path, since_ts: float):
    exts = {".npy", ".png", ".pth"}
    files = []
    for p in workdir.iterdir():
        if p.is_file() and p.suffix.lower() in exts and p.stat().st_mtime >= since_ts - 1:
            files.append(p)
    return files


def _summarize_run(run_dir: Path, algo: str, seed: int):
    rewards = _load_npy(run_dir / "rewards.npy")
    eval_rewards = _load_npy(run_dir / "eval_rewards.npy")
    svc = _load_npy(run_dir / "svc_history.npy")
    rld = _load_npy(run_dir / "rld_history.npy")
    srd = _load_npy(run_dir / "srd_history.npy")
    critic1 = _load_npy(run_dir / "critic1_losses.npy")

    return {
        "algo": algo,
        "seed": seed,
        "final_reward": _safe_last(rewards),
        "best_reward": _safe_max(rewards),
        "reward_last50_mean": _safe_mean_last(rewards, 50),
        "eval_last": _safe_last(eval_rewards),
        "eval_mean": _safe_mean_last(eval_rewards, 999999),
        "svc_last10_mean": _safe_mean_last(svc, 10),
        "rld_last10_mean": _safe_mean_last(rld, 10),
        "srd_last10_mean": _safe_mean_last(srd, 10),
        "critic1_last100_mean": _safe_mean_last(critic1, 100),
    }


def _aggregate(rows, key):
    vals = [r[key] for r in rows if np.isfinite(r[key])]
    if not vals:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals))


def main():
    parser = argparse.ArgumentParser(description="Baseline comparison runner for main.py")
    parser.add_argument("--algos", default="DDPG,SAC", help="Comma-separated algo names")
    parser.add_argument("--seeds", default="2,7,11", help="Comma-separated seed list")
    parser.add_argument("--output_root", default="outputs/compare_baseline1", help="Output directory")
    parser.add_argument("--extra_args", default="", help="Extra args forwarded to main.py")
    args = parser.parse_args()

    workdir = Path(__file__).resolve().parent
    output_root = (workdir / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    algos = _parse_list(args.algos)
    seeds = [int(x) for x in _parse_list(args.seeds)]
    extra = _parse_list(args.extra_args)

    per_run_rows = []
    started_at = time.strftime("%Y%m%d-%H%M%S")

    for algo in algos:
        for seed in seeds:
            run_name = f"{algo.lower()}_seed{seed}_{started_at}"
            run_dir = output_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = [sys.executable, "main.py", "--algo_name", algo, "--seed", str(seed)] + extra
            print(f"[RUN] {' '.join(cmd)}")

            t0 = time.time()
            proc = subprocess.run(cmd, cwd=workdir)
            elapsed = time.time() - t0

            meta = {
                "algo": algo,
                "seed": seed,
                "command": cmd,
                "returncode": proc.returncode,
                "elapsed_sec": elapsed,
            }
            (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

            artifacts = _collect_artifacts(workdir, since_ts=t0)
            for p in artifacts:
                shutil.copy2(p, run_dir / p.name)

            if proc.returncode != 0:
                print(f"[WARN] run failed: {algo} seed={seed}")
                continue

            per_run_rows.append(_summarize_run(run_dir, algo, seed))

    if not per_run_rows:
        print("No successful runs. Nothing to summarize.")
        return

    per_run_csv = output_root / f"per_run_summary_{started_at}.csv"
    with per_run_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_run_rows)

    metrics = [
        "final_reward",
        "best_reward",
        "reward_last50_mean",
        "eval_last",
        "eval_mean",
        "svc_last10_mean",
        "rld_last10_mean",
        "srd_last10_mean",
        "critic1_last100_mean",
    ]

    algo_summary = []
    for algo in algos:
        rows = [r for r in per_run_rows if r["algo"] == algo]
        if not rows:
            continue
        row = {"algo": algo, "runs": len(rows)}
        for m in metrics:
            mean_v, std_v = _aggregate(rows, m)
            row[f"{m}_mean"] = mean_v
            row[f"{m}_std"] = std_v
        algo_summary.append(row)

    summary_csv = output_root / f"algo_summary_{started_at}.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(algo_summary[0].keys()))
        writer.writeheader()
        writer.writerows(algo_summary)

    print(f"Saved: {per_run_csv}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
