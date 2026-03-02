import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _parse_list(raw: str):
    return [x.strip() for x in raw.split(",") if x.strip()]


def _load_npy(path: Path):
    if not path.exists():
        return None
    arr = np.load(path, allow_pickle=True)
    return np.asarray(arr, dtype=float).reshape(-1)


def _safe_mean(arr):
    if arr is None or arr.size == 0:
        return np.nan
    return float(np.mean(arr))


def _safe_last(arr):
    if arr is None or arr.size == 0:
        return np.nan
    return float(arr[-1])


def _safe_mean_last(arr, k):
    if arr is None or arr.size == 0:
        return np.nan
    kk = min(k, arr.size)
    return float(np.mean(arr[-kk:]))


def _safe_max(arr):
    if arr is None or arr.size == 0:
        return np.nan
    return float(np.max(arr))


def _collect_root_artifacts(workdir: Path, since_ts: float):
    exts = {".npy", ".png", ".pth"}
    files = []
    for p in workdir.iterdir():
        if p.is_file() and p.suffix.lower() in exts and p.stat().st_mtime >= since_ts - 1:
            files.append(p)
    return files


def _find_latest_marl_output(workdir: Path, since_ts: float):
    out_root = workdir / "outputs"
    if not out_root.exists():
        return None
    cands = [
        p for p in out_root.iterdir()
        if p.is_dir() and p.name.startswith("marl_") and p.stat().st_mtime >= since_ts - 1
    ]
    if not cands:
        return None
    cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0]


def _summarize_run(run_dir: Path, algo: str, seed: int):
    rewards = _load_npy(run_dir / "rewards.npy")
    eval_rewards = _load_npy(run_dir / "eval_rewards.npy")
    marl_test_rewards = _load_npy(run_dir / "marl_test_rewards.npy")
    svc = _load_npy(run_dir / "svc_history.npy")
    rld = _load_npy(run_dir / "rld_history.npy")
    srd = _load_npy(run_dir / "srd_history.npy")
    critic1 = _load_npy(run_dir / "critic1_losses.npy")
    critic_losses = _load_npy(run_dir / "critic_losses.npy")

    return {
        "algo": algo,
        "seed": seed,
        "final_reward": _safe_last(rewards),
        "best_reward": _safe_max(rewards),
        "reward_last50_mean": _safe_mean_last(rewards, 50),
        "eval_last": _safe_last(eval_rewards),
        "eval_mean": _safe_mean(eval_rewards),
        "marl_test_mean": _safe_mean(marl_test_rewards),
        "svc_last10_mean": _safe_mean_last(svc, 10),
        "rld_last10_mean": _safe_mean_last(rld, 10),
        "srd_last10_mean": _safe_mean_last(srd, 10),
        "critic1_last100_mean": _safe_mean_last(critic1, 100),
        "critic_last100_mean": _safe_mean_last(critic_losses, 100),
    }


def _aggregate(rows, key):
    vals = [r[key] for r in rows if np.isfinite(r[key])]
    if not vals:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals))


def _plot_summary_bars(algo_summary, output_root: Path, stamp: str):
    algos = [r["algo"] for r in algo_summary]
    if not algos:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    specs = [
        ("reward_last50_mean", "reward_last50_std", "Reward Last50 Mean", "higher"),
        ("eval_mean", "eval_mean_std", "Eval Mean (single-agent)", "higher"),
        ("svc_last10_mean", "svc_last10_std", "SVC Last10 Mean", "higher"),
        ("rld_last10_mean", "rld_last10_std", "RLD Last10 Mean", "lower"),
    ]

    x = np.arange(len(algos))
    for ax, (m_key, s_key, title, pref) in zip(axes, specs):
        means = [r.get(m_key, np.nan) for r in algo_summary]
        stds = [r.get(s_key, np.nan) for r in algo_summary]
        ax.bar(x, means, yerr=stds, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=15)
        ax.set_title(f"{title} ({pref} is better)")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = output_root / f"comparison_summary_{stamp}.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_reward_curves(per_run_rows, output_root: Path, stamp: str):
    algo_to_curves = {}
    for row in per_run_rows:
        run_dir = Path(row.get("run_dir", ""))
        rewards = _load_npy(run_dir / "rewards.npy")
        if rewards is None or rewards.size == 0:
            continue
        algo_to_curves.setdefault(row["algo"], []).append(rewards)

    if not algo_to_curves:
        return

    plt.figure(figsize=(12, 6))
    for algo, curves in sorted(algo_to_curves.items()):
        min_len = min(len(c) for c in curves)
        if min_len <= 1:
            continue
        mat = np.vstack([c[:min_len] for c in curves])
        mean = np.mean(mat, axis=0)
        std = np.std(mat, axis=0)
        x = np.arange(min_len)
        plt.plot(x, mean, linewidth=2.0, label=algo)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.title("Reward Curves (mean±std across seeds)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = output_root / f"comparison_reward_curves_{stamp}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()


def _run_single_algo(workdir: Path, output_root: Path, algo: str, seed: int, extra_args):
    run_dir = output_root / f"single_{algo.lower()}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "main.py", "--algo_name", algo, "--seed", str(seed)] + extra_args
    print(f"[RUN] {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=workdir)
    elapsed = time.time() - t0

    meta = {
        "runner": "main.py",
        "algo": algo,
        "seed": seed,
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    for p in _collect_root_artifacts(workdir, t0):
        shutil.copy2(p, run_dir / p.name)

    if proc.returncode != 0:
        return None
    row = _summarize_run(run_dir, algo, seed)
    row["run_dir"] = str(run_dir)
    return row


def _run_marl(workdir: Path, output_root: Path, label: str, seed: int, extra_args):
    run_dir = output_root / f"marl_{label.lower()}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "main_marl.py", "--seed", str(seed)] + extra_args
    print(f"[RUN] {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=workdir)
    elapsed = time.time() - t0

    meta = {
        "runner": "main_marl.py",
        "algo": label,
        "seed": seed,
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    latest_marl = _find_latest_marl_output(workdir, t0)
    if latest_marl is not None:
        dst = run_dir / "marl_outputs"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(latest_marl, dst)
        for p in dst.iterdir():
            if p.is_file() and p.suffix.lower() in {".npy", ".png", ".pth"}:
                shutil.copy2(p, run_dir / p.name)

    for p in _collect_root_artifacts(workdir, t0):
        shutil.copy2(p, run_dir / p.name)

    if proc.returncode != 0:
        return None
    row = _summarize_run(run_dir, label, seed)
    row["run_dir"] = str(run_dir)
    return row


def main():
    parser = argparse.ArgumentParser(description="Unified baseline runner: single-agent + MARL")
    parser.add_argument("--single_algos", default="DDPG,SAC,TD3", help="Comma-separated single-agent algos")
    parser.add_argument("--include_marl", default="true", help="true/false")
    parser.add_argument("--marl_label", default="MARL", help="Label used in summary for MARL")
    parser.add_argument("--seeds", default="2,7,11", help="Comma-separated seed list")
    parser.add_argument("--output_root", default="outputs/compare_baseline2", help="Output directory")
    parser.add_argument("--single_extra_args", default="", help="Extra args for main.py, comma-separated")
    parser.add_argument("--marl_extra_args", default="", help="Extra args for main_marl.py, comma-separated")
    args = parser.parse_args()

    workdir = Path(__file__).resolve().parent
    output_root = (workdir / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    include_marl = args.include_marl.strip().lower() in {"1", "true", "yes", "y"}
    single_algos = _parse_list(args.single_algos)
    seeds = [int(x) for x in _parse_list(args.seeds)]
    single_extra_args = _parse_list(args.single_extra_args)
    marl_extra_args = _parse_list(args.marl_extra_args)

    per_run_rows = []

    for algo in single_algos:
        for seed in seeds:
            row = _run_single_algo(workdir, output_root, algo, seed, single_extra_args)
            if row is not None:
                per_run_rows.append(row)

    if include_marl:
        for seed in seeds:
            row = _run_marl(workdir, output_root, args.marl_label, seed, marl_extra_args)
            if row is not None:
                per_run_rows.append(row)

    if not per_run_rows:
        print("No successful runs. Nothing to summarize.")
        return

    stamp = time.strftime("%Y%m%d-%H%M%S")
    per_run_csv = output_root / f"per_run_summary_{stamp}.csv"
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
        "marl_test_mean",
        "svc_last10_mean",
        "rld_last10_mean",
        "srd_last10_mean",
        "critic1_last100_mean",
        "critic_last100_mean",
    ]

    algo_summary = []
    for algo in sorted(set(r["algo"] for r in per_run_rows)):
        rows = [r for r in per_run_rows if r["algo"] == algo]
        row = {"algo": algo, "runs": len(rows)}
        for m in metrics:
            mean_v, std_v = _aggregate(rows, m)
            row[f"{m}_mean"] = mean_v
            row[f"{m}_std"] = std_v
        algo_summary.append(row)

    summary_csv = output_root / f"algo_summary_{stamp}.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(algo_summary[0].keys()))
        writer.writeheader()
        writer.writerows(algo_summary)

    _plot_summary_bars(algo_summary, output_root, stamp)
    _plot_reward_curves(per_run_rows, output_root, stamp)

    print(f"Saved: {per_run_csv}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
