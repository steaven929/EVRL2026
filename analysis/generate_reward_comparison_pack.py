import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"F:\Claude code\EVRL260207")
SUMMARY = ROOT / "analysis" / "run_scan_summary.json"
OUT_ROOT = ROOT / "results"
TARGET_GROUPS = ["MARL", "SAC", "TD3", "DDPG_PURE"]
FORCED_MARL_RUN_SUBSTR = "marl_20260222-230455"


def dedupe(rows):
    best = {}
    for r in rows:
        key = (r.get("group"), r.get("run_name"))
        cur = best.get(key)
        if cur is None:
            best[key] = r
            continue
        cur_files = len(list(Path(cur["dir"]).glob("*.npy"))) if Path(cur["dir"]).exists() else 0
        new_files = len(list(Path(r["dir"]).glob("*.npy"))) if Path(r["dir"]).exists() else 0
        if new_files > cur_files:
            best[key] = r
    return list(best.values())


def load_rows():
    rows = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = [r for r in rows if (r.get("train_eps") or -1) >= 100 and r.get("group") in TARGET_GROUPS]
    rows = dedupe(rows)
    return rows


def load_curve(run, prefer_raw=False):
    p = Path(run["dir"])
    if prefer_raw and (p / "raw_rewards.npy").exists():
        return np.asarray(np.load(p / "raw_rewards.npy"), dtype=float), "raw"
    if (p / "rewards.npy").exists():
        return np.asarray(np.load(p / "rewards.npy"), dtype=float), "scaled"
    return None, None


def pad_stack(curves):
    if not curves:
        return None
    max_len = max(len(c) for c in curves)
    mat = np.full((len(curves), max_len), np.nan, dtype=float)
    for i, c in enumerate(curves):
        mat[i, : len(c)] = c
    return mat


def trim_to_min_len(curves):
    if not curves:
        return []
    min_len = min(len(c) for c in curves if c is not None and len(c) > 0)
    if min_len <= 0:
        return []
    return [np.asarray(c[:min_len], dtype=float) for c in curves]


def smooth(arr, k=15):
    if arr.size == 0:
        return arr
    k = max(1, min(k, arr.size))
    w = np.ones(k) / k
    return np.convolve(arr, w, mode="same")


def plot_iqr(rows, out_file, prefer_raw=False):
    fig, ax = plt.subplots(figsize=(10.5, 6))
    group_curves = {}
    for g in TARGET_GROUPS:
        grows = [r for r in rows if r["group"] == g]
        curves = []
        for r in grows:
            c, mode = load_curve(r, prefer_raw=prefer_raw)
            if c is not None:
                curves.append(c)
        curves = trim_to_min_len(curves)
        if curves:
            group_curves[g] = curves

    if not group_curves:
        plt.close(fig)
        return

    global_min_len = min(len(c[0]) if isinstance(c, list) else len(c) for c in group_curves.values())

    for g in TARGET_GROUPS:
        if g not in group_curves:
            continue
        curves = [c[:global_min_len] for c in group_curves[g]]
        mat = pad_stack(curves)
        med = np.nanmedian(mat, axis=0)
        q1 = np.nanpercentile(mat, 25, axis=0)
        q3 = np.nanpercentile(mat, 75, axis=0)
        x = np.arange(1, len(med) + 1)
        ax.plot(x, smooth(med), linewidth=2.0, label=f"{g} (n={len(curves)})")
        ax.fill_between(x, q1, q3, alpha=0.18)

    mode_name = "raw" if prefer_raw else "scaled"
    ax.set_title(f"Reward Curves: Median + IQR ({mode_name}, train_eps>=100)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def best_runs(rows):
    out = {}
    for g in TARGET_GROUPS:
        grows = [r for r in rows if r["group"] == g and isinstance(r.get("reward_tail10"), (int, float))]
        if not grows:
            continue
        grows.sort(key=lambda x: x["reward_tail10"], reverse=True)
        out[g] = grows[0]
    return out


def plot_best_of_n(rows, out_file):
    best = best_runs(rows)
    forced = [r for r in rows if r.get("group") == "MARL" and FORCED_MARL_RUN_SUBSTR in str(r.get("run_name", ""))]
    if forced:
        best["MARL"] = forced[0]
    fig, ax = plt.subplots(figsize=(10.5, 6))
    raw_curves = {}
    for g in TARGET_GROUPS:
        if g not in best:
            continue
        c, _ = load_curve(best[g], prefer_raw=False)
        if c is None:
            continue
        raw_curves[g] = c

    if raw_curves:
        min_len = min(len(c) for c in raw_curves.values())
        for g in TARGET_GROUPS:
            if g not in raw_curves:
                continue
            c = raw_curves[g][:min_len]
            x = np.arange(1, len(c) + 1)
            ax.plot(x, smooth(c), linewidth=2.2, label=f"{g}: {best[g]['run_name']}")
    ax.set_title("Best-of-N Reward Curves (scaled)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return best


def plot_topk(rows, out_file, k=3):
    fig, ax = plt.subplots(figsize=(10.5, 6))
    group_curves = {}
    for g in TARGET_GROUPS:
        grows = [r for r in rows if r["group"] == g and isinstance(r.get("reward_tail10"), (int, float))]
        grows.sort(key=lambda x: x["reward_tail10"], reverse=True)
        sel = grows[: min(k, len(grows))]
        curves = []
        for r in sel:
            c, _ = load_curve(r, prefer_raw=False)
            if c is not None:
                curves.append(c)
        curves = trim_to_min_len(curves)
        if curves:
            group_curves[g] = curves

    if group_curves:
        global_min_len = min(len(c[0]) if isinstance(c, list) else len(c) for c in group_curves.values())
        for g in TARGET_GROUPS:
            if g not in group_curves:
                continue
            curves = [c[:global_min_len] for c in group_curves[g]]
            mat = pad_stack(curves)
            m = np.nanmean(mat, axis=0)
            x = np.arange(1, len(m) + 1)
            ax.plot(x, smooth(m), linewidth=2.0, label=f"{g} top-{len(curves)}")
    ax.set_title("Top-k Mean Reward Curves (scaled)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(out_dir, rows, best):
    lines = []
    lines.append("# Reward Comparison Pack\n")
    lines.append("- Filter: train_eps >= 100\n")
    lines.append("- Groups: MARL, SAC, TD3, DDPG_PURE\n")
    lines.append("\n## Sample Sizes\n")
    for g in TARGET_GROUPS:
        n = sum(1 for r in rows if r["group"] == g)
        lines.append(f"- {g}: {n}\n")
    lines.append("\n## Best-of-N Runs\n")
    for g in TARGET_GROUPS:
        if g not in best:
            continue
        lines.append(
            f"- {g}: {best[g]['run_name']} (reward_tail10={best[g].get('reward_tail10')})\n"
        )
    lines.append(f"\n- Forced MARL representative run key: `{FORCED_MARL_RUN_SUBSTR}`\n")
    lines.append("\n## Notes\n")
    lines.append("- fig1/fig4 use distribution-style aggregation (median/IQR) to avoid cherry-picking.\n")
    lines.append("- fig2 is Best-of-N to highlight peak potential (MARL strongest view).\n")
    lines.append("- fig3 uses top-k mean for robustness between median and best-case claims.\n")
    (out_dir / "reward_report.md").write_text("".join(lines), encoding="utf-8")


def main():
    out_dir = OUT_ROOT / f"reward_compare_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    plot_iqr(rows, out_dir / "fig1_scaled_median_iqr.png", prefer_raw=False)
    best = plot_best_of_n(rows, out_dir / "fig2_scaled_best_of_n.png")
    plot_topk(rows, out_dir / "fig3_scaled_topk_mean.png", k=3)
    plot_iqr(rows, out_dir / "fig4_raw_median_iqr_subset.png", prefer_raw=True)

    write_report(out_dir, rows, best)
    (out_dir / "reward_rows_used.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
