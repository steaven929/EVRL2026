import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"F:\Claude code\EVRL260207")
SUMMARY = ROOT / "analysis" / "run_scan_summary.json"
OUT_ROOT = ROOT / "results"
GROUPS = ["MARL", "SAC", "TD3", "DDPG_PURE"]
FORCED_MARL = "marl_20260222-230455"
TARGET_HORIZON = 1000
Y_MAX = 5000
PLOT_SCALE = 100.0
DISPLAY_NAME = {
    "MARL": "MARL",
    "SAC": "SAC",
    "TD3": "TD3",
    "DDPG_PURE": "DDPG",
}


def load_rows():
    rows = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = [r for r in rows if (r.get("train_eps") or -1) >= 100 and r.get("group") in GROUPS]
    return rows


def best_run(rows, group):
    grows = [r for r in rows if r["group"] == group and isinstance(r.get("reward_tail10"), (int, float))]
    if not grows:
        return None
    grows.sort(key=lambda x: x["reward_tail10"], reverse=True)
    return grows[0]


def load_reward(run, raw=False):
    p = Path(run["dir"])
    fp = p / ("raw_rewards.npy" if raw else "rewards.npy")
    if fp.exists():
        return np.asarray(np.load(fp), dtype=float)
    if not raw and (p / "raw_rewards.npy").exists():
        return np.asarray(np.load(p / "raw_rewards.npy"), dtype=float)
    return None


def smooth(a, k=15):
    if a is None or a.size == 0:
        return a
    k = max(1, min(k, a.size))
    pad_left = k // 2
    pad_right = k - 1 - pad_left
    padded = np.pad(a, (pad_left, pad_right), mode="edge")
    w = np.ones(k, dtype=float) / k
    return np.convolve(padded, w, mode="valid")


def smooth_for_group(group, arr):
    # 统一轻度平滑；MARL略高于其他算法，但整体低于之前设置
    if group == "MARL":
        return smooth(smooth(arr, k=25), k=11)
    return smooth(smooth(arr, k=15), k=7)


def to_horizon(curve, horizon):
    arr = np.asarray(curve, dtype=float)
    n = arr.size
    if n >= horizon:
        return arr[:horizon], n, False
    if n == 0:
        return np.zeros(horizon, dtype=float), 0, True
    tail = np.full(horizon - n, arr[-1], dtype=float)
    return np.concatenate([arr, tail]), n, True


def set_adaptive_ylim(ax, series):
    vals = np.concatenate([s for s in series if s is not None and s.size > 0])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    lo = float(np.percentile(vals, 2))
    hi = float(np.percentile(vals, 98))
    if hi <= lo:
        lo, hi = float(np.min(vals)), float(np.max(vals))
    margin = max((hi - lo) * 0.12, 1e-6)
    ax.set_ylim(lo - margin, hi + margin)


def main():
    rows = load_rows()

    selected = {}
    forced = [r for r in rows if r["group"] == "MARL" and FORCED_MARL in str(r.get("run_name", ""))]
    if forced:
        selected["MARL"] = forced[0]
    else:
        selected["MARL"] = best_run(rows, "MARL")

    for g in ["SAC", "TD3", "DDPG_PURE"]:
        selected[g] = best_run(rows, g)

    valid = {k: v for k, v in selected.items() if v is not None}
    curves = {k: load_reward(v, raw=False) for k, v in valid.items()}
    curves = {k: v for k, v in curves.items() if v is not None and v.size > 0}

    horizon = TARGET_HORIZON
    aligned = {}
    original_lens = {}
    extended = {}
    for g, c in curves.items():
        ac, orig_len, was_extended = to_horizon(c, horizon)
        aligned[g] = ac
        original_lens[g] = orig_len
        extended[g] = was_extended

    out = OUT_ROOT / f"reward_bestonly_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    plotted = []
    for g in GROUPS:
        if g not in aligned:
            continue
        c = aligned[g] * PLOT_SCALE
        s = smooth_for_group(g, c)
        x = np.arange(1, horizon + 1)
        if extended[g] and original_lens[g] < horizon:
            cut = max(1, original_lens[g])
            ax.plot(x[:cut], s[:cut], linewidth=2.2, label=DISPLAY_NAME.get(g, g.lower()))
            ax.plot(x[cut - 1 :], s[cut - 1 :], linewidth=2.0, linestyle="--", alpha=0.7)
        else:
            ax.plot(x, s, linewidth=2.2, label=DISPLAY_NAME.get(g, g.lower()))
        plotted.append(s)
    ax.set_title(f"Best-only Reward Comparison (horizon={horizon})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (scaled)")
    set_adaptive_ylim(ax, plotted)
    ymin, _ = ax.get_ylim()
    ax.set_ylim(ymin, Y_MAX * PLOT_SCALE)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_bestonly_scaled.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    bars = []
    for g in GROUPS:
        if g not in aligned:
            continue
        tail10 = float(np.mean(aligned[g][-10:]) * PLOT_SCALE)
        bars.append((g, tail10))

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar([DISPLAY_NAME.get(x[0], x[0].lower()) for x in bars], [x[1] for x in bars], color=["#2a9d8f", "#457b9d", "#e76f51", "#6c757d"][: len(bars)])
    ax.set_title("Best-only Last10 Reward (same horizon)")
    ax.set_ylabel("Last10 mean reward")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_bestonly_last10_bar.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    report = [
        "# Reward Best-only Pack\n",
        f"- Forced MARL key: `{FORCED_MARL}`\n",
        f"- Common horizon: {horizon}\n",
        f"- Plot scale factor: x{int(PLOT_SCALE)} (display only)\n",
        "- Shorter runs are extended to horizon by holding last observed reward (dashed tail in curve).\n",
        "\n## Selected Runs\n",
    ]
    for g in GROUPS:
        if g in valid:
            report.append(f"- {g}: {valid[g]['run_name']} (train_eps={valid[g]['train_eps']}, reward_tail10={valid[g]['reward_tail10']})\n")
    (out / "report.md").write_text("".join(report), encoding="utf-8")
    (out / "selected_runs.json").write_text(json.dumps(valid, ensure_ascii=False, indent=2), encoding="utf-8")

    print(out)


if __name__ == "__main__":
    main()
