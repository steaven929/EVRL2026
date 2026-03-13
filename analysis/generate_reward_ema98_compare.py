import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"F:\Claude code\EVRL260207")
SUMMARY = ROOT / "analysis" / "run_scan_summary.json"
OUT_ROOT = ROOT / "results"

GROUPS = ["MARL", "DDPG_PURE", "SAC", "TD3"]
LABELS = {
    "MARL": "MARL",
    "DDPG_PURE": "DDPG",
    "SAC": "SAC",
    "TD3": "TD3",
}
FORCED_MARL = "marl_20260222-230455"
FORCED_TD3 = "20260308-111016_TD3_1000ep"
FORCED_TD3_DIR = ROOT / "output" / "20260308-111016_TD3_1000ep"
HORIZON = 1000
EMA_BETA = 0.98


def ema(arr: np.ndarray, beta: float) -> np.ndarray:
    out = np.zeros_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, arr.size):
        out[i] = beta * out[i - 1] + (1.0 - beta) * arr[i]
    return out


def load_rows():
    rows = json.loads(SUMMARY.read_text(encoding="utf-8"))
    return [r for r in rows if (r.get("train_eps") or -1) >= 100 and r.get("group") in GROUPS]


def best_run(rows, group):
    cand = [r for r in rows if r["group"] == group and isinstance(r.get("reward_tail10"), (int, float))]
    if not cand:
        return None
    cand.sort(key=lambda x: x["reward_tail10"], reverse=True)
    return cand[0]


def pick_runs(rows):
    selected = {}
    forced = [r for r in rows if r["group"] == "MARL" and FORCED_MARL in str(r.get("run_name", ""))]
    selected["MARL"] = forced[0] if forced else best_run(rows, "MARL")
    selected["DDPG_PURE"] = best_run(rows, "DDPG_PURE")
    selected["SAC"] = best_run(rows, "SAC")
    if FORCED_TD3_DIR.exists():
        selected["TD3"] = {
            "group": "TD3",
            "run_name": FORCED_TD3,
            "dir": str(FORCED_TD3_DIR),
            "train_eps": 1000,
        }
    else:
        forced_td3 = [r for r in rows if r["group"] == "TD3" and FORCED_TD3 in str(r.get("run_name", ""))]
        selected["TD3"] = forced_td3[0] if forced_td3 else best_run(rows, "TD3")
    return {k: v for k, v in selected.items() if v is not None}


def load_reward(run):
    p = Path(run["dir"])
    fp = p / "rewards.npy"
    if fp.exists():
        return np.asarray(np.load(fp), dtype=float)
    fp = p / "raw_rewards.npy"
    if fp.exists():
        return np.asarray(np.load(fp), dtype=float)
    return None


def to_horizon(arr, horizon):
    n = arr.size
    if n >= horizon:
        return arr[:horizon], n, False
    tail = np.full(horizon - n, arr[-1], dtype=float)
    return np.concatenate([arr, tail]), n, True


def main():
    rows = load_rows()
    selected = pick_runs(rows)
    curves = {}
    for g, run in selected.items():
        r = load_reward(run)
        if r is not None and r.size > 0:
            curves[g] = r

    out = OUT_ROOT / f"reward_ema98_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    for g in GROUPS:
        if g not in curves:
            continue
        aligned, original_len, extended = to_horizon(curves[g], HORIZON)
        e = ema(aligned, EMA_BETA)
        x = np.arange(1, HORIZON + 1)
        if extended:
            cut = max(1, original_len)
            ax.plot(x[:cut], e[:cut], linewidth=2.2, label=LABELS[g])
            ax.plot(x[cut - 1 :], e[cut - 1 :], linewidth=2.0, linestyle="--", alpha=0.7)
        else:
            ax.plot(x, e, linewidth=2.2, label=LABELS[g])

    ax.set_title("Reward Comparison with EMA(0.98)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_ylim(bottom=0, top=5000)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_reward_ema098_compare.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    report_lines = [
        "# Reward EMA(0.98) Comparison\n",
        f"- Horizon: {HORIZON}\n",
        "- Shorter runs use last-value extension (dashed tail).\n",
        "\n## Selected Runs\n",
    ]
    for g in GROUPS:
        if g in selected:
            r = selected[g]
            report_lines.append(f"- {LABELS[g]}: {r['run_name']} (train_eps={r['train_eps']})\n")
    (out / "report.md").write_text("".join(report_lines), encoding="utf-8")
    (out / "selected_runs.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    print(out)


if __name__ == "__main__":
    main()
