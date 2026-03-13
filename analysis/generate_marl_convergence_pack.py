import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"F:\Claude code\EVRL260207")
RUN_DIR = ROOT / "outputs" / "marl_20260222-230455"
OUT_ROOT = ROOT / "results"


def ema(arr: np.ndarray, beta: float) -> np.ndarray:
    out = np.zeros_like(arr, dtype=float)
    if arr.size == 0:
        return out
    out[0] = arr[0]
    for i in range(1, arr.size):
        out[i] = beta * out[i - 1] + (1.0 - beta) * arr[i]
    return out


def tail_stats(arr: np.ndarray, window: int) -> dict:
    w = min(window, arr.size)
    tail = arr[-w:]
    prev = arr[-2 * w : -w] if arr.size >= 2 * w else arr[:w]
    x = np.arange(w, dtype=float)
    slope = float(np.polyfit(x, tail, 1)[0]) if w >= 2 else 0.0
    return {
        "window": int(w),
        "tail_mean": float(np.mean(tail)),
        "tail_std": float(np.std(tail)),
        "tail_cv": float(np.std(tail) / (np.mean(tail) + 1e-8)),
        "prev_mean": float(np.mean(prev)),
        "delta_tail_prev": float(np.mean(tail) - np.mean(prev)),
        "tail_slope_per_ep": slope,
    }


def write_stats(out_dir: Path, stats100: dict, stats200: dict):
    payload = {"tail100": stats100, "tail200": stats200}
    (out_dir / "convergence_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# MARL Convergence Summary\n",
        "\n## Tail-100\n",
        f"- mean: {stats100['tail_mean']:.4f}\n",
        f"- std: {stats100['tail_std']:.4f}\n",
        f"- cv: {stats100['tail_cv']:.4f}\n",
        f"- delta(vs previous 100): {stats100['delta_tail_prev']:.4f}\n",
        f"- slope per episode: {stats100['tail_slope_per_ep']:.6f}\n",
        "\n## Tail-200\n",
        f"- mean: {stats200['tail_mean']:.4f}\n",
        f"- std: {stats200['tail_std']:.4f}\n",
        f"- cv: {stats200['tail_cv']:.4f}\n",
        f"- delta(vs previous 200): {stats200['delta_tail_prev']:.4f}\n",
        f"- slope per episode: {stats200['tail_slope_per_ep']:.6f}\n",
    ]
    (out_dir / "convergence_stats.md").write_text("".join(lines), encoding="utf-8")


def main():
    rewards_path = RUN_DIR / "rewards.npy"
    if not rewards_path.exists():
        raise FileNotFoundError(f"Missing rewards file: {rewards_path}")

    rewards = np.asarray(np.load(rewards_path), dtype=float)
    x = np.arange(1, rewards.size + 1)

    ema95 = ema(rewards, 0.95)
    ema98 = ema(rewards, 0.98)
    ma25 = np.convolve(np.pad(rewards, (12, 12), mode="edge"), np.ones(25) / 25.0, mode="valid")

    out_dir = OUT_ROOT / f"marl_convergence_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fig1: full training with smoothing overlays
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.plot(x, rewards, color="#9aa0a6", alpha=0.35, linewidth=1.0, label="Raw")
    ax.plot(x, ma25, color="#1d3557", linewidth=1.8, label="MA(25)")
    ax.plot(x, ema95, color="#2a9d8f", linewidth=2.0, label="EMA(0.95)")
    ax.plot(x, ema98, color="#e76f51", linewidth=2.2, label="EMA(0.98)")
    ax.set_title("MARL Reward Convergence View (Full Training)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_full_raw_ema.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Fig2: zoom on last 400 episodes
    tail_n = min(400, rewards.size)
    xt = np.arange(rewards.size - tail_n + 1, rewards.size + 1)
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.plot(xt, rewards[-tail_n:], color="#9aa0a6", alpha=0.35, linewidth=1.0, label="Raw")
    ax.plot(xt, ma25[-tail_n:], color="#1d3557", linewidth=1.8, label="MA(25)")
    ax.plot(xt, ema95[-tail_n:], color="#2a9d8f", linewidth=2.0, label="EMA(0.95)")
    ax.plot(xt, ema98[-tail_n:], color="#e76f51", linewidth=2.2, label="EMA(0.98)")
    ax.set_title(f"MARL Reward Tail Zoom (Last {tail_n} Episodes)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_tail400_zoom.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Fig3: rolling mean with uncertainty band (window 50)
    w = 50
    means = np.array([np.mean(rewards[max(0, i - w + 1) : i + 1]) for i in range(rewards.size)], dtype=float)
    stds = np.array([np.std(rewards[max(0, i - w + 1) : i + 1]) for i in range(rewards.size)], dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.plot(x, means, color="#264653", linewidth=2.0, label="Rolling mean (50)")
    ax.fill_between(x, means - stds, means + stds, color="#2a9d8f", alpha=0.2, label="±1 std")
    ax.set_title("MARL Rolling Stability (window=50)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_rolling_mean_std.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    s100 = tail_stats(rewards, 100)
    s200 = tail_stats(rewards, 200)
    write_stats(out_dir, s100, s200)

    print(out_dir)


if __name__ == "__main__":
    main()
