import json
from pathlib import Path

import numpy as np


TARGET = Path(r"F:\Claude code\EVRL260207\results\reward_ema98_20260308-182430--------------")
SELECTED = TARGET / "selected_runs.json"
HORIZON = 1000
EMA_BETA = 0.98


def load_np(path: Path):
    if not path.exists():
        return None
    try:
        return np.asarray(np.load(path), dtype=float)
    except Exception:
        return None


def ema(arr: np.ndarray, beta: float):
    out = np.zeros_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, arr.size):
        out[i] = beta * out[i - 1] + (1.0 - beta) * arr[i]
    return out


def to_horizon(arr: np.ndarray, horizon: int):
    n = arr.size
    if n >= horizon:
        return arr[:horizon]
    tail = np.full(horizon - n, arr[-1], dtype=float)
    return np.concatenate([arr, tail])


def tail_mean(a, k=20):
    if a is None or a.size == 0:
        return None
    return float(np.mean(a[-min(k, a.size) :]))


def fmt(v):
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def main():
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    algo_order = ["MARL", "DDPG_PURE", "SAC", "TD3"]

    rows = []
    for algo in algo_order:
        if algo not in selected:
            continue
        r = selected[algo]
        run_dir = Path(r["dir"])

        rewards = load_np(run_dir / "rewards.npy")
        if rewards is None:
            rewards = load_np(run_dir / "raw_rewards.npy")
        if rewards is None:
            continue

        aligned = to_horizon(rewards, HORIZON)
        ema_last = float(ema(aligned, EMA_BETA)[-1])

        svc = load_np(run_dir / "svc_history.npy")
        rld = load_np(run_dir / "rld_history.npy")
        srd = load_np(run_dir / "srd_history.npy")

        ce = load_np(run_dir / "charging_efficiency_history.npy")
        qd = load_np(run_dir / "charger_queue_delay_history.npy")
        ens = load_np(run_dir / "energy_not_served_history.npy")
        socv = load_np(run_dir / "safety_soc_violation_count_history.npy")

        rows.append(
            {
                "Algorithm": "DDPG" if algo == "DDPG_PURE" else algo,
                "RunName": r.get("run_name"),
                "TrainEps": int(r.get("train_eps", len(rewards))),
                "RewardTail10": tail_mean(rewards, 10),
                "RewardTail50": tail_mean(rewards, 50),
                "RewardMean": float(np.mean(rewards)),
                "EMA098_Last@1000": ema_last,
                "SVC_Tail20": tail_mean(svc, 20),
                "RLD_Tail20": tail_mean(rld, 20),
                "SRD_Tail20": tail_mean(srd, 20),
                "ChargeEff_Tail20": tail_mean(ce, 20),
                "QueueDelay_Tail20": tail_mean(qd, 20),
                "EnergyNotServed_Tail20": tail_mean(ens, 20),
                "SOCViolation_Tail20": tail_mean(socv, 20),
                "SourceDir": str(run_dir),
            }
        )

    # CSV
    if rows:
        keys = list(rows[0].keys())
        lines = [",".join(keys)]
        for row in rows:
            vals = []
            for k in keys:
                v = row[k]
                s = fmt(v)
                if "," in s:
                    s = f'"{s}"'
                vals.append(s)
            lines.append(",".join(vals))
        csv_path = TARGET / "comparison_metrics_table.csv"
        try:
            csv_path.write_text("\n".join(lines), encoding="utf-8")
        except PermissionError:
            (TARGET / "comparison_metrics_table_v2.csv").write_text("\n".join(lines), encoding="utf-8")

    # Markdown table
    md = []
    md.append("# Comparison Table (Source-aligned with reward_ema98 figure)\n\n")
    md.append("Data source is strictly the selected runs in `selected_runs.json` under this folder.\n\n")
    md.append("| Algo | TrainEps | RewardTail10 | RewardTail50 | RewardMean | EMA0.98@1000 Last | SVC_tail20 | RLD_tail20 | SRD_tail20 | ChargeEff_tail20 | QueueDelay_tail20 | ENS_tail20 | SOCV_tail20 |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in rows:
        md.append(
            f"| {row['Algorithm']} | {row['TrainEps']} | {fmt(row['RewardTail10'])} | {fmt(row['RewardTail50'])} | {fmt(row['RewardMean'])} | {fmt(row['EMA098_Last@1000'])} | {fmt(row['SVC_Tail20'])} | {fmt(row['RLD_Tail20'])} | {fmt(row['SRD_Tail20'])} | {fmt(row['ChargeEff_Tail20'])} | {fmt(row['QueueDelay_Tail20'])} | {fmt(row['EnergyNotServed_Tail20'])} | {fmt(row['SOCViolation_Tail20'])} |\n"
        )

    md.append("\n## Source Runs\n")
    for row in rows:
        md.append(f"- {row['Algorithm']}: `{row['SourceDir']}`\n")

    md_path = TARGET / "comparison_metrics_table.md"
    try:
        md_path.write_text("".join(md), encoding="utf-8")
    except PermissionError:
        (TARGET / "comparison_metrics_table_v2.md").write_text("".join(md), encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
