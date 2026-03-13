import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"F:\Claude code\EVRL260207")
DATA = ROOT / "data" / "SIP"
OUT = ROOT / "results" / "marl_supplementary_20260308-212225"


def _load_config():
    return json.loads((DATA / "config_sip.json").read_text(encoding="utf-8"))


def _demand_files():
    files = []
    for p in DATA.glob("od_demand_*.csv"):
        tail = p.stem.split("_")[-1]
        if tail.isdigit():
            files.append(p)
    return sorted(files, key=lambda p: int(p.stem.split("_")[-1]))


def _read_demand_total(fp: Path):
    df = pd.read_csv(fp)
    tcols = [c for c in df.columns if str(c).isdigit()]
    if not tcols:
        return 0.0
    return float(df[tcols].to_numpy(dtype=float).sum())


def fig_dataset_overview(cfg, out_file):
    labels = ["Zones", "Vehicles", "Time Steps", "Parking Slots", "Chargers"]
    parking_total = sum(z["num_parking_slots"] for z in cfg["ZONES"])
    charger_total = sum(len(z.get("charging_slots", [])) for z in cfg["ZONES"])
    vals = [cfg["ZONE_NUM"], cfg["CAR_NUM"], cfg["TIME_NUM"], parking_total, charger_total]

    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    bars = ax.bar(labels, vals, color=["#457b9d", "#2a9d8f", "#e9c46a", "#8d99ae", "#e76f51"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Dataset Overview (SIP)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_charger_distribution(cfg, out_file):
    chargers = [len(z.get("charging_slots", [])) for z in cfg["ZONES"]]
    x = np.arange(len(chargers))
    colors = ["#2a9d8f" if c > 0 else "#e76f51" for c in chargers]

    fig, ax = plt.subplots(figsize=(11.2, 3.6))
    ax.bar(x, chargers, color=colors, width=0.85)
    ax.set_title("Charging Infrastructure by Zone")
    ax.set_xlabel("Zone ID")
    ax.set_ylabel("# Chargers")
    ax.set_xticks(np.arange(0, len(chargers), 5))
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.01, 0.95, f"zones without chargers: {sum(1 for c in chargers if c==0)}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_distance_profile(out_file):
    dist = pd.read_csv(DATA / "od_demand_dist.csv", header=None).values.astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    im = axes[0].imshow(dist, cmap="YlOrRd", aspect="auto")
    axes[0].set_title("OD Distance Matrix")
    axes[0].set_xlabel("Destination Zone")
    axes[0].set_ylabel("Origin Zone")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    vals = dist.flatten()
    axes[1].hist(vals, bins=25, color="#457b9d", alpha=0.85)
    axes[1].axvline(np.percentile(vals, 50), color="#2a9d8f", linestyle="--", label="P50")
    axes[1].axvline(np.percentile(vals, 90), color="#e76f51", linestyle="--", label="P90")
    axes[1].set_title("Distance Distribution")
    axes[1].set_xlabel("Distance")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_demand_file_totals(out_file):
    files = _demand_files()
    ids = [int(p.stem.split("_")[-1]) for p in files]
    totals = [_read_demand_total(p) for p in files]
    colors = ["#e76f51" if t == 0 else "#2a9d8f" for t in totals]

    fig, ax = plt.subplots(figsize=(11.2, 3.8))
    ax.bar(ids, totals, color=colors, width=0.8)
    ax.set_title("Total Demand per Demand File")
    ax.set_xlabel("Demand file index (od_demand_k)")
    ax.set_ylabel("Total demand")
    ax.grid(axis="y", alpha=0.25)
    zero_ids = [i for i, t in zip(ids, totals) if t == 0]
    ax.text(0.01, 0.95, f"zero-demand files: {zero_ids}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_background_notes(cfg, out_file):
    dist = pd.read_csv(DATA / "od_demand_dist.csv", header=None).values.astype(float)
    files = _demand_files()
    totals = [_read_demand_total(p) for p in files]
    zero_files = [p.name for p, t in zip(files, totals) if t == 0]
    charger_total = sum(len(z.get("charging_slots", [])) for z in cfg["ZONES"])
    no_charger = [z["zone_id"] for z in cfg["ZONES"] if len(z.get("charging_slots", [])) == 0]

    lines = []
    lines.append("# Dataset Background Notes\n")
    lines.append(f"- Zones: {cfg['ZONE_NUM']}, Vehicles: {cfg['CAR_NUM']}, Time steps: {cfg['TIME_NUM']}\n")
    lines.append(f"- Total chargers: {charger_total}; zones without chargers: {no_charger}\n")
    lines.append(
        f"- Distance stats: mean={float(np.mean(dist)):.3f}, p50={float(np.percentile(dist,50)):.3f}, p90={float(np.percentile(dist,90)):.3f}\n"
    )
    lines.append(f"- Demand files: {len(files)}; zero-demand files: {zero_files}\n")
    lines.append("\n## Suggested text for experiment setup\n")
    lines.append("- We use the SIP dataset with 50 zones and 200 SAEVs over 26 decision intervals.\n")
    lines.append("- Charging resources are unevenly distributed, with two zones lacking charging piles.\n")
    lines.append("- OD distances are heterogeneous, indicating nontrivial relocation costs.\n")
    lines.append("- We explicitly report demand slices with zero total demand and handle them transparently in evaluation.\n")
    out_file.write_text("".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _load_config()

    fig_dataset_overview(cfg, OUT / "figB1_dataset_overview.png")
    fig_charger_distribution(cfg, OUT / "figB2_charger_distribution.png")
    fig_distance_profile(OUT / "figB3_distance_profile.png")
    fig_demand_file_totals(OUT / "figB4_demand_file_totals.png")
    write_background_notes(cfg, OUT / "dataset_background_notes.md")
    print(OUT)


if __name__ == "__main__":
    main()
