import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"F:\Claude code\EVRL260207")
DATA = ROOT / "data" / "SIP"
OUT = ROOT / "results" / "marl_supplementary_20260308-212225"


def demand_files():
    files = []
    for p in DATA.glob("od_demand_*.csv"):
        tail = p.stem.split("_")[-1]
        if tail.isdigit():
            files.append(p)
    return sorted(files, key=lambda p: int(p.stem.split("_")[-1]))


def load_aggregate_tensor(zones: int):
    files = demand_files()
    all_m = []
    for fp in files:
        df = pd.read_csv(fp)
        tcols = [c for c in df.columns if str(c).isdigit()]
        if not tcols:
            continue
        t = len(tcols)
        m = np.zeros((t, zones, zones), dtype=float)
        for _, row in df.iterrows():
            o = int(row["O"])
            d = int(row["D"])
            if 0 <= o < zones and 0 <= d < zones:
                m[:, o, d] = row[tcols].to_numpy(dtype=float)
        all_m.append(m)
    if not all_m:
        return None
    return np.mean(np.stack(all_m, axis=0), axis=0)


def fig_demand_time_series(m, out_file):
    total_t = m.sum(axis=(1, 2))
    x = np.arange(len(total_t))
    fig, ax = plt.subplots(figsize=(10.2, 3.8))
    ax.plot(x, total_t, linewidth=2.2, color="#2a9d8f")
    ax.fill_between(x, 0, total_t, alpha=0.2, color="#2a9d8f")
    ax.set_title("Demand Time Profile (mean over demand files)")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Total demand")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_od_heatmap(m, out_file):
    od = m.sum(axis=0)
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    im = ax.imshow(od, cmap="magma", aspect="auto")
    ax.set_title("Aggregated OD Demand Heatmap")
    ax.set_xlabel("Destination zone")
    ax.set_ylabel("Origin zone")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_zone_in_out_rank(m, out_file):
    od = m.sum(axis=0)
    outflow = od.sum(axis=1)
    inflow = od.sum(axis=0)
    idx = np.arange(len(outflow))
    order_out = np.argsort(outflow)[::-1]
    order_in = np.argsort(inflow)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    axes[0].bar(idx[:15], outflow[order_out[:15]], color="#457b9d")
    axes[0].set_xticks(idx[:15])
    axes[0].set_xticklabels(order_out[:15], rotation=0)
    axes[0].set_title("Top-15 Outflow Zones")
    axes[0].set_xlabel("Zone ID")
    axes[0].set_ylabel("Demand volume")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(idx[:15], inflow[order_in[:15]], color="#e76f51")
    axes[1].set_xticks(idx[:15])
    axes[1].set_xticklabels(order_in[:15], rotation=0)
    axes[1].set_title("Top-15 Inflow Zones")
    axes[1].set_xlabel("Zone ID")
    axes[1].set_ylabel("Demand volume")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_supply_demand_ratio(m, out_file):
    conf = json.loads((DATA / "config_sip.json").read_text(encoding="utf-8"))
    zones = conf["ZONE_NUM"]
    cars = conf["CAR_NUM"]
    base_supply = cars / zones

    od = m.sum(axis=0)
    outflow = od.sum(axis=1)
    norm_demand = outflow / (np.mean(outflow) + 1e-8)
    ratio = base_supply / (norm_demand + 1e-8)

    fig, ax = plt.subplots(figsize=(10.8, 3.8))
    ax.bar(np.arange(zones), ratio, color="#8d99ae")
    ax.axhline(1.0, linestyle="--", color="#2a9d8f", linewidth=1.5)
    ax.set_title("Relative Supply-to-Demand Index by Zone")
    ax.set_xlabel("Zone ID")
    ax.set_ylabel("Supply / Normalized demand")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_notes(m, out_file):
    od = m.sum(axis=0)
    outflow = od.sum(axis=1)
    inflow = od.sum(axis=0)
    total_t = m.sum(axis=(1, 2))
    lines = []
    lines.append("# Extra Dataset Figures Notes\n")
    lines.append(f"- Mean time-slice demand: {float(np.mean(total_t)):.3f}\n")
    lines.append(f"- Peak/valley demand: {float(np.max(total_t)):.3f} / {float(np.min(total_t)):.3f}\n")
    lines.append(f"- Top outflow zone IDs: {np.argsort(outflow)[-5:][::-1].tolist()}\n")
    lines.append(f"- Top inflow zone IDs: {np.argsort(inflow)[-5:][::-1].tolist()}\n")
    lines.append("\nUse Fig B5-B8 in experiment setup or appendix for demand structure explanation.\n")
    out_file.write_text("".join(lines), encoding="utf-8")


def main():
    conf = json.loads((DATA / "config_sip.json").read_text(encoding="utf-8"))
    m = load_aggregate_tensor(conf["ZONE_NUM"])
    if m is None:
        raise RuntimeError("No valid demand files parsed")

    OUT.mkdir(parents=True, exist_ok=True)
    fig_demand_time_series(m, OUT / "figB5_demand_time_profile.png")
    fig_od_heatmap(m, OUT / "figB6_od_heatmap.png")
    fig_zone_in_out_rank(m, OUT / "figB7_zone_in_out_rank.png")
    fig_supply_demand_ratio(m, OUT / "figB8_supply_demand_index.png")
    write_notes(m, OUT / "dataset_background_extra_notes.md")
    print(OUT)


if __name__ == "__main__":
    main()
