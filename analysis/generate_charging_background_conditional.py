import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"F:\Claude code\EVRL260207")
DATA = ROOT / "data" / "SIP"
OUT = ROOT / "results" / "marl_supplementary_20260308-212225"


def load_data():
    cfg = json.loads((DATA / "config_sip.json").read_text(encoding="utf-8"))
    dist = pd.read_csv(DATA / "od_demand_dist.csv", header=None).values.astype(float)
    z = cfg["ZONE_NUM"]
    chargers = np.array([len(zone.get("charging_slots", [])) for zone in cfg["ZONES"]], dtype=float)

    # demand intensity from raw q-files
    q_dir = DATA / "demand_original"
    q_files = sorted([p for p in q_dir.glob("q_*.csv") if p.stem.split("_")[-1].isdigit()], key=lambda p: int(p.stem.split("_")[-1]))
    outflow = np.zeros(z, dtype=float)
    for fp in q_files:
        df = pd.read_csv(fp, header=None, names=["dataset", "O", "D", "T", "V"])
        df = df[(df["O"] >= 0) & (df["O"] < z)]
        g = df.groupby("O", as_index=True)["V"].sum()
        outflow[g.index.to_numpy(dtype=int)] += g.to_numpy(dtype=float)
    if len(q_files) > 0:
        outflow /= len(q_files)

    return cfg, dist, chargers, outflow


def nearest_charger_distance(dist, chargers):
    z = len(chargers)
    charger_zones = np.where(chargers > 0)[0]
    dmin = np.zeros(z, dtype=float)
    for i in range(z):
        if chargers[i] > 0:
            dmin[i] = 0.0
        else:
            dmin[i] = float(np.min(dist[i, charger_zones])) if charger_zones.size else np.nan
    return dmin


def coeff_var(x):
    x = np.asarray(x, dtype=float)
    m = float(np.mean(x))
    s = float(np.std(x))
    return 0.0 if abs(m) < 1e-12 else s / abs(m)


def plot_accessibility(dmin, chargers, out_file):
    z = np.arange(dmin.size)
    colors = ["#2a9d8f" if c > 0 else "#e76f51" for c in chargers]
    fig, ax = plt.subplots(figsize=(10.8, 3.8))
    ax.bar(z, dmin, color=colors)
    ax.set_title("Nearest-Charger Distance by Zone")
    ax.set_xlabel("Zone ID")
    ax.set_ylabel("Distance to nearest charger")
    ax.set_xticks(np.arange(0, dmin.size, 5))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mismatch(outflow, chargers, dmin, out_file):
    # demand high + low charger access = potential mismatch
    x = outflow
    y = chargers
    c = dmin
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    sc = ax.scatter(x, y, c=c, cmap="YlOrRd", s=46, alpha=0.85)
    ax.set_title("Demand-Charging Mismatch Scatter")
    ax.set_xlabel("Average demand outflow per zone")
    ax.set_ylabel("Chargers per zone")
    ax.grid(alpha=0.25)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Nearest-charger distance")
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, dist, chargers, outflow = load_data()
    dmin = nearest_charger_distance(dist, chargers)

    notes = []
    notes.append("# Conditional Charging Background Plot Decision\n")
    notes.append(f"- zones={cfg['ZONE_NUM']}\n")
    notes.append(f"- chargers unique={sorted(set(chargers.tolist()))}\n")
    notes.append(f"- outflow cv={coeff_var(outflow):.4f}\n")
    notes.append(f"- nearest-charger distance cv={coeff_var(dmin):.4f}\n")

    # thresholds for meaningful heterogeneity
    created = []
    if np.std(dmin) > 1e-6 and np.max(dmin) > 0:
        plot_accessibility(dmin, chargers, OUT / "figB13_charger_accessibility.png")
        created.append("figB13_charger_accessibility.png")
        notes.append("- figB13 created: charger accessibility shows spatial heterogeneity.\n")
    else:
        notes.append("- figB13 skipped: charger accessibility is nearly uniform.\n")

    if coeff_var(outflow) > 0.05 or np.std(chargers) > 1e-6:
        plot_mismatch(outflow, chargers, dmin, OUT / "figB14_demand_charger_mismatch.png")
        created.append("figB14_demand_charger_mismatch.png")
        notes.append("- figB14 created: demand-charger mismatch is informative.\n")
    else:
        notes.append("- figB14 skipped: mismatch indicators are too uniform.\n")

    notes.append(f"- created={created}\n")
    (OUT / "dataset_charging_conditional_notes.md").write_text("".join(notes), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
