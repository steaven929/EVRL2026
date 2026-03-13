import json
from pathlib import Path

import numpy as np
import pandas as pd


root = Path(r"F:\Claude code\EVRL260207\data\SIP")
conf = json.loads((root / "config_sip.json").read_text(encoding="utf-8"))
print("ZONE_NUM", conf["ZONE_NUM"], "CAR_NUM", conf["CAR_NUM"], "TIME_NUM", conf["TIME_NUM"])

zones = conf["ZONES"]
parking = [z["num_parking_slots"] for z in zones]
chargers = [len(z.get("charging_slots", [])) for z in zones]
print("parking per zone unique", sorted(set(parking)), "total", sum(parking))
print(
    "chargers per zone unique",
    sorted(set(chargers)),
    "total",
    sum(chargers),
    "zones_without_charger",
    sum(1 for c in chargers if c == 0),
)

dist = pd.read_csv(root / "od_demand_dist.csv", header=None).values.astype(float)
print(
    "dist shape",
    dist.shape,
    "min",
    float(dist.min()),
    "max",
    float(dist.max()),
    "mean",
    float(dist.mean()),
    "p50",
    float(np.percentile(dist, 50)),
    "p90",
    float(np.percentile(dist, 90)),
)

cong = pd.read_csv(root / "cong.csv", header=None).values.astype(float)
print("cong shape", cong.shape, "min", float(cong.min()), "max", float(cong.max()), "mean", float(cong.mean()))

files = []
for p in root.glob("od_demand_*.csv"):
    tail = p.stem.split("_")[-1]
    if tail.isdigit():
        files.append(p)
files = sorted(files, key=lambda p: int(p.stem.split("_")[-1]))
print("demand files", len(files))

all_file_totals = []
time_totals = []
zone_out = np.zeros(conf["ZONE_NUM"], dtype=float)
zone_in = np.zeros(conf["ZONE_NUM"], dtype=float)

z = conf["ZONE_NUM"]
for fp in files:
    df = pd.read_csv(fp)
    # expected columns: O, D, 0..25
    time_cols = [c for c in df.columns if str(c).isdigit()]
    if len(time_cols) == 0:
        continue
    t = len(time_cols)
    m = np.zeros((t, z, z), dtype=float)
    for _, row in df.iterrows():
        o = int(row["O"])
        d = int(row["D"])
        if 0 <= o < z and 0 <= d < z:
            vals = row[time_cols].to_numpy(dtype=float)
            m[:, o, d] = vals
    all_file_totals.append(float(m.sum()))
    time_totals.append(m.sum(axis=(1, 2)))
    zone_out += m.sum(axis=(0, 2))
    zone_in += m.sum(axis=(0, 1))

all_file_totals = np.asarray(all_file_totals, dtype=float)
print(
    "file total demand mean",
    float(all_file_totals.mean()),
    "std",
    float(all_file_totals.std()),
    "min",
    float(all_file_totals.min()),
    "max",
    float(all_file_totals.max()),
)

time_totals = np.asarray(time_totals, dtype=float)
mean_t = time_totals.mean(axis=0)
print("time demand first10", np.round(mean_t[:10], 2).tolist())
print(
    "time demand peak_t",
    int(np.argmax(mean_t)),
    "peak",
    float(mean_t.max()),
    "valley_t",
    int(np.argmin(mean_t)),
    "valley",
    float(mean_t.min()),
)

idx_out = np.argsort(zone_out)[-5:][::-1]
idx_in = np.argsort(zone_in)[-5:][::-1]
print("zone_out top5", idx_out.tolist(), "vals", np.round(zone_out[idx_out], 1).tolist())
print("zone_in top5", idx_in.tolist(), "vals", np.round(zone_in[idx_in], 1).tolist())


def gini(v):
    x = np.sort(np.asarray(v, dtype=float))
    n = x.size
    s = x.sum()
    if s <= 0:
        return float("nan")
    return float(2 * np.sum((np.arange(1, n + 1)) * x) / (n * s) - (n + 1) / n)


print("zone_out_gini", gini(zone_out))
print("zone_in_gini", gini(zone_in))
