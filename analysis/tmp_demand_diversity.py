import pandas as pd
from pathlib import Path


root = Path(r"F:\Claude code\EVRL260207\data\SIP")
files = sorted([p for p in root.glob("od_demand_*.csv") if p.stem.split("_")[-1].isdigit()], key=lambda p: int(p.stem.split("_")[-1]))

totals = []
zeros = []
for fp in files:
    df = pd.read_csv(fp)
    tcols = [c for c in df.columns if str(c).isdigit()]
    s = float(df[tcols].to_numpy(dtype=float).sum()) if tcols else 0.0
    totals.append((fp.name, s))
    if s == 0:
        zeros.append(fp.name)

uniq = sorted(set(round(s, 2) for _, s in totals))
print("unique_totals_count", len(uniq))
print("unique_totals", uniq[:10])
print("zero_files", zeros)
print("first5", totals[:5])
