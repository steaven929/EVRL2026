import numpy as np
import pandas as pd
from pathlib import Path


root = Path(r"F:\Claude code\EVRL260207\data\SIP\demand_original")
files = sorted([p for p in root.glob("q_*.csv") if p.stem.split("_")[-1].isdigit()], key=lambda p: int(p.stem.split("_")[-1]))
T = 26
all_t = []

for fp in files:
    df = pd.read_csv(fp, header=None, names=["dataset", "O", "D", "T", "V"])
    df = df[(df["T"] >= 0) & (df["T"] < T)]
    t = np.zeros(T, dtype=float)
    g = df.groupby("T", as_index=True)["V"].sum()
    t[g.index.to_numpy(dtype=int)] = g.to_numpy(dtype=float)
    all_t.append(t)

A = np.stack(all_t, axis=0)
mean_t = A.mean(axis=0)
max_idx = int(np.argmax(mean_t))
print("mean_peak", float(mean_t[max_idx]), "at_t", max_idx)
print("mean_t", np.round(mean_t, 2).tolist())

max_file = A.max(axis=0)
idx2 = int(np.argmax(max_file))
print("single_file_peak", float(max_file[idx2]), "at_t", idx2)
