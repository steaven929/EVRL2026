import numpy as np
import pandas as pd
from pathlib import Path


root = Path(r"F:\Claude code\EVRL260207\data\SIP")
files = sorted([p for p in root.glob("od_demand_*.csv") if p.stem.split("_")[-1].isdigit()], key=lambda p: int(p.stem.split("_")[-1]))

print("files", len(files))

stack = []
for fp in files:
    df = pd.read_csv(fp)
    tcols = [c for c in df.columns if str(c).isdigit()]
    arr = df[tcols].to_numpy(dtype=float)
    stack.append(arr)
    print(fp.name, "sum", float(arr.sum()), "std", float(arr.std()), "nonzero_ratio", float((arr > 0).mean()))

S = np.stack(stack, axis=0)  # [file, odpair, time]
print("S shape", S.shape)
per_file_total = S.sum(axis=(1, 2))
print("per_file_total std", float(per_file_total.std()))
per_time_total_by_file = S.sum(axis=1)
print("per_time_total_by_file std first10", np.round(per_time_total_by_file.std(axis=0)[:10], 4).tolist())
print("per_time_mean first10", np.round(per_time_total_by_file.mean(axis=0)[:10], 4).tolist())
