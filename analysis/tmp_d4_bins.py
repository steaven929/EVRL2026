import csv
from pathlib import Path
import numpy as np


p = Path(r"F:\Claude code\EVRL260207\results\marl_two_dimension_20260312-220209\marl_two_dimension_ranking.csv")
rows = list(csv.DictReader(p.open(encoding="utf-8")))
bins = [(100, 199, "100-199"), (200, 399, "200-399"), (400, 799, "400-799"), (800, 5000, "800+")]

for lo, hi, name in bins:
    g = [r for r in rows if lo <= int(float(r["train_eps"])) <= hi]
    if not g:
        print(name, "n=0")
        continue
    d = float(np.mean([float(r["demand_satisfaction_score"]) for r in g]))
    c = float(np.mean([float(r["charging_satisfaction_score"]) for r in g]))
    b = float(np.mean([float(r["balanced_score"]) for r in g]))
    print(name, "n", len(g), "demand", round(d, 4), "charging", round(c, 4), "balanced", round(b, 4))
