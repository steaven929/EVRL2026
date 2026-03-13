from pathlib import Path


runs = {
    "DDPG_PURE": Path(r"F:\Claude code\EVRL260207\output\20260307-212745_DDPG_PURE_1000ep"),
    "SAC": Path(r"F:\Claude code\EVRL260207\output2\compare_baseline2\single_sac_seed2"),
    "TD3_700": Path(r"F:\Claude code\EVRL260207\output2\compare_baseline2\run_20260306-010650\single_td3_seed7"),
    "TD3_350": Path(r"F:\Claude code\EVRL260207\output2\compare_baseline2\run_20260304-230706-\single_td3_seed7"),
}

for name, p in runs.items():
    print(f"\n[{name}] {p}")
    if not p.exists():
        print("missing dir")
        continue
    npys = sorted([x.name for x in p.glob("*.npy")])
    print("npy_count", len(npys))
    for n in npys:
        print(" -", n)
