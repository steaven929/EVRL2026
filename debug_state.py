import numpy as np
from specific.dclily_elec import dclily, ZONE_NUM, TIME_SLICE

print(f"ZONE_NUM: {ZONE_NUM}")
print(f"TIME_SLICE: {TIME_SLICE}")
print(f"Expected state dim: {3 * ZONE_NUM * TIME_SLICE + 2 * ZONE_NUM + 1}")

env = dclily()
state = env.reset()

print(f"\nActual state shape: {state.shape}")
print(f"State dtype: {state.dtype}")

# 分析状态各部分
print(f"\nState breakdown:")
print(f"First 100 elements (od_part): {state[:100]}")
print(f"Next 50 elements (charger_states): {state[100:150]}")
if len(state) > 150:
    print(f"Next 50 elements (avg_soc): {state[150:200]}")
if len(state) > 200:
    print(f"Next 50 elements (vehicle_counts): {state[200:250]}")
if len(state) > 250:
    print(f"Last element (timestep): {state[250]}")
