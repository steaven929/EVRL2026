import numpy as np
import sys
import os
import io

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append('.')

from core.util import get_args
import core.environment as dclilyenv
from gym.envs.registration import register

# Register environment
register(
    id='dclily',
    entry_point='specific.dclily_elec:dclily',
)

# Get config
cfg = get_args()

# Create environment
env, agent = dclilyenv.env_agent_config(cfg)
state = env.reset()

print("=== State Dimension Debug ===")
print(f"State shape: {state.shape}")
print(f"State dtype: {state.dtype}")
print(f"Expected dim: {env.expected_state_dim}")
print()

# Check OD data shape
print("=== OD Data Shape ===")
print(f"od_in shape: {env.od_in.shape}")
print(f"od_out shape: {env.od_out.shape}")
print(f"TIME_NUM: {env.TIME_NUM}")
print(f"TIME_SLICE: {env.TIME_SLICE}")
print(f"ZONE_NUM: {env.ZONE_NUM}")
print()

# Manually construct state to see what's missing
from specific.dclily_elec import TIME_SLICE, ZONE_NUM, OD_MAX, CHARGER_STATES, TIME_NUM

timestep = min(env.episode_timestep, env.od_in.shape[1] - TIME_SLICE)
timestep = max(0, timestep)

od_in2 = np.array(env.od_in[:, timestep:timestep + TIME_SLICE]).T
od_out2 = np.array(env.od_out[:, timestep:timestep + TIME_SLICE]).T
od_part = np.concatenate((od_in2, od_out2), axis=1).ravel()
od_part = np.clip(od_part, 0, OD_MAX - 1).astype(np.int64)

print("=== State Components ===")
print(f"od_part shape: {od_part.shape} (expected: {2*TIME_SLICE*ZONE_NUM})")
print(f"od_in2 shape: {od_in2.shape}")
print(f"od_out2 shape: {od_out2.shape}")
print()

print("=== Expected State Breakdown ===")
print(f"OD demand: 2*TIME_SLICE*ZONE_NUM = 2*{TIME_SLICE}*{ZONE_NUM} = {2*TIME_SLICE*ZONE_NUM}")
print(f"Charger states: TIME_SLICE*ZONE_NUM = {TIME_SLICE}*{ZONE_NUM} = {TIME_SLICE*ZONE_NUM}")
print(f"Avg SOC per zone: ZONE_NUM = {ZONE_NUM}")
print(f"Vehicle counts per zone: ZONE_NUM = {ZONE_NUM}")
print(f"Current timestep: 1")
print(f"Total: {2*TIME_SLICE*ZONE_NUM + TIME_SLICE*ZONE_NUM + ZONE_NUM + ZONE_NUM + 1}")
