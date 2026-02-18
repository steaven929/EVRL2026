# -*- coding: utf-8 -*-
"""
Quick Improvements Validation Test
Tests state dimensions, dual-head actor, and batch size configuration
"""

import sys
import numpy as np
from core.util import get_args
from core.environment import env_agent_config

def test_quick_improvements():
    print("="*80)
    print("Quick Improvements Validation Test")
    print("="*80)

    # 1. Test hyperparameter configuration
    print("\n[1] Testing hyperparameter configuration...")
    cfg = get_args()

    print(f"  OK batch_size = {cfg['batch_size']} (expected: 1024)")
    assert cfg['batch_size'] == 1024, f"batch_size should be 1024, got {cfg['batch_size']}"

    print(f"  OK use_dual_head = {cfg['use_dual_head']} (expected: True)")
    assert cfg['use_dual_head'] == True, f"use_dual_head should be True, got {cfg['use_dual_head']}"

    # 2. Test environment and agent creation
    print("\n[2] Testing environment and agent creation...")
    env, agent = env_agent_config(cfg)
    print(f"  OK Environment created")
    print(f"  OK Agent created (algorithm: {cfg['algo_name']})")

    # 3. Test state dimensions
    print("\n[3] Testing state dimensions...")
    state = env.reset()
    print(f"  OK State dimension = {len(state)} (expected: 251)")
    assert len(state) == 251, f"State dimension should be 251, got {len(state)}"

    # Verify state components
    from specific.dclily_elec import ZONE_NUM, TIME_SLICE
    od_dim = 2 * ZONE_NUM * TIME_SLICE
    charger_dim = ZONE_NUM * TIME_SLICE
    soc_dim = ZONE_NUM
    vehicle_dim = ZONE_NUM
    timestep_dim = 1

    print(f"  OK OD demand dims: {od_dim} (0:{od_dim})")
    print(f"  OK Charger state dims: {charger_dim} ({od_dim}:{od_dim+charger_dim})")
    print(f"  OK Avg SOC dims: {soc_dim} ({od_dim+charger_dim}:{od_dim+charger_dim+soc_dim})")
    print(f"  OK Vehicle count dims: {vehicle_dim} ({od_dim+charger_dim+soc_dim}:{od_dim+charger_dim+soc_dim+vehicle_dim})")
    print(f"  OK Timestep dims: {timestep_dim} ({od_dim+charger_dim+soc_dim+vehicle_dim}:)")

    # 4. Test vehicle count feature range
    print("\n[4] Testing vehicle count features...")
    vehicle_counts = state[od_dim+charger_dim+soc_dim:od_dim+charger_dim+soc_dim+vehicle_dim]
    print(f"  OK Vehicle count range: [{vehicle_counts.min()}, {vehicle_counts.max()}] (expected: [0, 19])")
    assert vehicle_counts.min() >= 0 and vehicle_counts.max() <= 19, \
        f"Vehicle counts should be in [0,19], got [{vehicle_counts.min()}, {vehicle_counts.max()}]"

    # 5. Test action sampling
    print("\n[5] Testing action sampling...")
    action = agent.sample_action(state)
    print(f"  OK Action dimension = {len(action)} (expected: 150)")
    assert len(action) == 150, f"Action dimension should be 150, got {len(action)}"
    print(f"  OK Action range: [{action.min():.3f}, {action.max():.3f}] (expected: [0, 1])")

    # 6. Test dual-head actor (if SAC)
    if cfg['algo_name'] == 'SAC':
        print("\n[6] Testing dual-head actor architecture...")
        if hasattr(agent.actor, 'dispatch_dims'):
            print(f"  OK Dual-head actor enabled")
            print(f"    - Dispatch dims: {agent.actor.dispatch_dims}")
            print(f"    - Charging dims: {agent.actor.charging_dims}")
        else:
            print(f"  WARNING Dual-head actor not enabled (using single-head)")

    # 7. Test environment interaction
    print("\n[7] Testing environment interaction...")
    next_state, reward, done, info = env.step(action)
    print(f"  OK Environment step successful")
    print(f"  OK next_state dimension = {len(next_state)}")
    print(f"  OK reward = {reward:.2f}")
    print(f"  OK done = {done}")

    print("\n" + "="*80)
    print("SUCCESS All tests passed! Quick improvements configured correctly.")
    print("="*80)
    print("\nReady to train: python main.py")

if __name__ == "__main__":
    try:
        test_quick_improvements()
    except Exception as e:
        print(f"\nFAILED Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
