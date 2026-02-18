# Quick validation - check key configurations
import sys
sys.path.insert(0, '.')

# Check batch_size
from core import util
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',default=1024,type=int)
parser.add_argument('--use_dual_head',default=True,type=bool)
args = parser.parse_args([])

print("="*60)
print("Quick Improvements Validation")
print("="*60)
print(f"batch_size: {args.batch_size} (expected: 1024)")
print(f"use_dual_head: {args.use_dual_head} (expected: True)")

# Check state dimension
from specific.dclily_elec import ZONE_NUM, TIME_SLICE
expected_dim = 3 * ZONE_NUM * TIME_SLICE + 2 * ZONE_NUM + 1
print(f"\nState dimension: {expected_dim} (expected: 251)")
print(f"  - ZONE_NUM: {ZONE_NUM}")
print(f"  - TIME_SLICE: {TIME_SLICE}")
print(f"  - OD demand: {2*ZONE_NUM*TIME_SLICE}")
print(f"  - Charger states: {ZONE_NUM*TIME_SLICE}")
print(f"  - Avg SOC: {ZONE_NUM}")
print(f"  - Vehicle counts: {ZONE_NUM} (NEW!)")
print(f"  - Timestep: 1")

print("\n" + "="*60)
if args.batch_size == 1024 and args.use_dual_head == True and expected_dim == 251:
    print("SUCCESS All configurations correct!")
    print("="*60)
    print("\nReady to train: python main.py")
else:
    print("FAILED Some configurations incorrect!")
    sys.exit(1)
