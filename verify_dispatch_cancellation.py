# -*- coding: utf-8 -*-
"""
Verify bidirectional dispatch cancellation
"""
import re

# Read RL action file
with open('action/RL-DDPG_actions.txt', 'r', encoding='utf-8') as f:
    content = f.read()

print("="*80)
print("Bidirectional Dispatch Analysis")
print("="*80)

# Extract timestep information
timesteps = re.findall(
    r'时间步 (\d+).*?【服务订单】\s*\n(.*?)【调度动作】\s*\n(.*?)【充电动作】',
    content,
    re.DOTALL
)

print(f"\nAnalyzing {len(timesteps)} timesteps...\n")

dispatch_swaps = []
service_swaps = []

for match in timesteps:
    if len(match) >= 3:
        step = int(match[0])
        service = match[1]
        dispatch = match[2]

        # Check dispatch actions for mutual swaps
        dispatch_dict = {}
        dispatch_lines = [l.strip() for l in dispatch.split('\n') if '区域' in l and '辆' in l]
        for line in dispatch_lines:
            m = re.search(r'区域(\d+)\s*→\s*区域(\d+):\s*(\d+)辆', line)
            if m:
                from_z = int(m.group(1))
                to_z = int(m.group(2))
                count = int(m.group(3))
                key = (from_z, to_z)
                dispatch_dict[key] = count

        # Check for bidirectional dispatch
        for (f, t), count in dispatch_dict.items():
            if (t, f) in dispatch_dict:
                dispatch_swaps.append((step, f, t, count, dispatch_dict[(t, f)]))

        # Check service orders for mutual swaps
        service_dict = {}
        service_lines = [l.strip() for l in service.split('\n') if '区域' in l and '辆' in l]
        for line in service_lines:
            m = re.search(r'区域(\d+)\s*→\s*区域(\d+):\s*(\d+)辆', line)
            if m:
                from_z = int(m.group(1))
                to_z = int(m.group(2))
                count = int(m.group(3))
                key = (from_z, to_z)
                service_dict[key] = service_dict.get(key, 0) + 1

        # Check for bidirectional service
        for (f, t), count in service_dict.items():
            if (t, f) in service_dict:
                service_swaps.append((step, f, t, count, service_dict[(t, f)]))

print("=== Dispatch Actions (Repositioning) ===")
if dispatch_swaps:
    print(f"Found {len(dispatch_swaps)} bidirectional dispatch pairs:")
    for step, f, t, c1, c2 in dispatch_swaps:
        print(f"  Step {step}: Zone {f}<->{t} ({c1} vs {c2} vehicles)")
    print("\nWARNING: Bidirectional dispatch detected!")
    print("The _cancel_bidirectional_dispatch() function may not be working correctly.")
else:
    print("No bidirectional dispatch found.")
    print("The _cancel_bidirectional_dispatch() function is working correctly!")

print("\n=== Service Orders (Customer Demand) ===")
if service_swaps:
    print(f"Found {len(service_swaps)} bidirectional service pairs:")
    for step, f, t, c1, c2 in service_swaps[:5]:
        print(f"  Step {step}: Zone {f}<->{t} ({c1} vs {c2} vehicles)")
    print("\nNote: These are customer demands and cannot be cancelled.")
    print("Mutual swaps in service orders are unavoidable.")
else:
    print("No bidirectional service orders found.")

print("\n" + "="*80)
print("Conclusion")
print("="*80)
if not dispatch_swaps:
    print("Optimization 1 (Reduce mutual swaps) is ALREADY IMPLEMENTED.")
    print("The _cancel_bidirectional_dispatch() function successfully prevents")
    print("bidirectional dispatch actions.")
    print("\nThe mutual swaps you see in the output are from SERVICE ORDERS,")
    print("which are customer demands and cannot be cancelled.")
    print("\nRecommendation: Move to Optimization 2 (Increase proactive dispatch)")
else:
    print("Optimization 1 needs to be fixed.")
    print("Bidirectional dispatch actions are still present.")
