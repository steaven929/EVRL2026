import numpy as np

print("="*80)
print("REWARD FUNCTION ADJUSTMENT ANALYSIS")
print("="*80)

# Current configuration
print("\n[CURRENT CONFIGURATION]")
charging_bonus_current = 15000
charging_maint_current = 3000
service_divisor_current = 10

# Assumptions for calculation
vehicles_charging = 10
orders_served = 10
steps_per_episode = 26
revenue_per_order = 100

# Current rewards per episode
charging_reward_current = (charging_bonus_current + charging_maint_current) * vehicles_charging * steps_per_episode
service_reward_current = (revenue_per_order * orders_served * steps_per_episode) / service_divisor_current

print(f"\nCharging reward per episode:")
print(f"  ({charging_bonus_current} + {charging_maint_current}) × {vehicles_charging} vehicles × {steps_per_episode} steps")
print(f"  = {charging_reward_current:,}")

print(f"\nService reward per episode:")
print(f"  ({revenue_per_order} × {orders_served} orders × {steps_per_episode} steps) / {service_divisor_current}")
print(f"  = {service_reward_current:,}")

print(f"\nRatio (Charging:Service): {charging_reward_current/service_reward_current:.1f}:1")
print(f"=> Charging is {charging_reward_current/service_reward_current:.0f}x more profitable than service")

# Option 1: Conservative adjustment
print("\n" + "="*80)
print("[OPTION 1: CONSERVATIVE ADJUSTMENT]")
print("="*80)
charging_bonus_opt1 = 7500  # -50%
charging_maint_opt1 = 1500  # -50%
service_divisor_opt1 = 5    # 2x increase

charging_reward_opt1 = (charging_bonus_opt1 + charging_maint_opt1) * vehicles_charging * steps_per_episode
service_reward_opt1 = (revenue_per_order * orders_served * steps_per_episode) / service_divisor_opt1

print(f"\nChanges:")
print(f"  charging_bonus: {charging_bonus_current} → {charging_bonus_opt1} (-50%)")
print(f"  charging_maintenance: {charging_maint_current} → {charging_maint_opt1} (-50%)")
print(f"  service_divisor: /{service_divisor_current} → /{service_divisor_opt1} (2x increase)")

print(f"\nNew rewards per episode:")
print(f"  Charging: {charging_reward_opt1:,}")
print(f"  Service: {service_reward_opt1:,}")
print(f"  Ratio: {charging_reward_opt1/service_reward_opt1:.1f}:1")

print(f"\nImpact:")
print(f"  Charging reward reduced by {(1 - charging_reward_opt1/charging_reward_current)*100:.0f}%")
print(f"  Service reward increased by {(service_reward_opt1/service_reward_current - 1)*100:.0f}%")
print(f"  Ratio improved from {charging_reward_current/service_reward_current:.0f}:1 to {charging_reward_opt1/service_reward_opt1:.0f}:1")

# Option 2: Moderate adjustment (my original recommendation)
print("\n" + "="*80)
print("[OPTION 2: MODERATE ADJUSTMENT (Original Recommendation)]")
print("="*80)
charging_bonus_opt2 = 5000  # -67%
charging_maint_opt2 = 1000  # -67%
service_divisor_opt2 = 2    # 5x increase

charging_reward_opt2 = (charging_bonus_opt2 + charging_maint_opt2) * vehicles_charging * steps_per_episode
service_reward_opt2 = (revenue_per_order * orders_served * steps_per_episode) / service_divisor_opt2

print(f"\nChanges:")
print(f"  charging_bonus: {charging_bonus_current} → {charging_bonus_opt2} (-67%)")
print(f"  charging_maintenance: {charging_maint_current} → {charging_maint_opt2} (-67%)")
print(f"  service_divisor: /{service_divisor_current} → /{service_divisor_opt2} (5x increase)")

print(f"\nNew rewards per episode:")
print(f"  Charging: {charging_reward_opt2:,}")
print(f"  Service: {service_reward_opt2:,}")
print(f"  Ratio: {charging_reward_opt2/service_reward_opt2:.1f}:1")

print(f"\nImpact:")
print(f"  Charging reward reduced by {(1 - charging_reward_opt2/charging_reward_current)*100:.0f}%")
print(f"  Service reward increased by {(service_reward_opt2/service_reward_current - 1)*100:.0f}%")
print(f"  Ratio improved from {charging_reward_current/service_reward_current:.0f}:1 to {charging_reward_opt2/service_reward_opt2:.0f}:1")

# Option 3: Aggressive adjustment
print("\n" + "="*80)
print("[OPTION 3: AGGRESSIVE ADJUSTMENT]")
print("="*80)
charging_bonus_opt3 = 3000  # -80%
charging_maint_opt3 = 600   # -80%
service_divisor_opt3 = 1    # 10x increase

charging_reward_opt3 = (charging_bonus_opt3 + charging_maint_opt3) * vehicles_charging * steps_per_episode
service_reward_opt3 = (revenue_per_order * orders_served * steps_per_episode) / service_divisor_opt3

print(f"\nChanges:")
print(f"  charging_bonus: {charging_bonus_current} → {charging_bonus_opt3} (-80%)")
print(f"  charging_maintenance: {charging_maint_current} → {charging_maint_opt3} (-80%)")
print(f"  service_divisor: /{service_divisor_current} → /{service_divisor_opt3} (10x increase)")

print(f"\nNew rewards per episode:")
print(f"  Charging: {charging_reward_opt3:,}")
print(f"  Service: {service_reward_opt3:,}")
print(f"  Ratio: {charging_reward_opt3/service_reward_opt3:.1f}:1")

print(f"\nImpact:")
print(f"  Charging reward reduced by {(1 - charging_reward_opt3/charging_reward_current)*100:.0f}%")
print(f"  Service reward increased by {(service_reward_opt3/service_reward_current - 1)*100:.0f}%")
print(f"  Ratio improved from {charging_reward_current/service_reward_current:.0f}:1 to {charging_reward_opt3/service_reward_opt3:.0f}:1")

# Comparison table
print("\n" + "="*80)
print("COMPARISON TABLE")
print("="*80)
print(f"\n{'Configuration':<20} {'Charging':<15} {'Service':<15} {'Ratio':<10} {'Balance'}")
print("-"*80)
print(f"{'Current':<20} {charging_reward_current:>14,} {service_reward_current:>14,} {charging_reward_current/service_reward_current:>9.1f}:1  Charging-heavy")
print(f"{'Option 1 (Safe)':<20} {charging_reward_opt1:>14,} {service_reward_opt1:>14,} {charging_reward_opt1/service_reward_opt1:>9.1f}:1  Still charging-heavy")
print(f"{'Option 2 (Moderate)':<20} {charging_reward_opt2:>14,} {service_reward_opt2:>14,} {charging_reward_opt2/service_reward_opt2:>9.1f}:1  More balanced")
print(f"{'Option 3 (Aggressive)':<20} {charging_reward_opt3:>14,} {service_reward_opt3:>14,} {charging_reward_opt3/service_reward_opt3:>9.1f}:1  Service-favored")

# Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print("\n[OPTION 1: CONSERVATIVE] - Recommended if you want to be safe")
print("  Pros:")
print("    - Minimal risk of over-correction")
print("    - Charging still profitable (encourages battery management)")
print("    - Gradual shift toward service")
print("  Cons:")
print("    - May still favor charging too much (45:1 ratio)")
print("    - Might need further adjustment")
print("  Best for: Testing the waters, risk-averse approach")

print("\n[OPTION 2: MODERATE] - Recommended for balanced approach")
print("  Pros:")
print("    - Significant improvement (180:1 → 12:1)")
print("    - Service becomes meaningful contributor")
print("    - Still rewards charging (necessary for operations)")
print("  Cons:")
print("    - Larger change, more risk")
print("    - May need fine-tuning")
print("  Best for: Wanting clear improvement in one iteration")

print("\n[OPTION 3: AGGRESSIVE] - Recommended if service is top priority")
print("  Pros:")
print("    - Service becomes primary objective")
print("    - Closest to real-world economics")
print("    - Forces agent to balance")
print("  Cons:")
print("    - Risk of over-correction")
print("    - Agent might neglect charging")
print("    - Could lead to battery depletion issues")
print("  Best for: If you want service rate >20% quickly")

print("\n[MY RECOMMENDATION]")
print("  Start with OPTION 1 (Conservative):")
print("    - Train 100-200 episodes")
print("    - If service rate reaches 5-10%, success!")
print("    - If still <3%, move to Option 2")
print("  ")
print("  This iterative approach is safer than jumping to Option 2 or 3")
print("  and risking over-correction.")

print("\n" + "="*80)
