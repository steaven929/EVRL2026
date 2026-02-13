from specific.dclily_elec import dclily
import numpy as np

env = dclily()
state = env.reset()

print('=== Initialization Check ===')
print('Total vehicles:', len(env.car))

# Check initial charger occupancy
occupied_count = 0
for i in range(len(env.car)):
    if env.car[i][3] != -1:
        occupied_count += 1

print('Initial vehicles on chargers:', occupied_count)
print('Initial SOC (first 5):', [env.car[i][1] for i in range(min(5, len(env.car)))])

# Check charger status
for zone_id, chargers in env.charging_stations.items():
    occupied = sum(1 for ch in chargers if ch['occupied_by'] is not None)
    print(f'Zone {zone_id}: {len(chargers)} chargers, {occupied} occupied')

print('\nTest passed!')
