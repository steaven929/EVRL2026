from specific.dclily_elec import DclilyEnv
import numpy as np

env = DclilyEnv()
state = env.reset()

print('=== 初始化检查 ===')
print('车辆总数:', len(env.car))

# 检查初始充电桩占用
occupied_count = 0
for i in range(len(env.car)):
    if env.car[i][3] != -1:
        occupied_count += 1

print('初始充电桩占用车辆数:', occupied_count)
print('初始SOC (前5辆):', [env.car[i][1] for i in range(min(5, len(env.car)))])

# 检查充电桩状态
for zone_id, chargers in env.charging_stations.items():
    occupied = sum(1 for ch in chargers if ch['occupied_by'] is not None)
    print(f'Zone {zone_id}: {len(chargers)} chargers, {occupied} occupied')

print('\n测试通过！')
