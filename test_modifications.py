import sys
sys.path.insert(0, 'specific')
from dclily_elec import DclilyEnv
import numpy as np

# 创建环境
env = DclilyEnv()
state = env.reset()

print('=== 初始化检查 ===')
print(f'车辆总数: {len(env.car)}')
print(f'初始SOC (前5辆): {[env.car[i][1] for i in range(min(5, len(env.car)))]}')
print(f'初始充电桩占用 (前5辆): {[env.car[i][3] for i in range(min(5, len(env.car)))]}')

# 检查充电桩状态
total_occupied = 0
for zone_id, chargers in env.charging_stations.items():
    occupied = sum(1 for ch in chargers if ch['occupied_by'] is not None)
    total_occupied += occupied
    print(f'Zone {zone_id}: {len(chargers)} chargers, {occupied} occupied')

print(f'总充电桩占用数: {total_occupied}')

# 执行一步
action = np.random.rand(12)
next_state, reward, done, info = env.step(action)

print(f'\n=== 执行一步后 ===')
print(f'SOC (前5辆): {[env.car[i][1] for i in range(min(5, len(env.car)))]}')
print(f'Reward: {reward:.2f}')
print(f'Done: {done}')
print('\n测试通过！')
