# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:37:47 2024

@author: TLab
"""

import pandas as pd
import numpy as np

# 设置随机种子以确保结果的可重复性
# np.random.seed(42)

# 初始化需求矩阵的基本参数
num_locations = 4  # 地点数量，0到3
num_times = 12 + 3     # 时间段数量，0到14

# 使用列表来动态构建需求数据
demand_data = []

# 定义需求生成的规则
def generate_demand(min_demand=0, max_demand=5, probability=1.0):
    if np.random.random() < probability:
        return np.random.randint(min_demand, max_demand + 1)
    else:
        return 0

# 生成需求数据
for o in range(num_locations):
    for d in range(num_locations):
        if o != d:  # 跳过从自己到自己的需求
            demand_row = [o, d]
            for t in range(num_times):  # 对每个时间段
                if t < 6:  # 对于前6个时间段
                    if (o in [0, 1] and d in [2, 3]):
                        demand_row.append(generate_demand(3, 5))
                    else:
                        demand_row.append(generate_demand(0, 1, 0.2))
                elif 6 <= t < 12:  # 对于后6个时间段
                    if (o in [2, 3] and d in [0, 1]):
                        demand_row.append(generate_demand(3, 5))
                    else:
                        demand_row.append(generate_demand(0, 1, 0.2))
                else:  # 最后三个时间段
                    demand_row.append(0)  # 添加0表示无需求
            demand_data.append(demand_row)

# 将需求数据转换为pandas DataFrame
columns = ['O', 'D'] + [str(i) for i in range(num_times)]
df = pd.DataFrame(demand_data, columns=columns)

# 保存到CSV文件
df.to_csv('od_demand.csv', index=False)

print("od_demand.csv has been generated successfully.")
