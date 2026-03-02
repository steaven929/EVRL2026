# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:10:18 2023

@author: TLab
"""
import random
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import copy


def solveproblem(dist, avail, demand, car, reward, re, car_mem, timestep, real_reward, detail, revise, pre_avail):
    available = copy.deepcopy(avail[1])
    x = len(available)
    m = len(car)
    real_reward_back = np.array([0] * x)
    
    # 创建问题实例
    model = gp.Model("Vehicle_Assignment")
    
    # 设置输出标志，关闭求解输出
    model.setParam('OutputFlag', 0)
    
    # 创建变量
    car_assignment = [[model.addVar(vtype=GRB.BINARY, name=f'car_assignment_{i}_{j}') for j in range(x)] for i in   #当前区域 → 区域 j的任务
                      range(m)]
    zone_assignment = [[model.addVar(lb=0, vtype=GRB.INTEGER, name=f'zone_assignment_{i}_{j}') for j in range(x)] for i
                       in range(x)]
    available_assignment = [model.addVar(lb=0, vtype=GRB.INTEGER, name=f'zone_assignment_{i}') for i in range(x)]
    # 更新模型以添加变量
    model.update()
    
    # 目标函数
    
    # ================= 修改开始2026.1.10 =================
    # 定义惩罚阈值和力度
    SOC_THRESHOLD = 600.0  # 阈值：电量低于600开始惩罚
    PENALTY_FACTOR = 0.0  # Professor 2026.3.1: 暂时禁用SOC惩罚，让车辆能够接单
    
    # 目标函数修改：在原有收益 car_mem[i][2] 的基础上，减去 SOC 惩罚项
    # 逻辑：如果 SOC < 600，系数变为 (原有价值 - 巨大的惩罚)，优化器为了求最大值，就会自动把该车辆的 assignment 置为 0
    model.setObjective(
        gp.quicksum(
            car_assignment[i][j] * (
                    car_mem[i][2] - (
                (SOC_THRESHOLD - car[i][1]) * PENALTY_FACTOR if car[i][1] < SOC_THRESHOLD else 0
            )
            )
            for i in range(m) for j in range(x)
        ) + gp.quicksum(
            zone_assignment[i][j] for i in range(x) for j in range(x) if i != j),
        GRB.MAXIMIZE
    )
    # ================= 修改结束 =================
    # if re > 0 and revise >= 2:
    # model.setObjective(
    #     gp.quicksum(car_assignment[i][j] * car_mem[i][2] for i in range(m) for j in range(x)) + gp.quicksum(
    #         zone_assignment[i][j] for i in range(x) for j in range(x) if i != j), GRB.MAXIMIZE)
    # else:
    # model.setObjective(gp.quicksum(zone_assignment[i][j] for i in range(x) for j in range(x) if i != j), GRB.MAXIMIZE)
    
    # 约束
    for i in range(x):
        model.addConstr(available_assignment[i] >= 0)
        for j in range(x):
            model.addConstr(zone_assignment[i][j] <= demand[i][j])  # 分配少于需求
            for k in range(m):
                if car[k][0] == i and (dist[i][j] > car[k][1] or i == j):  # 扫描所有i-j，找所有起点为i的车，如果到不了则置为0，如果i=j自己到自己也置为0
                    model.addConstr(car_assignment[k][j] == 0)
    
    for i in range(m):
        if car[i][2] <= 0:
            model.addConstr(gp.quicksum(car_assignment[i]) <= 1)  # 不在路上的分配一个位置
        else:
            model.addConstr(gp.quicksum(car_assignment[i]) == 0)  # 在路上的不分配
    
    for i in range(x):
        model.addConstr(available_assignment[i] == available[i] + gp.quicksum(
            zone_assignment[i][j] for j in range(x) if i != j) - gp.quicksum(
            zone_assignment[j][i] for j in range(x) if i != j))
        for j in range(x):
            if i != j:
                # model.addConstr(zone_assignment[i][j] == gp.quicksum(car_assignment[m][j] for m in range(m) if car[m][0] == i))
                model.addConstr(
                    zone_assignment[i][j] ==
                    gp.quicksum(car_assignment[kk][j] for kk in range(m) if car[kk][0] == i)
                )
    
    # 求解问题
    model.optimize()
    
    # 处理解
    
    zone_assignment_np = np.zeros((x, x), dtype=int)
    
    avail[1] = np.array([var.X for var in available_assignment])
    
    if re > 0 and revise >= 3:
        reward_zone = available - avail[1] - pre_avail
        reward_zone_num = np.array([0] * x)
        reward_zone_value = np.array([0.0] * x)
        if detail:
            print(available)
            print(avail[1])
            print(pre_avail)
            print("额外奖励")
            print(reward_zone)
    
    for i in range(m):  # 车数
        for j in range(x):  # zone数
            if car_assignment[i][j].x > 0.5 and car[i][0] != j:
                avail[0][car[i][0]] -= 1
                zone_assignment_np[car[i][0]][j] += 1
                # avail[1][car[i][0]] += 1
                # avail[1][j] -= 1
                car[i][2] = dist[car[i][0]][j]
                real_reward_back[car[i][0]] += re * car[i][2]
                if re < 0:
                    if revise < 2:
                        reward[timestep][car[i][0]] += re * car[i][2]  # 调度惩罚
                    else:
                        reward[timestep][car[i][0]] += re * car[i][2]  # 后续增加电费
                    car_mem[i][0] = car[i][0]
                    car_mem[i][1] = timestep
                    car_mem[i][2] = 0
                    if detail:
                        print("%.1f 车辆%d调度%d -> %d，时间片%d" % (re * car[i][2], i, car[i][0], j, timestep))
                else:
                    if revise >= 3:
                        if reward_zone[j] > 0:
                            reward_zone_num += 1
                            reward_zone_value += re * car[i][2]
                    
                    if car_mem[i][0] != -1:
                        if detail:
                            print("%.1f 车辆%d完成%d -> %d，车辆来自调度区域%d 时间%d" % (
                            re * car[i][2], i, car[i][0], j, car_mem[i][0], car_mem[i][1]))
                        
                        reward[car_mem[i][1]][car_mem[i][0]] += re * car[i][2]
                        car_mem[i][0] = -1
                        # car_mem[i][2]=2
                    else:
                        if revise >= 2:
                            reward[timestep][car[i][0]] += re * car[i][2]
                        if detail:
                            print("%.1f 车辆%d完成%d -> %d" % (re * car[i][2], i, car[i][0], j))
                
                car[i][1] -= car[i][2]
                car[i][0] = j
    
    real_reward += real_reward_back
    
    if revise == 0:
        reward[timestep] = real_reward_back
    
    if re > 0 and revise == 3:
        for i in range(x):
            if reward_zone[i] > 0:
                rrr = reward_zone_value[i] / reward_zone_num[i] * reward_zone[i]
                reward[timestep][i] += rrr
                if detail:
                    print("区域%d让出%d个位置，额外奖励%.1f" % (i, reward_zone[i], rrr))
    
    return car, avail, car_mem, reward, real_reward, zone_assignment_np


def test():
    
    # 示例数据
    
    x = 30
    m = 300
    
    dist = [[0 for i in range(x)] for j in range(x)]
    
    available = [[0 for i in range(x)],[0 for i in range(x)]]
    
    demand = [[0 for i in range(x)] for j in range(x)]
    
    car = [[0, 0, 0] for i in range(m)]
    
    for i in range(x):
        available[1][i] = random.randint(0,10)
        for j in range(x):
            demand[i][j]=random.randint(0,50)
            if i < j :
                dist[i][j]=random.randint(0,20)
            else :
                dist[i][j]=dist[j][i]
    
    for i in range(m):
        car[i][0] = random.randint(0,x)
        car[i][1] = random.randint(0,30)     
        #car[i][2] = random.randint(0,1)  
    print(solveproblem(dist,available,demand,car,[1]*x,1))

def test2():
     
     
     # 示例数据
     
     x = 3
     m = 5
     
     dist = [[0,5,5],[5,0,5],[5,5,0]]
     
     available = [[2,2,1],[1,1,2]]
     
     demand = [[0,5,0],[0,0,0],[5,5,0]]
     
     car = [[0, 0, 0],[0, 5, 0],[1, 5, 0],[1, 5, 0],[2, 5, 0]]
     
     car_mem=np.array([[-1,0,2] for _ in range(m)])
     
     timestep = 0
     
     real_reward=np.array([0]*x)
     
     detail=1
     
     revise=3
     
     pre_avail=[0,0,0]
     
     car, avail, car_mem, reward, real_reward,_=solveproblem(dist,available,demand,car,[[0]*x],1,car_mem,timestep,real_reward,detail,revise,pre_avail)
     # print("返回内容：",
     #       solveproblem(dist, available, demand, car, [[0] * x], 1, car_mem, timestep, real_reward, detail, revise,
     #                    pre_avail))
     #
     print(avail)

def test3():

    # 示例数据
    x = 3
    m = 5

    dist = [[0,5,5],[5,0,5],[5,5,0]]
    available = [[2,2,1],[1,1,2]]
    demand = [[0,5,0],[0,0,0],[5,5,0]]
    car = [[0, 0, 0],[0, 5, 0],[1, 5, 0],[1, 5, 0],[2, 5, 0]]

    car_mem = np.array([[-1,0,2] for _ in range(m)])
    timestep = 0
    real_reward = np.array([0]*x)
    detail = 1
    revise = 3
    pre_avail = [0,0,0]

    # ❶ 先只做一次调用，把返回值完整打印出来
    result = solveproblem(dist, available, demand, car, [[0]*x], 1,
                          car_mem, timestep, real_reward, detail, revise, pre_avail)

    print("solveproblem 返回内容：", result)

    # ❷ 暂时不要解包（注释掉）
    # car, avail, car_mem, reward, real_reward = result


def test2_rl_inference():
    # 1. 初始化环境和智能体
    env = dclily()
    # 假设你的动作空间维度是 3 * ZONE_NUM = 9 (因为 x=3)
    agent = MockAgent(action_dim=3 * env.config['ZONE_NUM'])
    
    # 2. 加载训练好的模型权重
    # !!! 替换为你的模型权重路径 !!!
    MODEL_PATH = "path/to/your/best_ddpg_weights.pth"
    agent.load_weights(MODEL_PATH)
    
    # 3. 环境初始化
    x = 3
    m = 5
    # 由于环境已经初始化了这些变量，这里可以直接使用 env.variable
    # 但为了兼容旧的 test2 结构，我们保留部分初始化。
    # 最好是直接使用 env.reset() 来获取初始状态。
    
    # 初始化环境 (这会设置 dist, available, car 等)
    initial_state = env.reset()
    
    # === RL 决策过程开始 ===
    
    # 获取当前状态
    current_state = initial_state
    
    # 模型选择动作 (explore=False 表示推理模式)
    # 动作是一个连续向量，维度是 3 * ZONE_NUM
    action_vector = agent.select_action(current_state, explore=False)
    
    # === RL 决策过程结束 ===
    
    # 4. 提取 RL 动作中的关键偏好参数
    # 在你的环境中，动作 action 是一个 (3 * ZONE_NUM,) 的向量
    # action[0:ZONE_NUM] = 出发偏好
    # action[ZONE_NUM:2*ZONE_NUM] = 目的地偏好
    # action[2*ZONE_NUM:3*ZONE_NUM] = 充电偏好 (charge_pref)
    
    # ⚠️ 关键步骤：使用 RL 输出的动作来驱动 `solveproblem`
    # 在你的环境代码中，你使用的 revise 和 action_table_true 是分离的。
    # 我们需要找到是哪个参数（如 revise）被 RL 真正影响了。
    
    # 在你的 step 函数中：
    # 1. 动作 action 影响了 action_table_true (OD 调度)
    # 2. 动作 action (charge_pref) 影响了 check_idle_vehicles (充电调度)
    # 3. revise=3 是一个硬编码值，但 solveproblem 的参数中并没有 action_vector 中的任何值
    
    # *********** 修正：解决 RL 动作如何影响 solveproblem 的问题 ***********
    # 你的 solveproblem 签名：gur.solveproblem(self.dist, self.available, action_table_true, ...)
    # 并没有直接使用 'revise'。你需要在 test2 中模拟整个 step 函数的逻辑。
    
    # 5. 模拟 step 逻辑，将 action_vector 拆解并用于调度
    
    # 提取 action_vector 的不同部分
    ZONE_NUM = env.config['ZONE_NUM']
    depart_pref = action_vector[0:ZONE_NUM]
    dest_pref = action_vector[ZONE_NUM:2 * ZONE_NUM]
    charge_pref = action_vector[2 * ZONE_NUM:3 * ZONE_NUM]
    
    # 模拟 env.step() 中的 action_table_true 计算逻辑（简化）
    # 这部分逻辑较为复杂，如果目标是测试，**最简单的方法是直接调用 env.step()**
    
    # 6. 使用 RL 动作执行一步
    new_state, reward, done, info = env.step(action_vector)
    
    print("\n--- RL 推理测试结果 ---")
    print(f"RL 模型选取的动作 (偏好向量): {action_vector}")
    print(f"本次运行的总奖励: {reward}")
    print(f"服务率 (svc): {info.get('svc'):.4f}")
    
    # 打印环境内部的关键变量，用于验证 RL 动作的有效性
    # 注意：env.available[0] 是车辆数
    print(f"更新后的车辆在区数 (avail[0]): {env.available[0]}")
    # env.car 存储了车辆状态，可以检查是否有车辆正在执行任务 (car[i][2] > 0)
    
    # 判定学习结果：如果 reward 显著高于随机策略的平均 reward，则 RL 学习有效。
    # 这是一个合格的 RL 推理测试。

if __name__ == '__main__':
    test2_rl_inference()


if __name__ == '__main__':  
    test2()