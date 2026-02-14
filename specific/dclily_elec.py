# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:19:07 2024

@author: TLab
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
import random
import copy
import math
import specific.optim as gur
import os
# from collections import Dict
import json


# 相对路径写法    当前文件所在目录：specific/
current_dir = os.path.dirname(os.path.abspath(__file__))

# config.json 就在同一目录
config_path = os.path.join(current_dir, 'config.json')

#print('[DEBUG] Loading config from:', config_path)

with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
    
#-------------1---------------------------
# script_dir = os.path.dirname(os.path.abspath('F:\\AAAA组会\\代码备份\\2501007加了conj参数后备份\\dclily_torch_ddpg_demo基于0425修改\\specific'))
# config_path = os.path.join(script_dir, 'specific', 'config.json')
#
# with open(config_path, 'r') as file:
#     config = json.load(file)

# 加载JSON文件-----22222222222-------------
# with open('.\\specific\\config.json', 'r') as file:
#    config = json.load(file)

PL_NUM = config['PL_NUM']
ZONE_NUM = config['ZONE_NUM']
CAR_NUM = config['CAR_NUM']
TIME_NUM = config['TIME_NUM']
TIME_LENGTH = config['TIME_LENGTH']
TIME_SLICE = config['TIME_SLICE']
OD_MAX = config['OD_MAX']
DIST_MAX = config['DIST_MAX']
car_speed_min = config['car_speed_min']
manzu_min = config['manzu_min']
diaodu_min = config['diaodu_min']
elec_min = config['elec_min']
BATTERY_CAPACITY = config['BATTERY_CAPACITY']
path = config['path']
training_set_number = config['training_set_number']
CHARGER_STATES = config['CHARGER_STATES']  # 充电区域
#NUM_CHARGERS = config['NUM_CHARGERS']  # 充电桩数量，弃用
#NO_CHARGER_ZONES = config['NO_CHARGER_ZONES']
CHARGER_ZONES = config['CHARGER_ZONES']  #弃置，原来按区域分的有充电桩区和无充电桩区
#CHARGER_PER_ZONE = config.get('CHARGER_PER_ZONE', [0]*ZONE_NUM)
ZONES = config['ZONES']


# Convert CHARGER_ZONES to list if it's not already
if isinstance(CHARGER_ZONES, int):
    CHARGER_ZONES = [CHARGER_ZONES]

#251007加入
if "ZONES" in config:
    NUM_CHARGERS = sum(len(zone.get("charging_slots", [])) for zone in config["ZONES"])   # 充电桩数量
    CHARGER_PER_ZONE = [zone.get("charging_slots", []) for zone in config["ZONES"]]  #充电桩二维列表
    CHARGER_ZONES = [zone["zone_id"] for zone in config["ZONES"] if zone.get("charging_slots")]   #弃置，原来定义的充电区域与非充电区域
    #NO_CHARGER_ZONES = [zone["zone_id"] for zone in config["ZONES"] if not zone.get("charging_slots")] #弃置，无用变量
else:
    # 向后兼容旧配置
    NUM_CHARGERS = config.get('NUM_CHARGERS', 0)
    CHARGER_PER_ZONE = config.get('CHARGER_PER_ZONE', [0]*ZONE_NUM)
    CHARGER_ZONES = config.get('CHARGER_ZONES', [])
#    NO_CHARGER_ZONES = config.get('NO_CHARGER_ZONES', [])

# 定义动作、状态

ACTION_SPACE = spaces.Box(low=0, high=1, shape=(3 * ZONE_NUM,))



# 当前动作为：发车倾向、目的地倾向、充电倾向

# =========2026.2.6修改，状态空间增强：添加区域平均SOC==========
# =========2026.2.13修改，状态空间增强：添加时间步信息==========
OBS_SPACE = spaces.MultiDiscrete(
    (TIME_SLICE * ZONE_NUM) * [OD_MAX, OD_MAX, CHARGER_STATES, NUM_CHARGERS] +
    ZONE_NUM * [10] +  # 4个区域的SOC (0-9)
    [TIME_NUM, TIME_NUM]  # 新增：当前时间步(0-11)、剩余时间步(0-11)
)
# ======修改结束


# 当前状态为：od_in与od_out，观察未来TIME_SLICE个时间片，多维离散状态空间

class dclily(gym.Env):
    def __init__(self):
        super(dclily, self).__init__()

        # 定义动作空间和观测空间
        self.config = config  #1011增强rl逻辑时 新增

        self.training = 1  # 1是否处于训练模式的标志
        self.action_space = ACTION_SPACE  # 1环境的动作空间
        self.observation_space = OBS_SPACE  # 1一个观测空间的对象

        # =========2026.2.7修改，方案D2：添加epoch追踪==========
        self.current_epoch = 0  # 当前训练轮数（用于探索策略衰减）
        # ======修改结束

        # =========2026.2.14修改，实验方案开关==========
        # 按用户要求固定为第一个实验（A1），不通过命令行切换
        self.exp_variant = 'a1'
        # ======修改结束

        # 自行定义

        self._last_episode_timestep = 0  # 1上一回合的时间步数
        self.episode_timestep = 0  # 1当前回合的时间步数
        self.n_episode = 0  # 1回合计数器

        self.dist = np.array(pd.read_csv(path + 'od_demand_dist.csv', header=None))  # 1''里面是csv文件的名字 2 np.array是转换数组
        self.od_demand_left = np.array([])
        self.od_demand_now = np.array([])
        self.available = np.array([])
        self.mem_od_demand = []  # 保存数据
        self.mem_available = np.array([])
        self.car = np.array([])  # （1存储车辆信息）第一位数：当前目的地，第二位数：目前电量可行驶公里数；第三位数：距离目的地距离remain， >0 表示在路上。电量会提前扣除  2025.12.8 第四位数规划停车位编号（remain==0 时才是物理占用） charger_id，（不能充电就是 -1）
        self.car_mem = np.array([[-1, 0, 5] for _ in range(
            len(self.car))])  # vehicle priority(你应该用不上) 第一位数：调度zone；第二位数：调度时间片 第三位数：是否被调度；是为1，否为2(改为其他计数方法)
        self.new = 1  # 如果不是新的环境，不需要重复载入数据
        self.cong = np.array(pd.read_csv(path + 'cong.csv'))
        # self.pla_num=PL_NUM
        # self.dd=np.ones((ZONE_NUM,ZONE_NUM))*0
        # self.sorted_dist=[]
        self.od_in = np.array([])
        self.od_out = np.array([])
        self.mem_od_in = []
        self.mem_od_out = []
        self.vehicle_logs = []  # 每步记录车辆状态
        self.ZONE_NUM = ZONE_NUM
        self.PL_NUM = PL_NUM
        self.CAR_NUM = CAR_NUM
        self.TIME_NUM = TIME_NUM
        self.TIME_SLICE = TIME_SLICE
        
        #self.charging_usage = [0] * ZONE_NUM  #只对各区域使用的充电桩计数
        self.charging_stations = {
            zone_id: [
                {"charger_id": charger_id, "type": "fast", "occupied_by": None, "power": 60}
                for charger_id in zone_cfg["charging_slots"]
            ]
            for zone_id, zone_cfg in enumerate(ZONES)
        }      #充电桩对象列表，目前有充电桩编号、充电桩类型、被占了没有，电量
        
        # Log extraction initialization
        self.log_data = []
        self.current_step_logs = {}
        self.log_file_path = "vehicle_action_log.csv"
        # Initialize CSV with header
        with open(self.log_file_path, "w") as f:
            f.write("Time,VehicleID,Zone,SOC,Action_ChargePref,Formula_Score,Formula_Prob,Decision_Charge,Final_Behavior\n")

        #self.vehicle_logs.append(log)
        
        '''
        self.slot_status = {
            0: [False, True, False, True],  # zone 0 的第0~3个车位状态（True 表示已占用）
            1: [False, False, False, False],
            ...
        }#0727充电桩
        '''

        # self.summ=0
        # self.odin=[0]*ZONE_NUM
        # self.odout=[0]*ZONE_NUM
        # self.training_set_number = training_set_number #训练集

        # 写入初始OD信息，目前这个代码写的比较烂，要求demand.csv的格式比较离谱
        # 我可以大概给你解释你不懂的哪一行，但是我建议你重写
    
    def print_vehicle_status(self, t):
        print(f"\n=== 车辆状态（时间片 {t}）===")
        for vid in range(self.CAR_NUM):
            zone = self.car[vid][0]
            soc = self.car[vid][1]
            remain = self.car[vid][2]
            
            # 判断车辆状态
            if remain > 0:
                state = f"执行任务 (剩余 {remain} )"
            else:
                state = "空闲"
            
            print(f"车辆 {vid:2d} | 区域={zone} | SOC={soc:.2f} | 状态={state}")
        print("====================================\n")
    
    def log_vehicle_state(t, vehicles, filename="vehicle_log.csv"):
        with open(filename, "a") as f:
            for v in vehicles:
                f.write(f"{t},{v.id},{v.origin},{v.destination},{v.soc:.2f},"
                        f"{v.current_zone},{v.target_zone},{v.distance_to_target:.2f}\n")
    
    def write_od_left(self, new):
        if self.training:  # 根据 self.training 的标志判断当前模式是训练模式还是测试模式，然后选择不同的文件来加载 OD（Origin-Destination）需求数据，并计算每个区域的 OD 流量（流入和流出量）
            if new:
                for tr in range(training_set_number):
                    OD_demand = pd.read_csv(path + 'od_demand_' + str(tr + 1) + '.csv')
                    df_od = np.array([[[0] * ZONE_NUM] * ZONE_NUM] * (OD_demand.shape[1] - 2))
                    if np.any(df_od > OD_MAX):
                        print("Warning: OD demand exceeded the maximum value.")

                    for i in range(OD_demand.shape[0]):
                        for j in range(2, OD_demand.shape[1]):
                            df_od[j - 2][int(OD_demand.iloc[i]['O'])][int(OD_demand.iloc[i]['D'])] = int(
                                OD_demand.iloc[i][j])
                    df_od = np.array(df_od)
                    self.mem_od_demand.append(copy.deepcopy(df_od))
                    od_in = []
                    od_out = []
                    for ins in range(ZONE_NUM):
                        od_in.append(np.array([sum(row[:, ins]) for row in df_od]))  # 存这个
                        od_out.append(np.array([sum(row[ins, :]) for row in df_od]))  # 存这个
                    self.mem_od_in.append(od_in)
                    self.mem_od_out.append(od_out)

            a = random.randint(0, training_set_number - 1)
            OD_demand = self.mem_od_demand[a]
            df_od = np.array(OD_demand)
            od_in = self.mem_od_in[a]
            od_out = self.mem_od_out[a]
        else:
            # 目前默认只有一个测试场景
            OD_demand = pd.read_csv(path + 'od_demand_test.csv')
            df_od = np.array([[[0] * ZONE_NUM] * ZONE_NUM] * (OD_demand.shape[1] - 2))
            for i in range(OD_demand.shape[0]):
                for j in range(2, OD_demand.shape[1]):
                    df_od[j - 2][OD_demand.iloc[i]['O']][OD_demand.iloc[i]['D']] = OD_demand.iloc[i][j]
            df_od = np.array(df_od)
            od_in = []
            od_out = []
            for ins in range(ZONE_NUM):
                od_in.append(np.array([sum(row[:, ins]) for row in df_od]))  # 存这个
                od_out.append(np.array([sum(row[ins, :]) for row in df_od]))  # 存这个

        self.od_demand_left = np.array(df_od)
        self.od_in = np.array(od_in)
        self.od_out = np.array(od_out)

    def update_od_left(self):
        # 更新当前OD
        self.od_demand_left = self.od_demand_left[1:, :, :]

    def write_od_now(self):
        # 获取当前OD
        self.od_demand_now = copy.deepcopy(self.od_demand_left[:TIME_SLICE, :, :])
    
    def write_ini_cap(self, new):
        """
        初始化停车容量、车辆 self.car、充电桩 charging_stations，
        并自动分配初始车辆到区域与充电桩。
        """
        
        # ======================
        # 1) 初始化 available
        # ======================
        if new:
            available = pd.read_csv(path + 'capacity.csv')
            
            # available.iloc[0] = 每个区域初始车辆数量
            # available.iloc[1] = 每个区域可用停车位数量（总车位 - 初始车辆）
            for i in range(ZONE_NUM):
                available.iloc[1][str(i)] -= available.iloc[0][str(i)]
            
            self.available = np.array(available)
            self.mem_available = copy.deepcopy(self.available)
        
        else:
            # 不是新 episode，就恢复存档
            self.available = copy.deepcopy(self.mem_available)
        
        # 初始区域车辆数量（用于构造 self.car）
        zone_init_car = self.available[0]  # 第一行是初始车辆数量
        total_cars = int(sum(zone_init_car))
        
        # ======================
        # 2) 初始化车辆 self.car
        # car[i] = [zone, SOC, remain_dist, charger_id]
        # ======================
        self.car = []
        BAT = BATTERY_CAPACITY
        
        for zone_id in range(ZONE_NUM):
            count = int(zone_init_car[zone_id])
            for _ in range(count):
                self.car.append([zone_id, BAT, 0, -1])  # 初始无 charger_id
        
        # 转成 numpy 数组（便于你后续使用）
        self.car = np.array(self.car, dtype=int)
        
        # ======================
        # 3) 初始化 charging_stations（构建充电桩结构）
        # ======================
        self.charging_stations = {
            zone_id: [
                {
                    "charger_id": cid,
                    "occupied_by": None,
                    "type": "fast",
                    "power": 60
                }
                for cid in ZONES[zone_id]["charging_slots"]
            ]
            for zone_id in range(ZONE_NUM)
        }
        
        # ======================
        # 4) 自动让区域里部分车辆占用充电桩
        #
        # 按照区域车辆数量 → 尽可能使用 charger
        #
        # 保证 car ↔ station 双向一致
        # ======================

        # =========2026.2.13修改，修复初始化充电桩占用问题==========
        # 问题：初始化时车辆占用充电桩，但已经满电，导致充电桩被长期占用
        # 解决：所有车辆初始时都不占用充电桩，让RL自主决定何时充电
        # car_index = 0
        #
        # for zone_id in range(ZONE_NUM):
        #     zone_car_cnt = int(zone_init_car[zone_id])
        #     chargers = self.charging_stations[zone_id]
        #     num_chargers = len(chargers)
        #
        #     # 能使用的充电桩数量
        #     use = min(zone_car_cnt, num_chargers)
        #
        #     # ① 前 use 个车占用充电桩
        #     for j in range(use):
        #         charger = chargers[j]
        #         charger["occupied_by"] = car_index
        #         self.car[car_index][3] = charger["charger_id"]
        #         car_index += 1
        #
        #     # ② 剩余车辆只停在停车位，不占桩
        #     for j in range(zone_car_cnt - use):
        #         self.car[car_index][3] = -1
        #         car_index += 1
        # ======修改结束
        
        # ======================
        # 完成初始化
        # ======================
       # print(">>> 初始化完成: self.car 与 charging_stations 已同步。")
    
    # SOC充电函数

    def f(self, t):  # 充电函数
        a = -0.00011169
        b = 0.02129281
        return a * t * t + b * t

    def g(self, y):
        a = -0.00011169
        b = 0.02129281

        # Calculate the square root term
        sqrt_term = math.sqrt(b ** 2 + 4 * a * y)

        # Calculate the positive branch of the inverse function
        t = (-b + sqrt_term) / (2 * a)

        return t

    def charging(self, battery, minute):
        """
           计算车辆在特定电量 battery 和已充电时间 minute 下的可充入电量（单位：电量值）

           参数
           ----
           battery : float
               当前电池剩余电量（单位与 BATTERY_CAPACITY 一致，如 Wh 或 kWh）

           minute : float
               当前已充电的时间（单位：分钟）

           返回
           ----
           delta_battery : float
               本次充电可获得的电量（单位与 battery 一致）

           函数逻辑
           -------
           - 使用 self.g() 对当前电量归一化后的值进行非线性变换，得到延迟因子
           - 计算实际的“有效充电时间” real_min
           - 若充电时间超出上限 107 分钟（可能是系统约定的最大时间窗），则认为直接充满
           - 否则根据 self.f() 的非线性函数估算充电效率曲线，得到实际可充电量
           """
        
        max_capacity = BATTERY_CAPACITY  # 电池最大容量
        real_min = minute + self.g(battery / max_capacity)
        if real_min > 107:
            # 若实际充电分钟数超过某个阈值（107分钟），直接返回充满所需电量 +1
            # 加1 可能是避免浮点误差使得值小于 max_capacity
            return max_capacity - battery + 1
        else:
            # 使用 f() 函数获取 real_min 对应的“累计充电比例”，再减去当前比例
            # 得到本次可以充入的电量
            return (self.f(real_min) - battery / max_capacity) * max_capacity
    
    def get_micro_state(self):
        # 1) OD in/out，裁到 0..OD_MAX-1，并转 int
        od_in2 = np.array(self.od_in[:, self.episode_timestep:self.episode_timestep + TIME_SLICE]).T
        od_out2 = np.array(self.od_out[:, self.episode_timestep:self.episode_timestep + TIME_SLICE]).T
        od_part = np.concatenate((od_in2, od_out2), axis=1).ravel()
        od_part = np.clip(od_part, 0, OD_MAX - 1).astype(np.int64)
        
        # 2) 充电状态 (第3段)：把占用率映射成 0..CHARGER_STATES-1
        charger_states = []
        for zone in range(ZONE_NUM):
            # 占用率 -> 状态档位
            ratio = self.available[0][zone] / (self.mem_available[0][zone] + 1e-5)
            val = int(ratio * CHARGER_STATES)
            val = max(0, min(CHARGER_STATES - 1, val))
            charger_states.extend([val] * TIME_SLICE)
        charger_states = np.array(charger_states, dtype=np.int64)
        
        # 3) 充电桩数量 (第4段)：每区装配的桩数，clip 到 0..NUM_CHARGERS-1
        num_chargers_feat = []
        for zone in range(ZONE_NUM):
            installed = 0
            if zone < len(CHARGER_PER_ZONE) and isinstance(CHARGER_PER_ZONE[zone], (list, tuple)):
                installed = len(CHARGER_PER_ZONE[zone])
            # 若想放“可用充电位数”，可在此用占用信息计算；这里先放装配数量
            if NUM_CHARGERS > 0:
                installed = max(0, min(NUM_CHARGERS - 1, installed))
            else:
                installed = 0
            num_chargers_feat.extend([installed] * TIME_SLICE)
        num_chargers_feat = np.array(num_chargers_feat, dtype=np.int64)

        # =========2026.2.6修改，状态空间增强：添加区域平均SOC==========
        # 4) 区域平均 SOC (新增)
        avg_soc_per_zone = []
        for zone in range(ZONE_NUM):
            # 计算该区域所有静止车辆的平均 SOC
            zone_vehicles = [self.car[i][1] for i in range(len(self.car))
                             if self.car[i][0] == zone and self.car[i][2] == 0]
            if zone_vehicles:
                avg_soc = np.mean(zone_vehicles) / BATTERY_CAPACITY  # 归一化到 0-1
            else:
                avg_soc = 0.5  # 默认值（区域无静止车辆时）

            # 离散化为 10 档 (0-9)
            soc_discrete = int(np.clip(avg_soc * 10, 0, 9))
            avg_soc_per_zone.append(soc_discrete)

        avg_soc_per_zone = np.array(avg_soc_per_zone, dtype=np.int64)
        # ======修改结束

        # =========2026.2.13修改，状态空间增强：添加时间步信息==========
        # 5) 时间步特征（新增）
        # 修复：确保时间步值在有效范围[0, TIME_NUM-1]内
        current_timestep = np.array([min(self.episode_timestep, TIME_NUM - 1)], dtype=np.int64)  # 当前时间步 (0-11)
        remaining_timesteps = np.array([max(0, TIME_NUM - self.episode_timestep - 1)], dtype=np.int64)  # 剩余时间步 (11-0)
        # ======修改结束

        # 6) 拼接成 [OD, OD, CHARGER_STATES, NUM_CHARGERS, AVG_SOC, TIMESTEP, REMAINING] * (TIME_SLICE*ZONE_NUM + ZONE_NUM + 2)
        state = np.concatenate([od_part, charger_states, num_chargers_feat, avg_soc_per_zone, current_timestep, remaining_timesteps])
        # 保险断言：长度必须满足 4 * ZONE_NUM * TIME_SLICE + ZONE_NUM + 2
        assert state.shape[0] == 4 * ZONE_NUM * TIME_SLICE + ZONE_NUM + 2, \
            f"state dim {state.shape[0]} != 4*ZONE_NUM*TIME_SLICE + ZONE_NUM + 2 (expected {4 * ZONE_NUM * TIME_SLICE + ZONE_NUM + 2})"
        return state
        
        # return np.concatenate([od_part, np.array(charger_states)])

    @property
    def state(self):
        return self.get_micro_state()

    # =========2026.2.12修改，方案F5：统一可调度车辆口径==========
    def _compute_dispatchable_counts(self):
        """按车辆真实状态重算各区域可调度车辆数（remain==0 且未占充电桩）。"""
        counts = np.zeros(self.ZONE_NUM, dtype=int)
        for i in range(len(self.car)):
            if self.car[i][2] == 0 and self.car[i][3] == -1:
                zone = int(self.car[i][0])
                if 0 <= zone < self.ZONE_NUM:
                    counts[zone] += 1
        return counts

    def _sync_dispatchable_available(self):
        """同步 available[0]，防止充电侧重复扣减导致调度预算漂移。"""
        self.available[0] = self._compute_dispatchable_counts().astype(self.available[0].dtype, copy=False)
    # ======修改结束

    def _sigmoid(self, x):
        """标准 Sigmoid 激活"""
        return 1.0 / (1.0 + math.exp(-x))

    # =========2026.1.30修改，最优解求解==========
    def _cancel_bidirectional_dispatch(self, matrix):
        """对调度矩阵做双向净流抵消，避免同时出现 i->j 与 j->i。"""
        net_matrix = np.array(matrix, copy=True, dtype=float)
        for i in range(self.ZONE_NUM):
            net_matrix[i][i] = 0
        for i in range(self.ZONE_NUM):
            for j in range(i + 1, self.ZONE_NUM):
                net = net_matrix[i][j] - net_matrix[j][i]
                if net >= 0:
                    net_matrix[i][j] = net
                    net_matrix[j][i] = 0
                else:
                    net_matrix[i][j] = 0
                    net_matrix[j][i] = -net
        return net_matrix
    # ======修改结束

    # =========2026.2.14修改，实验方案B2前瞻需求==========
    def _get_future_out_demand(self, step_offset=1):
        """返回未来 step_offset 时间片的各区域出发需求向量。"""
        idx = int(self.episode_timestep + step_offset)
        if self.od_out.size == 0 or idx < 0 or idx >= self.od_out.shape[1]:
            return np.zeros(self.ZONE_NUM, dtype=float)
        return np.asarray(self.od_out[:, idx], dtype=float)
    # ======修改结束

    def rl_should_charge(self, vehicle_id, charge_pref):
        """
        [已废弃 - 2026.2.6方案C] 使用 RL 输出来决定是否充电（替换固定阈值法）
        方案C改为基于数量的充电决策，不再使用此函数
        保留此函数以便回滚或对比实验

        charge_pref: 动作向量中充电倾向部分
        """
        # =========2026.2.4修改，方案一：简化充电决策逻辑==========
        # 问题: 复杂权重打分系统导致3次修复失败，架构性缺陷
        # 方案: 移除所有权重计算，直接使用RL的pref值作为充电概率
        # 目标: 让RL完全控制充电决策，无中间层干扰

        cfg = getattr(self, "config", None)
        zone = self.car[vehicle_id][0]
        soc = max(0.0, min(1.0, self.car[vehicle_id][1] / BATTERY_CAPACITY))
        pref = float(charge_pref[zone])

        # 直接使用RL输出的pref作为充电概率
        # 仅做基本裁剪，防止极端值
        lo, hi = (cfg.get("RL_CHARGE_DECISION_CLIP", [0.05, 0.95]) if cfg else [0.05, 0.95])
        p = min(max(pref, lo), hi)

        # 训练期：随机采样探索；推理期：按概率阈值决策
        r = random.random()
        if self.training:
            decision = (r < p)
        else:
            decision = (p >= 0.5)

        # Log the details for this vehicle
        if vehicle_id not in self.current_step_logs:
            self.current_step_logs[vehicle_id] = {}

        self.current_step_logs[vehicle_id].update({
            "soc": soc,
            "pref": pref,
            "prob": p,
            "decision_charge": decision
        })
        # ======修改结束

        return decision
    
    def handle_vehicle_charging(self, vehicle_id, target_zone):
        """
        未出发的车辆 → 由 check_idle_vehicles() 决定是否去充电（主动决策）
        刚到达目的地的车辆 → 由 handle_vehicle_charging() 决定下一步是否去充电（被动决策）
        
        """
        BATTERY_THRESHOLD = (BATTERY_CAPACITY / DIST_MAX) * BATTERY_CAPACITY
        current_battery = self.car[vehicle_id][1]
        current_zone = self.car[vehicle_id][0] if self.car[vehicle_id][2] == 0 else -1
        
        zone_capacity = self.mem_available[0][target_zone]
        current_vehicles = self.available[0][target_zone]
        has_charger = len(CHARGER_PER_ZONE[target_zone]) > 0
        has_space = current_vehicles <= zone_capacity
        #1
        
        if has_space and not has_charger:
            required_energy = self.dist[current_zone][target_zone]
            if current_battery > required_energy:
                self.car[vehicle_id][0] = target_zone
                self.car[vehicle_id][2] = self.dist[current_zone][target_zone]
                # === 【新增】 这是一个去充电的动作，必须规划充电位 ===
                self.assign_parking_slot(vehicle_id, is_charging_order=True)
                # print(
                #     f"[Charging] 车辆{vehicle_id} 到达区{current_zone} → 前往充电区 {target_zone}, 距离={self.car[vehicle_id][2]}")
                
            
            else:
                nearest_zone, _ = self.find_nearest_available_charger(current_zone)
                return nearest_zone
        
        #2
        elif has_space and has_charger:
            self.car[vehicle_id][0] = target_zone
            # === 【新增】 既然决定在本地充电，立刻规划/占用充电位
            self.assign_parking_slot(vehicle_id, is_charging_order=True)
        
        else:
            for alt_zone in range(ZONE_NUM):
                if (self.available[0][alt_zone] < self.mem_available[0][alt_zone] and
                        alt_zone != target_zone):
                    return alt_zone
        
        return target_zone
    
    def find_nearest_charger(self, current_zone):
        """
        在全网络中寻找距离 current_zone 最近的【空闲充电桩】。

        返回：
        - (zone_id, charger_id)：找到可用充电桩
        - (-1, -1)：没有任何可用充电桩
    
        注意：
        - 这里只是“查询”
        - 不修改 self.car
        - 不修改 occupied_by
    """
        min_dist = float('inf')
        best_zone = -1
        best_charger_id = -1
        
        # 遍历所有区域
        for zone_id, chargers in self.charging_stations.items():
            
            # 跳过没有充电桩的区域
            if not chargers:
                continue
            
            # 遍历该区域的每一个充电桩
            for ch in chargers:
                # 只考虑空闲的充电桩
                if ch["occupied_by"] is not None:
                    continue
                
                dist = self.dist[current_zone][zone_id]
                if dist < min_dist:
                    min_dist = dist
                    best_zone = zone_id
                    best_charger_id = ch["charger_id"]
        
        return best_zone, best_charger_id
    
    def find_preferred_charger(self, current_zone, charge_pref):
        candidates = []
        for zone in range(ZONE_NUM):
            if len(CHARGER_PER_ZONE[zone]) == 0:
                continue
            has_available_slot = any(self.available[1][zone] > slot for slot in CHARGER_PER_ZONE[zone])
            if has_available_slot:
                dist_score = -self.dist[current_zone][zone]
                prefer_score = charge_pref[zone]
                
                score = 0.7 * prefer_score + 0.3 * dist_score
                candidates.append((zone, score))
        if not candidates:
            return -1
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def check_idle_vehicles(self, charge_pref, charging_actions=None, epoch=0):
        """检查空闲车辆状态（由 RL 决定充电数量；再决定去哪儿充）

        参数:
            charge_pref: RL输出的充电比例 [zone0, zone1, zone2, zone3]
            charging_actions: 充电动作记录字典
            epoch: 当前训练轮数（用于方案D2的探索衰减）
        """
        # =========2026.2.6修改，方案C：从充电倾向改为充电数量==========
        # 原逻辑：逐车判断，基于概率决策（charge_pref作为概率）
        # 新逻辑：按区域分配充电数量，选择最需要充电的车辆
        # 优势：更直接的控制，避免概率随机性导致的不稳定

        if charging_actions is None:
            charging_actions = {}

        # =========2026.2.7修改，方案D2：ε-greedy充电探索==========
        # 前30轮，10%概率随机选择一个区域进行充电
        if epoch < 30 and random.random() < 0.1:
            random_zone = random.randint(0, ZONE_NUM-1)
            idle_vehicles_in_zone = []

            for i in range(len(self.car)):
                if self.car[i][2] == 0 and self.car[i][3] == -1:  # 静止且可调度车辆
                    if self.car[i][0] == random_zone:  # 在目标区域
                        idle_vehicles_in_zone.append((i, self.car[i][1]))  # (车辆ID, SOC)

            if idle_vehicles_in_zone:
                # 随机充电1-3辆车
                charge_count = random.randint(1, min(3, len(idle_vehicles_in_zone)))
                # 按SOC升序排序，优先充电低电量车辆
                idle_vehicles_in_zone.sort(key=lambda x: x[1])

                for vehicle_id, soc in idle_vehicles_in_zone[:charge_count]:
                    current_zone = self.car[vehicle_id][0]

                    # 找到最优充电站（复用原有逻辑）
                    target_charger = self.find_preferred_charger(current_zone, charge_pref)

                    if target_charger != -1:
                        # 释放旧充电桩（如果有）
                        old_charger_id = self.car[vehicle_id][3]
                        if old_charger_id != -1:
                            self.release_charger(current_zone, old_charger_id)
                            self.car[vehicle_id][3] = -1

                        # 设置新的目标区域
                        self.car[vehicle_id][0] = target_charger

                        # 设置行驶距离
                        dist_to_charge = self.dist[current_zone][target_charger]
                        self.car[vehicle_id][2] = dist_to_charge

                        # 规划停车位
                        self.assign_parking_slot(vehicle_id, is_charging_order=True)

                        # 记录充电动作
                        if self.car[vehicle_id][3] != -1:
                            if target_charger not in charging_actions:
                                charging_actions[target_charger] = 0
                            charging_actions[target_charger] += 1

                        # 立即物理锁桩
                        planned_id = self.car[vehicle_id][3]
                        if planned_id != -1:
                            if target_charger in self.charging_stations:
                                for station in self.charging_stations[target_charger]:
                                    if station['charger_id'] == planned_id:
                                        station['occupied_by'] = vehicle_id
                                        break

                        # 如果是本地充电，从调度池移除
                        if dist_to_charge == 0:
                            self.available[0][current_zone] -= 1

                # 确保 available 不会变成负数
                for z in range(self.ZONE_NUM):
                    if self.available[0][z] < 0:
                        self.available[0][z] = 0

                return  # 本步使用随机策略，跳过RL决策
        # ======探索逻辑结束

        # =========2026.2.12修改，方案F1：全局充电预算 + 跨区回填==========
        # 目标：减少后期“有动作但执行为0”的塌缩，改为全局分配可充电车辆
        pref_arr = np.clip(np.asarray(charge_pref, dtype=float), 0.0, 1.0)

        idle_vehicles_all = []
        for i in range(len(self.car)):
            if self.car[i][2] == 0 and self.car[i][3] == -1:
                zone = int(self.car[i][0])
                soc = float(self.car[i][1])
                idle_vehicles_all.append((i, zone, soc))
                # 为日志提供稳定观测，避免 N/A
                if i not in self.current_step_logs:
                    self.current_step_logs[i] = {}
                self.current_step_logs[i].update({
                    "soc": max(0.0, min(1.0, soc / BATTERY_CAPACITY)),
                    "pref": float(pref_arr[zone]),
                    "prob": float(pref_arr[zone]),
                    "score": float(pref_arr[zone]),
                    "decision_charge": False
                })

        total_idle = len(idle_vehicles_all)
        # =========2026.2.13修改，添加充电诊断输出==========
        # if self.episode_timestep <= 2:  # 只在前3步输出，避免刷屏
        #     print(f"\n[Charging Debug t={self.episode_timestep}]")
        #     print(f"  Idle vehicles: {total_idle}")
        #     print(f"  Charge prefs: {pref_arr}")
        #     print(f"  Max pref: {pref_arr.max():.3f}")
        # ======修改结束

        if total_idle > 0 and float(pref_arr.max()) > 1e-6:
            global_ratio = float(np.clip(0.6 * pref_arr.max() + 0.4 * pref_arr.mean(), 0.0, 1.0))
            charge_budget = int(round(global_ratio * total_idle))
            if pref_arr.max() > 0.8:
                charge_budget = max(charge_budget, int(round(0.6 * total_idle)))
            charge_budget = max(0, min(charge_budget, total_idle))

            # =========2026.2.13修改，添加强制充电约束（硬约束）==========
            # 目的：确保车队SOC不会耗尽，保证策略稳定性
            # 计算车队平均SOC
            total_soc = sum(self.car[i][1] for i in range(self.CAR_NUM))
            avg_soc = total_soc / self.CAR_NUM

            # 强制充电约束：当平均SOC过低时，强制最低充电比例
            if avg_soc < 1000:
                # 平均SOC < 1000 (50%)，强制至少30%的idle车辆充电
                min_charge_vehicles = max(3, int(total_idle * 0.3))
                charge_budget = max(charge_budget, min_charge_vehicles)
            elif avg_soc < 1200:
                # 平均SOC < 1200 (60%)，强制至少20%的idle车辆充电
                min_charge_vehicles = max(2, int(total_idle * 0.2))
                charge_budget = max(charge_budget, min_charge_vehicles)

            # 确保不超过总idle车辆数
            charge_budget = min(charge_budget, total_idle)
            # ======修改结束

            # =========2026.2.13修改，添加充电预算诊断==========
            # if self.episode_timestep <= 2:
            #     print(f"  Global ratio: {global_ratio:.3f}")
            #     print(f"  Charge budget: {charge_budget}")
            # ======修改结束

            # 优先充电低SOC车辆
            idle_vehicles_all.sort(key=lambda x: x[2])

            charged_count = 0
            for vehicle_id, current_zone, _soc in idle_vehicles_all:
                if charged_count >= charge_budget:
                    break

                target_charger = self.find_preferred_charger(current_zone, pref_arr)
                if target_charger == -1:
                    # 回填策略：首选最近可用充电桩，避免偏好区无车/无桩时直接归零
                    target_charger, _nearest_charger_id = self.find_nearest_charger(current_zone)
                if target_charger == -1:
                    continue

                old_charger_id = self.car[vehicle_id][3]
                if old_charger_id != -1:
                    self.release_charger(current_zone, old_charger_id)
                    self.car[vehicle_id][3] = -1

                self.car[vehicle_id][0] = target_charger
                dist_to_charge = self.dist[current_zone][target_charger]
                self.car[vehicle_id][2] = dist_to_charge

                # Debug: 检查充电桩可用性
                # if self.episode_timestep <= 2:
                #     chargers_in_zone = self.charging_stations.get(target_charger, [])
                #     occupied_count = sum(1 for ch in chargers_in_zone if ch["occupied_by"] is not None)
                #     print(f"    Vehicle {vehicle_id} -> Zone {target_charger}: {len(chargers_in_zone)} chargers, {occupied_count} occupied")

                self.assign_parking_slot(vehicle_id, is_charging_order=True)

                # Debug: 检查分配结果
                # if self.episode_timestep <= 2:
                #     assigned_charger_id = self.car[vehicle_id][3]
                #     print(f"      -> Assigned charger_id: {assigned_charger_id}")

                if self.car[vehicle_id][3] != -1:
                    charging_actions[target_charger] = charging_actions.get(target_charger, 0) + 1

                planned_id = self.car[vehicle_id][3]
                if planned_id != -1 and target_charger in self.charging_stations:
                    for station in self.charging_stations[target_charger]:
                        if station['charger_id'] == planned_id:
                            station['occupied_by'] = vehicle_id
                            break

                if dist_to_charge == 0:
                    self.available[0][current_zone] -= 1

                # 记录已执行充电决策
                self.current_step_logs[vehicle_id].update({
                    "decision_charge": True,
                    "prob": global_ratio,
                    "score": float(charge_budget)
                })
                charged_count += 1
        # ======修改结束

        # =========2026.2.13修改，输出充电执行结果==========
        # if self.episode_timestep <= 2:
        #     print(f"  Charged count: {charged_count if 'charged_count' in locals() else 0}")
        #     print(f"  Charging actions: {charging_actions}")
        # ======修改结束

        # 确保 available 不会变成负数
        for z in range(self.ZONE_NUM):
            if self.available[0][z] < 0:
                self.available[0][z] = 0
        # ======修改结束
    
    def release_charger(self, zone_id, charger_id):
        """释放指定区域的指定充电桩,清除车辆离开时的占用记录。"""
        if zone_id in self.charging_stations:
            for station in self.charging_stations[zone_id]:
                if station['charger_id'] == charger_id:
                    # 只有当确实被占用时才打印，避免刷屏
                    if station['occupied_by'] is not None:
                        # print(f"[System] 释放 区域{zone_id} 充电桩{charger_id} (原占用: 车辆{station['occupied_by']})")
                        station['occupied_by'] = None
                    break
    def print_charging_stations(self):
        """
        打印当前所有充电桩的状态，用于调试与状态校验。

        作用：
        - 验证 charging_stations 是否正确初始化
        - 检查每个 zone 下有哪些充电桩
        - 检查每个充电桩当前是否被车辆占用（occupied_by）
        - 方便调试 assign_charger / release_charger 逻辑
        """
        
        # 打印整体标题，方便在控制台中区分不同时间点的输出
        #print("\n===== Charging Stations Status =====")
        
        # 遍历每一个区域（zone）
        # zone_id: 区域编号
        # chargers: 该区域内的充电桩列表
        # for zone_id, chargers in self.charging_stations.items():
        #
        #     # 打印当前区域编号
        #     print(f"Zone {zone_id}:")
        #
        #     # 如果该区域没有任何充电桩，明确提示
        #     if len(chargers) == 0:
        #         print("  (No chargers)")
            
            # 遍历该区域内的每一个充电桩
           # for ch in chargers:
                # ch 是一个字典，结构类似：
                # {
                #   "charger_id": int,
                #   "type": "fast" / "slow",
                #   "occupied_by": None 或 car_id,
                #   "power": 60
                # }
                
                # 打印单个充电桩的详细状态
                # print(
                #     f"  Charger {ch['charger_id']} | "  # 充电桩编号（与 ZONES 中保持一致）
                #     f"type={ch['type']} | "  # 充电桩类型（目前固定 fast）
                #     f"occupied_by={ch['occupied_by']} | "  # 当前占用该桩的车辆 ID（None 表示空闲）
                #     f"power={ch['power']}kW"  # 充电功率（单位 kW）
                # )
        
        # 打印结尾分隔符，提升可读性
        # print("===================================\n")
    
    def assign_parking_slot(self, car_id, is_charging_order):
        """
        为车辆规划目标区域内的停车位类型（仅规划，不发生物理占用）

        参数
        ----
        car_id : int
            车辆编号
        is_charging_order : bool
            True  -> 充电订单，优先分配充电桩
            False -> OD / 调度订单，优先普通停车位（charger_id = -1）

        规则
        ----
        - 仅在「车辆已经确定目标区域，但仍在路上（remain > 0）」时调用
        - 不修改 charging_stations[*]['occupied_by']
        - charger_id 只是“计划停车位编号”
        
        -------260107 二编
        修正后的规划逻辑：防止非充电车辆由于车位不足而产生“幽灵占位”
        """
        
        # ===============================
        # 1. 读取车辆目标区域
        # ===============================
        target_zone = int(self.car[car_id][0])
        
        # 获取该区域的充电桩列表
        chargers = self.charging_stations.get(target_zone, [])
        num_chargers = len(chargers)
        
        # 获取区域总容量和当前在区域内的车辆数（近似计算）
        # 注意：这里需要一个准确的方法知道有多少车已经在该区域且是普通停车
        # 简化处理：我们尝试分配。
        
        # ===============================
        # 逻辑分支 A：这是一个充电订单
        # ===============================
        if is_charging_order:
            if num_chargers == 0:
                self.car[car_id][3] = -1
                return
            
            # 优先找空闲桩
            for ch in chargers:
                if ch["occupied_by"] is None:
                    self.car[car_id][3] = ch["charger_id"]
                    return
            
            # 全满了，只能排队或停普通位
            self.car[car_id][3] = -1
            return
        
        # ===============================
        # 逻辑分支 B：这是一个普通订单 (OD/调度)
        # ===============================
        else:
            # 1. 先默认给普通位
            self.car[car_id][3] = -1
            
            # 2. 检查隐患：是否普通位已满，必须占用充电桩？
            # 计算该区域目前已经被占用的充电桩数量
            occupied_chargers_count = sum(1 for ch in chargers if ch["occupied_by"] is not None)
            
            # 获取该区域当前总车数 (available[0] 是区域当前车数)
            current_total_cars = self.available[0][target_zone]
            total_capacity = self.mem_available[0][target_zone]  # 假设 mem_available 存的是总容量
            
            # 估算剩余的普通车位 = (总容量 - 充电桩数) - (当前总车数 - 当前占桩车数)
            # 这个计算比较复杂，因为 current_total_cars 包含了即将到达的车吗？
            #
            # 💡 更稳健的策略：
            # 如果是普通订单，我们尽量不分配 charger_id。
            # 但是，如果系统发现该区域空闲位置极少，且空闲位置 < 空闲充电桩数
            # 这意味着剩余的空位全都是充电桩！
            
            free_spots_total = self.available[1][target_zone]  # 剩余总空位
            free_chargers = sum(1 for ch in chargers if ch["occupied_by"] is None)
            
            # 如果 剩余总空位 <= 剩余充电桩数,也就是剩余普通停车位 ≤ 0，说明普通车位已经没了！
            # 这辆车必须去占一个充电桩，否则它就变成了“幽灵车”
            if free_spots_total <= free_chargers and free_chargers > 0:
                # 必须占用一个充电桩（即燃油车占位）
                for ch in chargers:
                    if ch["occupied_by"] is None:
                        self.car[car_id][3] = ch["charger_id"]
                        # print(f"警告：区域{target_zone}普通车位已满，车辆{car_id}被迫占用充电桩{ch['charger_id']}")
                        return
            
            # 否则，还是维持 -1
            self.car[car_id][3] = -1
            return
    
    def step(self, action):

        charge_pref = action[2 * ZONE_NUM: 3 * ZONE_NUM]
        # print(f"\n--- 时间片 {self.episode_timestep} ---")#20251201
        # print("\n====== 动作前 self.car ======")
        # for i in range(self.CAR_NUM):
        #     print(f"车辆 {i}: zone={self.car[i][0]}, SOC={self.car[i][1]}, remain={self.car[i][2]}, chargerid={self.car[i][3]}")
        # print("================================\n")
        t = self.episode_timestep

        # Reset logs for current step
        self.current_step_logs = {}

        # =========2026.2.4修改，记录充电动作==========
        # 记录本时间步的充电动作（zone_id: count）
        charging_actions = {}
        # ======修改结束

        #self.print_vehicle_status(t)

        # 实现一步的逻辑
        e = 1  # 一个trick

        action = copy.deepcopy(action)
        rate = 0.6
        reward = [[0] * ZONE_NUM for _ in range(TIME_NUM)]# 初始化一个二维列表，存储奖励值。大小：TIME_NUM x ZONE_NUM
        real_reward = np.array([0] * ZONE_NUM)
        j = 0
        pre_avail = copy.deepcopy(self.available[1])
        # 先移动，再看动作，再满足需求

        # 移动

        con = self.cong[self.episode_timestep - 1][0]  # 道路拥堵参数

        i = 0
        while i < len(self.car):
            #self.car_mem[i][2] = max(5, self.car_mem[i][2] + 1)  # 优先级计算
            # === 新增代码开始：处理实际充电 ===
            # 判断条件：车辆不在路上 (remain <= 0) 且 车辆分配了充电桩 (charger_id != -1)
            if self.car[i][2] <= 0 and self.car[i][3] != -1:
                # 调用你的 charging 函数或者简单的线性充电公式
                # 注意：这里需要确保 charging 函数返回的是增量，或者你直接在这里计算
                # 假设简单的线性充电：
                charge_speed = 1800  # 假设 60kW 功率，或者根据 charging_stations[zone][id]['power'] 获取
                added_energy = charge_speed * (TIME_LENGTH / 60)  # 转换为 kWh
                
                # 更新电量
                self.car[i][1] = min(BATTERY_CAPACITY, self.car[i][1] + added_energy)

                # =========2026.2.13修改，优化充电释放策略==========
                # 原策略：充满才释放（需要7步）
                # 新策略：充到80%就释放（约5步），提高充电桩利用率
                release_threshold = BATTERY_CAPACITY * 0.8  # 1600 SOC
                if self.car[i][1] >= release_threshold:
                    self.car[i][1] = min(BATTERY_CAPACITY, self.car[i][1])  # 确保不超过上限
                    # 释放充电桩
                    current_zone = self.car[i][0]
                    charger_id = self.car[i][3]
                    self.release_charger(current_zone, charger_id)
                    self.car[i][3] = -1  # 释放充电桩状态
                # ======修改结束

            # =========2026.2.13修改，添加SOC消耗逻辑==========
            # 车辆移动时消耗电量
            if self.car[i][2] > 0:
                # 计算本时间步实际移动的距离
                max_distance = car_speed_min * TIME_LENGTH * con
                actual_distance = min(self.car[i][2], max_distance)

                # SOC消耗：1 SOC per unit distance（与最优解一致）
                soc_consumed = actual_distance * 1.0
                self.car[i][1] = max(0, self.car[i][1] - soc_consumed)
                self.car[i][2] -= actual_distance
                # ======修改结束

                # === 车辆到达时刻 (Arrival) ===
                if self.car[i][2] <= 0:
                    self.car[i][2] = 0
                    current_zone = self.car[i][0]
                    #self.available[0][self.car[i][0]] += 1
                    
                    # 处理充电逻辑，这一part有什么用？？？bug
                    target_zone1 = self.car[i][0]
                    target_zone = self.handle_vehicle_charging(i, target_zone1)
                    
                    # =================================================
                    # 【新增代码】物理占用同步：人车合一
                    # =================================================
                    planned_charger_id = self.car[i][3]
                    
                    # 如果这辆车身上有预定的充电桩 ID (无论是真充电，还是被迫占位)
                    if planned_charger_id != -1:
                        if current_zone in self.charging_stations:
                            for station in self.charging_stations[current_zone]:
                                if station['charger_id'] == planned_charger_id:
                                    # 核心：将环境里的桩标记为被当前车辆(i)占用
                                    station['occupied_by'] = i
                                    # print(f"车辆{i} 物理落锁 区域{current_zone} 充电桩{planned_charger_id}")
                                    break
                    # 🔍 打印前10辆车当前电量，用于验证是否存在低电量车
                    #print("🔋 前10辆车辆当前的电量可行驶距离：", [round(self.car[i][1], 1) for i in range(min(10, CAR_NUM))])

                    # =========2026.2.13修改，允许充电车辆参与服务==========
                    # 原逻辑：充电车辆完全不可用（planned_charger_id != -1时不计入available）
                    # 新逻辑：充电车辆仍可服务，减少充电的机会成本
                    # 这样让RL更容易学习充电策略
                    # 【关键修改】充电车辆也计入available
                    self.available[0][self.car[i][0]] += 1  # 所有到达车辆都可用
                    # ======修改结束

                    # # 原有逻辑：到达后判断下一步（注意：这通常是针对刚才完成移动的车）
                    # #处理充电逻辑，这一part有什么用？？？bug
                    # target_zone1 = self.car[i][0]
                    # target_zone = self.handle_vehicle_charging(i, target_zone1)
            
            i += 1

        # =========2026.2.12修改，方案F5：动作解码前同步可调度车辆数==========
        # 目的：调度预算应基于车辆真实状态，而非历史扣减残值
        self._sync_dispatchable_available()
        # ======修改结束

        # =========2026.2.6修改，方案4B：延后充电决策，避免充电-调度冲突==========
        # 注释掉原位置的充电决策（在调度前）
        # 原因：此时调用会导致空闲车辆优先被Gurobi调度走，充电决策时无车可充
        # self.check_idle_vehicles(charge_pref, charging_actions)
        # ======修改结束

        # 多区域OD分配，为每个区域的车辆分配目的地

        action_table_true = np.ones((ZONE_NUM, ZONE_NUM)) * 0  # 存储最终的 OD 分配结果
        i = 0
        j = 0
        real_act = []  # 调整后的 OD 动作分配表

        # 对输入的动作进行OD分配（动作前4位为车辆出发偏好，后四位为到达偏好）
        for i in range(0, ZONE_NUM):
            action_table = copy.deepcopy(action[ZONE_NUM:])
            action_table[i] = 0
            real_act.append(action_table)

            # 冲突动作的删除处理
            for j in range(i):
                if real_act[i][j] > real_act[j][i]:
                    real_act[j][i] = 0


        # =========2026.2.12修改，方案F5：使用统一口径的可调度预算==========
        dispatchable_available = self._compute_dispatchable_counts()
        # ======修改结束

        # =========2026.2.14修改，实验A/B动作解码逻辑==========
        variant = getattr(self, 'exp_variant', 'baseline')
        enable_a1 = variant in ('a1', 'a2', 'b1', 'b2', 'c1')
        enable_a2 = variant in ('a2', 'b1', 'b2', 'c1')
        enable_b1 = variant in ('b1', 'b2', 'c1')
        enable_b2 = variant in ('b2', 'c1')

        current_total_demand = int(np.sum(self.od_demand_now[0])) if self.od_demand_now.size > 0 else 0
        no_order_mode = (current_total_demand <= 0)

        if enable_b2:
            next_out_vector = self._get_future_out_demand(step_offset=1)
            next2_out_vector = self._get_future_out_demand(step_offset=2)
            lookahead_out = 0.7 * next_out_vector + 0.3 * next2_out_vector
            lookahead_total = float(np.sum(lookahead_out))
            if lookahead_total > 0:
                lookahead_dest_pref = lookahead_out / lookahead_total
            else:
                lookahead_dest_pref = np.zeros(self.ZONE_NUM, dtype=float)
        else:
            lookahead_total = 0.0
            lookahead_dest_pref = np.zeros(self.ZONE_NUM, dtype=float)
        # ======修改结束

        for i in range(0, ZONE_NUM):
            # 先对可发出的车进行一次限制
            # limit_now=int(max(0,(self.available[0][i]-self.od_out[i][self.episode_timestep]/e)*float(action[i])))
            # 上面那个是TRE的
            # =========2026.2.14修改，实验A/B动作解码逻辑==========
            available_now = max(0, int(dispatchable_available[i]))
            current_out_demand = int(np.sum(self.od_demand_now[0][i])) if self.od_demand_now.size > 0 else 0

            if enable_b1 and no_order_mode:
                if enable_b2 and lookahead_total > 0:
                    preposition_ratio = min(0.35, 0.10 + 0.25 * max(0.0, float(action[i])))
                    raw_limit = available_now * preposition_ratio
                    limit_now = int(round(raw_limit))
                    if available_now > 0 and float(action[i]) > 0.20:
                        limit_now = max(1, limit_now)
                    action_table = np.array(lookahead_dest_pref, dtype=float)
                    action_table[i] = 0.0
                else:
                    limit_now = 0
                    action_table = np.array(real_act[i], copy=True)
            else:
                if enable_a1:
                    raw_limit = available_now * math.sqrt(max(0.0, float(action[i])))
                else:
                    raw_limit = available_now * float(action[i])
                limit_now = int(round(raw_limit))

                if enable_a2:
                    if available_now > 0 and current_out_demand > 0 and float(action[i]) > 0.15:
                        limit_now = max(1, limit_now)
                else:
                    if available_now > 0 and float(action[i]) > 0.5:
                        limit_now = max(1, limit_now)
                action_table = np.array(real_act[i], copy=True)

            limit_now = max(0, min(limit_now, available_now))
            # ======修改结束

            while limit_now > 0:
                a = -1
                pos = -1
                for j in range(0, ZONE_NUM):
                    if i == j:
                        continue
                    if action_table[j] > a:
                        pos = j
                        a = action_table[j]

                action_table_true[i][pos] += 1
                limit_now -= 1
                action_table[pos] *= rate

                # print("动作%d"%self.episode_timestep)
        # print(action)
        # print(action_table_true)
        
        #1118注释
        # print(">>> 调用 solveproblem 前尺寸：")
        # print("dist", len(self.dist), len(self.dist[0]))
        # #print("demand", len(self.demand), len(self.demand[0]))
        # print("avail0/1", len(self.available[0]), len(self.available[1]))
        # print("pre_avail", len(pre_avail))
        # print("real_reward", len(real_reward))
        
        #print("reward[t]", len(reward[timestep]))
        
        # for i in range(len(self.car)):
        #     # 当前车辆信息
        #     current_zone = self.car[i][0]
        #     battery = self.car[i][1]
        #     distance_to_target = self.car[i][2]
        #     # 获取目标区域（动作分配后的目标）
        #     target_zone = np.argmax(action_table_true[i]) if np.any(action_table_true[i] > 0) else current_zone
        #
        #     self.vehicle_logs.append({
        #         "timestep": self.episode_timestep,
        #         "vehicle_id": i,
        #         "current_zone": current_zone,
        #         "battery": battery,
        #         "target_zone": target_zone,
        #         "distance_to_target": distance_to_target
        #     })
        
        # 我这篇文章用的是优化方法，你可以不用弄，reward单智能体应该用不上，我就直接让他等同于real_reward
        
        # =========2026.1.30修改，最优解求解==========
        # 调度前执行双向对调净流抵消
        action_table_true = self._cancel_bidirectional_dispatch(action_table_true)
        # ======修改结束

        prev_car = copy.deepcopy(self.car)  #对比solveproblem 前后的状态，识别哪些订单是这一轮新产生的
        
        self.car, self.available, self.car_mem, reward, real_reward, rel_dispatch = gur.solveproblem(self.dist, self.available,
                                                                                            action_table_true, self.car,
                                                                                            reward, -diaodu_min,
                                                                                            self.car_mem,
                                                                                            self.episode_timestep,
                                                                                            real_reward, 0, 0,
                                                                                            pre_avail)

        # 满足需求（满足需求按理来说应该顺序满足，我这边简化一下也给优化做了）
        demand = self.od_demand_now[0]
        
        # 20251202=== DEBUG：打印本时间片需求情况 ===
        #print(f"\n[DEBUG] t={self.episode_timestep} OD 需求订单：")
        
        total_orders = 0
        for o in range(self.ZONE_NUM):
            for d in range(self.ZONE_NUM):
                if demand[o][d] > 0:
                   # print(f"  OD {o} -> {d} : {demand[o][d]} 单")
                    total_orders += demand[o][d]
        
      #  print(f"  本时间片订单总数 = {total_orders}")


        self._last_rld = np.sum(rel_dispatch)  # 修改

        self.car, self.available, self.car_mem, reward, real_reward, rel_demand = gur.solveproblem(self.dist, self.available,
                                                                                            demand, self.car, reward,
                                                                                            manzu_min, self.car_mem,
                                                                                            self.episode_timestep,
                                                                                            real_reward, 0, 0,
                                                                                            pre_avail)

        # =========2026.2.6修改，方案4B：在调度完成后执行充电决策==========
        # 移到这里：两次 solveproblem 之后，让调度后剩余的空闲车辆去充电
        # 优势：避免充电-调度冲突，确保充电决策基于调度后的真实空闲车辆
        # =========2026.2.7修改，方案D2：使用self.current_epoch而不是参数==========
        self.check_idle_vehicles(charge_pref, charging_actions, epoch=self.current_epoch)
        self._sync_dispatchable_available()
        # ======修改结束

        for vid in range(len(self.car)):
            # 目标区域发生变化 → 说明是新订单
            if self.car[vid][0] != prev_car[vid][0]:
                # === 【新增修复 2026.1.7】 车辆离开，释放原区域的充电桩 ===
                old_zone = prev_car[vid][0]
                old_charger_id = prev_car[vid][3]
                
                if old_charger_id != -1:
                    self.release_charger(old_zone, old_charger_id)
                # ===============================================
                
                # =========2026.2.12修改，方案F3：避免覆盖本步已分配的充电任务==========
                # 若车辆已经拥有充电桩计划，不再按普通订单覆盖停车位规划
                if self.car[vid][3] == -1:
                    self.assign_parking_slot(vid, is_charging_order=False)    # is_charging_order=False表示表示充电订单 规划调度和od需求后停车的具体位置
                # ======修改结束
        
        total_demand = np.sum(demand)##0729 计算trb论文指标加入
        self._last_svc = np.sum(real_reward) / total_demand if total_demand > 0 else 0#0729 计算trb论文指标加入
        self._last_srd = np.sum(real_reward) / total_demand if total_demand > 0 else 0  # 修改

        # =========2026.2.14修改，添加服务率惩罚==========
        # 计算服务率：实际服务的订单数 / 总需求数
        served_orders = np.sum(rel_demand)  # 实际服务的订单数
        service_rate = served_orders / total_demand if total_demand > 0 else 1.0

        # 服务率惩罚：如果服务率低于阈值，给予适度惩罚
        service_rate_penalty = 0
        if service_rate < 0.5:  # 服务率低于50%时开始惩罚
            # 惩罚力度：(0.5 - service_rate) * 8000
            # 例如：service_rate=0时，惩罚=4000；service_rate=0.3时，惩罚=1600
            service_rate_penalty = (0.5 - service_rate) * 8000
        # ======修改结束

        # ====== Debug log start ======
        #print(f"\n[Step {self.episode_timestep}] Action received:")

        # 打印出发倾向
       # print("  出发倾向:")
       # for i in range(ZONE_NUM):
       #     print(f"    区域 {i}: {action[i]:.3f}  可发车数 ≈ {int(self.available[0][i] * action[i])}")

        # 打印目的地倾向
       # print("  目的地倾向:")
        #for i in range(ZONE_NUM):
        #    print(f"    区域 {i}: {action[ZONE_NUM + i]:.3f}")

        #print("  充电倾向:")
        #for i in range(ZONE_NUM):
        #    print(f"    区域 {i}: {charge_pref[i]:.3f}")

        # 打印OD调度矩阵（从 i 区域 到 j 区域 发出车辆数）
        #print("  调度决策矩阵（action_table_true）:")
        #for i in range(ZONE_NUM):
         #   row = "    从区域 {:d} → ".format(i)
         #   row += "  ".join(["{:d}".format(int(action_table_true[i][j])) for j in range(ZONE_NUM)])
         #   print(row)
        # ====== Debug log end ======
        self.episode_timestep += 1

        # 写入新的状态

        self.update_od_left()

        self.write_od_now()
        done = self.terminal##0729 计算trb论文指标加入

        # =========2026.2.5修改，方案1：添加即时充电奖励==========
        # 1. 服务订单收益（已在real_reward中）
        service_revenue = sum(real_reward)

        # 2. 充电加分：统计正在充电的车辆数
        charging_bonus = 0
        charging_count = 0
        for i in range(self.CAR_NUM):
            # 判断车辆是否在充电：charger_id != -1 且不在移动中
            if self.car[i][3] != -1 and self.car[i][2] == 0:
                charging_count += 1
        # =========2026.2.12修改，实验19：大幅提升充电奖励==========
        # =========2026.2.13修改，进一步提升充电奖励==========
        # =========2026.2.13修改，完整组合方案：大幅提升充电奖励==========
        # 问题：充电奖励5000仍不足以抵消服务收入的吸引力
        # 解决：提升到15000，配合允许充电车辆服务，消除机会成本
        # 即时充电奖励：每辆充电车辆+15000（从5000提升到15000）
        charging_bonus = 15000 * charging_count
        # ======修改结束

        # 3. 低电量惩罚：分段惩罚
        # =========2026.2.11修改，实验12：渐进式训练（Curriculum Learning）==========
        # 根据当前训练轮数动态调整SOC惩罚强度
        epoch = self.current_epoch if hasattr(self, 'current_epoch') else 0

        # =========2026.2.13修改，修正Curriculum Learning策略==========
        # 问题：后期惩罚过低，导致策略不稳定
        # 解决：保持较高的SOC惩罚，确保策略稳定性
        # 阶段1（0-300轮）：高惩罚，强制学习充电
        # 阶段2（300-600轮）：中等惩罚，优化策略
        # 阶段3（600+轮）：稳定惩罚，保持性能
        if epoch < 300:
            # 高惩罚阶段
            penalty_scale = 1.5
            vehicle_penalty = 750
        elif epoch < 600:
            # 中等惩罚阶段（线性衰减）
            progress = (epoch - 300) / 300  # 0到1
            penalty_scale = 1.5 - 0.3 * progress  # 1.5衰减到1.2
            vehicle_penalty = 750 - 150 * progress  # 750衰减到600
        else:
            # 稳定惩罚阶段（不再降低）
            penalty_scale = 1.2
            vehicle_penalty = 600
        # ======修改结束

        soc_penalty = 0
        low_soc_count = 0
        for i in range(self.CAR_NUM):
            soc = self.car[i][1]
            # =========2026.2.13修改，完整组合方案：添加指数级SOC惩罚==========
            # 当SOC<400时，惩罚加倍（3x），强化低电量危机感
            exponential_multiplier = 3.0 if soc < 400 else 1.0
            # ======修改结束

            if soc >= 600:
                penalty = 0
            elif soc >= 400:
                penalty = 2.0 * penalty_scale * (600 - soc)
            elif soc >= 200:
                penalty = 2.0 * penalty_scale * 200 + 10.0 * penalty_scale * (400 - soc)
                penalty *= exponential_multiplier  # 应用指数惩罚
                low_soc_count += 1
            else:
                penalty = 2.0 * penalty_scale * 200 + 10.0 * penalty_scale * 200 + 50.0 * penalty_scale * (200 - soc)
                penalty *= exponential_multiplier  # 应用指数惩罚
                low_soc_count += 1
            soc_penalty += penalty

        low_soc_vehicle_penalty = low_soc_count * vehicle_penalty
        soc_penalty += low_soc_vehicle_penalty
        # ======修改结束

        # =========2026.2.13修改，添加SOC维持奖励（Reward Shaping）==========
        # =========2026.2.13修改，完整组合方案：增强SOC维持奖励==========
        # 目的：奖励保持高SOC的行为，而不仅仅是惩罚低SOC
        # 计算车队平均SOC，给予正向奖励
        total_soc = sum(self.car[i][1] for i in range(self.CAR_NUM))
        avg_soc = total_soc / self.CAR_NUM

        # SOC维持奖励：平均SOC越高，奖励越大
        # 目标：鼓励保持SOC在1200以上（60%）
        # 从0.5提升到2.0，增强高SOC维持的激励
        if avg_soc >= 1200:
            soc_maintain_bonus = (avg_soc - 1200) * 2.0  # 超过1200的部分，每点2.0奖励（从0.5提升）
        else:
            soc_maintain_bonus = 0

        # =========2026.2.13修改，完整组合方案：添加充电维护奖励==========
        # 目的：额外奖励充电行为，确保充电策略的稳定性
        # 充电维护奖励：每辆充电车辆额外+3000
        # 与充电奖励(15000)叠加，总计18000/车
        charging_maintenance_bonus = 3000 * charging_count
        # ======修改结束

        # 4. 调度成本（已在real_reward中，但需要单独计算）
        # 注意：第一次solveproblem的real_reward包含调度成本（负值）
        # 这里不需要额外减去，因为已经在real_reward中了

        # =========2026.2.5修改，实验2：移除固定惩罚==========
        # 5. 固定惩罚（移除以避免掩盖充电奖励信号）
        fixed_penalty = 0  # 原500改为0
        # ======修改结束

        # =========2026.2.12修改，实验19：充电奖励不缩放==========
        # =========2026.2.13修改，添加SOC维持奖励==========
        # =========2026.2.13修改，完整组合方案：添加充电维护奖励==========
        # =========2026.2.14修改，添加服务率惩罚==========
        # 新奖励函数：充电奖励单独计算，不除以10
        # 添加SOC维持奖励，鼓励保持高SOC
        # 添加充电维护奖励，进一步强化充电激励
        # 添加服务率惩罚，鼓励服务更多订单
        reward_value = (service_revenue - soc_penalty) / 10 + charging_bonus + charging_maintenance_bonus + soc_maintain_bonus - fixed_penalty - service_rate_penalty
        # ======修改结束

        # =========2026.2.4修改，添加充电动作和调度矩阵到info==========
        info = {            ##0729 计算trb论文指标加入
            "rld": self._last_rld,          ##0729 计算trb论文指标加入
            "svc": self._last_svc,
            "srd": self._last_srd,  # 修改
            "dispatch_orders": rel_dispatch,  # 调度订单矩阵
            "demand_orders": rel_demand,  # 需求订单矩阵
            "charging_actions": charging_actions,  # 充电动作记录
            "action_table_true": action_table_true  # RL调度倾向矩阵
        }           ##0729 计算trb论文指标加入
        # ======修改结束

        if self.terminal:
            # 如果采用平均reward，是否需要计算还剩下的油费和租费
            # =========2026.2.5修改，终止时也使用即时奖励逻辑==========
            # 计算新奖励函数的各个部分
            service_revenue = sum(real_reward)

            charging_count = 0
            for i in range(self.CAR_NUM):
                if self.car[i][3] != -1 and self.car[i][2] == 0:
                    charging_count += 1
            # =========2026.2.12修改，方案F4：终止步奖励与中间步完全一致==========
            charging_bonus = 3000 * charging_count
            # ======修改结束

            # =========2026.2.11修改，实验12：终止时也使用渐进式惩罚==========
            epoch = self.current_epoch if hasattr(self, 'current_epoch') else 0

            if epoch < 200:
                penalty_scale = 1.0
                vehicle_penalty = 500
            elif epoch < 400:
                progress = (epoch - 200) / 200
                penalty_scale = 1.0 - 0.25 * progress
                vehicle_penalty = 500 - 125 * progress
            else:
                penalty_scale = 0.75
                vehicle_penalty = 375

            soc_penalty = 0
            low_soc_count = 0
            for i in range(self.CAR_NUM):
                soc = self.car[i][1]
                if soc >= 600:
                    penalty = 0
                elif soc >= 400:
                    penalty = 2.0 * penalty_scale * (600 - soc)
                elif soc >= 200:
                    penalty = 2.0 * penalty_scale * 200 + 10.0 * penalty_scale * (400 - soc)
                    low_soc_count += 1
                else:
                    penalty = 2.0 * penalty_scale * 200 + 10.0 * penalty_scale * 200 + 50.0 * penalty_scale * (200 - soc)
                    low_soc_count += 1
                soc_penalty += penalty

            low_soc_vehicle_penalty = low_soc_count * vehicle_penalty
            soc_penalty += low_soc_vehicle_penalty
            # ======修改结束

            # =========2026.2.14修改，终止时也添加服务率惩罚==========
            # 计算服务率：实际服务的订单数 / 总需求数
            served_orders = np.sum(rel_demand)  # 实际服务的订单数
            service_rate = served_orders / total_demand if total_demand > 0 else 1.0

            # 服务率惩罚：如果服务率低于阈值，给予适度惩罚
            service_rate_penalty = 0
            if service_rate < 0.5:  # 服务率低于50%时开始惩罚
                # 惩罚力度：(0.5 - service_rate) * 8000
                # 例如：service_rate=0时，惩罚=4000；service_rate=0.3时，惩罚=1600
                service_rate_penalty = (0.5 - service_rate) * 8000
            # ======修改结束

            # =========2026.2.5修改，实验2：终止时也移除固定惩罚==========
            fixed_penalty = 0  # 原500改为0
            final_reward = (service_revenue - soc_penalty) / 10 + charging_bonus - fixed_penalty - service_rate_penalty
            # ======修改结束
            # ======修改结束
            return self.state, final_reward, self.terminal, info
        else:
            self._last_episode_timestep = self.episode_timestep
            
            # --- Write logs for this step ---
            with open(self.log_file_path, "a") as f:
                for vid in range(self.CAR_NUM):
                    # Get logged data or defaults
                    v_log = self.current_step_logs.get(vid, {})
                    
                    # Determine final behavior
                    behavior = "Idle"
                    if self.car[vid][2] > 0:
                         behavior = f"Moving_to_{self.car[vid][0]}"
                    elif self.car[vid][0] in CHARGER_ZONES: # Assuming charging if in charger zone and not moving (simplified)
                         # Check if actually charging? For now just location
                         behavior = f"In_Zone_{self.car[vid][0]}"
                    else:
                         behavior = f"In_Zone_{self.car[vid][0]}"

                    # If we have specific charging decision log, use it to refine behavior description
                    if v_log.get("decision_charge"):
                        behavior += "_ChargingDecided"
                    
                    # Construct line
                    # Time,VehicleID,Zone,SOC,Action_ChargePref,Formula_Score,Formula_Prob,Decision_Charge,Final_Behavior
                    line = f"{t},{vid},{self.car[vid][0]},{self.car[vid][1]:.2f}," \
                           f"{v_log.get('pref', -1):.3f},{v_log.get('score', -1):.3f}," \
                           f"{v_log.get('prob', -1):.3f},{v_log.get('decision_charge', 'N/A')},{behavior}\n"
                    f.write(line)
            # -------------------------------- 2026.1.30注释
            # print("\n====== 动作后 self.car ======")
            # for i in range(self.CAR_NUM):
            #     print(f"车辆 {i}: zone={self.car[i][0]}, SOC={self.car[i][1]}, remain={self.car[i][2]}")
            # print("================================\n")

            # =========2026.2.2修改，使用新奖励函数==========
            return self.state, reward_value, self.terminal, info
            # ======修改结束

    @property
    def terminal(self):
        return self.is_terminal()

    def is_terminal(self):
        if self.episode_timestep >= TIME_NUM:
            return 1
        else:
            return 0

    def set_training_mode(self, training):
        """
        设置训练模式
        training = 0: 测试模式（保存到 RL-DDPG_actions.txt）
        training = 1: 训练模式（不保存）
        training = 2: 探索期记录模式（保存到 RL-DDPG_actions_exploration.txt）
        """
        self.training = training

    # =========2026.2.7修改，方案D2：添加set_epoch方法==========
    def set_epoch(self, epoch):
        """设置当前训练轮数（用于探索策略衰减）"""
        self.current_epoch = epoch
    # ======修改结束

    # =========2026.2.14修改，实验方案开关==========
    def set_experiment_variant(self, variant):
        """设置实验方案：baseline/a1/a2/b1/b2/c1。"""
        self.exp_variant = str(variant).lower()
    # ======修改结束

    def reset(self):
        # 重置环境并返回初始状态
        new = self.new
        self.episode_timestep = 0
        self._last_episode_timestep = 0
        self.n_episode += 1
        self.write_od_left(new)
        self.write_od_now()
        self.debug_log = []
        
        self.write_ini_cap(new)
        # 🔹 2. 立刻打印充电桩状态（确认初始化是否成功）
        self.print_charging_stations()
        #self.print_vehicle_status(0)
       # self.car_mem = [[-1, 0, 5] for _ in range(CAR_NUM)]
        self.car_mem = [[-1, 0, 5] for _ in range(len(self.car))]

        if new:
            self.mem_available = copy.deepcopy(self.available)
            self.new = 0

        self._last_edge = None
        self._last_episode_timestep = self.episode_timestep

        return self.state

    def render(self, mode='human'):
        # 渲染环境的逻辑，一般不涉及，可视化内容
        pass

    def seed(self, seed=None):
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

