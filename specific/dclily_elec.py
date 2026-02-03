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

OBS_SPACE = spaces.MultiDiscrete((TIME_SLICE * ZONE_NUM) * [OD_MAX, OD_MAX,CHARGER_STATES, NUM_CHARGERS])


# 当前状态为：od_in与od_out，观察未来TIME_SLICE个时间片，多维离散状态空间

class dclily(gym.Env):
    def __init__(self):
        super(dclily, self).__init__()

        # 定义动作空间和观测空间
        self.config = config  #1011增强rl逻辑时 新增

        self.training = 1  # 1是否处于训练模式的标志
        self.action_space = ACTION_SPACE  # 1环境的动作空间
        self.observation_space = OBS_SPACE  # 1一个观测空间的对象

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
        
        car_index = 0
        
        for zone_id in range(ZONE_NUM):
            zone_car_cnt = int(zone_init_car[zone_id])
            chargers = self.charging_stations[zone_id]
            num_chargers = len(chargers)
            
            # 能使用的充电桩数量
            use = min(zone_car_cnt, num_chargers)
            
            # ① 前 use 个车占用充电桩
            for j in range(use):
                charger = chargers[j]
                charger["occupied_by"] = car_index
                self.car[car_index][3] = charger["charger_id"]
                car_index += 1
            
            # ② 剩余车辆只停在停车位，不占桩
            for j in range(zone_car_cnt - use):
                self.car[car_index][3] = -1
                car_index += 1
        
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
        
        # 4) 拼接成 [OD, OD, CHARGER_STATES, NUM_CHARGERS] * (TIME_SLICE*ZONE_NUM)
        state = np.concatenate([od_part, charger_states, num_chargers_feat])
        # 保险断言：长度必须满足 4 * ZONE_NUM * TIME_SLICE
        assert state.shape[0] == 4 * ZONE_NUM * TIME_SLICE, \
            f"state dim {state.shape[0]} != 4*ZONE_NUM*TIME_SLICE"
        return state
        
        # return np.concatenate([od_part, np.array(charger_states)])

    @property
    def state(self):
        return self.get_micro_state()

    def _sigmoid(self, x):
        """标准 Sigmoid 激活"""
        return 1.0 / (1.0 + math.exp(-x))

    def rl_should_charge(self, vehicle_id, charge_pref):
        """
        使用 RL 输出来决定是否充电（替换固定阈值法）
        charge_pref: 动作向量中充电倾向部分
        """
        # 若 self.config 存在则使用权重，否则使用默认
        cfg = getattr(self, "config", None)
        w = (cfg.get("RL_CHARGE_WEIGHTS") if cfg else None) or \
            {"bias": 0.0, "soc": -3.0, "pref": 2.0, "queue": -1.5, "dist": -0.5}   #权重

        # 车辆状态
        zone = self.car[vehicle_id][0]
        soc = max(0.0, min(1.0, self.car[vehicle_id][1] / BATTERY_CAPACITY))
        pref = float(charge_pref[zone])

        # 区域车位紧张度（空余越少，queue_term 越大）
        used_ratio = 1.0 - (self.available[0][zone] / (self.mem_available[0][zone] + 1e-5))
        queue_term = used_ratio

        # 距离最近充电区（越远越抑制充电意愿）
        nearest_zone, _  = self.find_nearest_charger(zone)
        dist_term = 0.0
        if nearest_zone != -1:
            dist_term = float(self.dist[zone][nearest_zone]) / (DIST_MAX + 1e-5)
        
        if soc > 0.6:
          #  print(f"[t={self.episode_timestep}] 车辆{vehicle_id} soc={soc:.3f} > 0.6，强制不充电")
            return False

        # 综合打分 + Sigmoid 转概率
        
        score = (w["bias"]
                 + w["soc"] * soc
                 + w["pref"] * pref
                 #+ w["queue"] * queue_term
                 + w["dist"] * dist_term)
        p = self._sigmoid(score)
        
        # 判断是否去充电的测试  dist:当前区域距离最近区域的 距离，归一化后的结果dist
        # print(
        #     f"[t={self.episode_timestep}] 车辆{vehicle_id} RL充电打分： "
        #     f"soc={soc:.3f}, pref={pref:.3f}, dist={dist_term:.3f}, "
        #     f"score={score:.3f}, p={p:.3f}"
        # )
        
        # 裁剪范围，防极端 0/1
        lo, hi = (cfg.get("RL_CHARGE_DECISION_CLIP", [0.05, 0.95]) if cfg else [0.05, 0.95])

        p = min(max(p, lo), hi)
        r = random.random()
        # print(f"车辆{vehicle_id}: soc={soc:.2f}, pref={pref:.2f}, dist={dist_term:.2f}, "
        #      f"score={score:.2f}, p={p:.3f}, random={r:.3f}, training={self.training}")

        # 训练期：随机采样探索；推理期：按概率阈值决策
        decision = False
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
            "score": score,
            "prob": p,
            "decision_charge": decision
        })
        
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
    
    def check_idle_vehicles(self, charge_pref):
        """检查空闲车辆状态（由 RL 决定是否充电；再决定去哪儿充）"""
        # 创建一个临时列表来存储需要扣除available的区域，避免在循环中直接修改影响逻辑
        cars_to_hide = []
        
        for i in range(len(self.car)):
            # 仅检查未执行任务的车辆
            if self.car[i][2] != 0:
                continue
            
            # RL 概率决策：是否要充电
            want_charge = self.rl_should_charge(i, charge_pref)
            # print(f"[t={self.episode_timestep}] 车辆{i} want_charge = {want_charge}")
            
            if not want_charge:
                continue  # 不充电则保持待命
            
            # 若决定充电 → 调用现有的“去哪充”逻辑
            current_zone = self.car[i][0]
            target_charger = self.find_preferred_charger(current_zone, charge_pref)
            
            if target_charger != -1:
                # print(f"[t={self.episode_timestep}] 车辆{i} 前往充电站 zone={target_charger}")
                # === 【新增修复 1】: 车要走了，如果手里有桩，必须先释放！===
                old_charger_id = self.car[i][3]
                if old_charger_id != -1:
                    self.release_charger(current_zone, old_charger_id)
                    self.car[i][3] = -1  # 清除身上的标记
                # ① 设置新的目标区域
                self.car[i][0] = target_charger
                
                # ② 设置行驶距离
                dist_to_charge = self.dist[current_zone][target_charger]
                self.car[i][2] = dist_to_charge
                
                # ③ 规划停车位
                self.assign_parking_slot(i, is_charging_order=True)
                
                # ④ 立即物理锁桩
                planned_id = self.car[i][3]
                if planned_id != -1:
                    if target_charger in self.charging_stations:
                        for station in self.charging_stations[target_charger]:
                            if station['charger_id'] == planned_id:
                                station['occupied_by'] = i
                                break
                
                # === 【核心修复】 ===
                # 如果是本地充电 (距离为0)，必须从 available 中移除该车，
                # 防止 Gurobi 在随后的 solveproblem 中将其调度走。
                if dist_to_charge == 0:
                    self.available[0][current_zone] -= 1
                    # print(f"[System] 车辆{i} 在区域{current_zone}本地充电，已从调度池移除。")
        
        # 确保 available 不会变成负数 (理论上逻辑正确不会发生，但为了安全)
        for z in range(self.ZONE_NUM):
            if self.available[0][z] < 0:
                self.available[0][z] = 0
    
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
                if self.car[i][1] >= BATTERY_CAPACITY:
                    self.car[i][1] = BATTERY_CAPACITY
                
                # (可选) 如果充满了，是否自动释放充电桩？
                # if self.car[i][1] >= BATTERY_CAPACITY:
                #    self.car[i][3] = -1  # 释放充电桩状态
            if self.car[i][2] > 0:
                self.car[i][2] -= car_speed_min * TIME_LENGTH * con
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
                   # 【关键修改】只有没占桩（没在充电）的车，才放入 available 供 Gurobi 调度
                    if planned_charger_id == -1:
                        self.available[0][self.car[i][0]] += 1
                    # # 原有逻辑：到达后判断下一步（注意：这通常是针对刚才完成移动的车）
                    # #处理充电逻辑，这一part有什么用？？？bug
                    # target_zone1 = self.car[i][0]
                    # target_zone = self.handle_vehicle_charging(i, target_zone1)
            
            i += 1
        self.check_idle_vehicles(charge_pref)

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


        for i in range(0, ZONE_NUM):
            # 先对可发出的车进行一次限制
            # limit_now=int(max(0,(self.available[0][i]-self.od_out[i][self.episode_timestep]/e)*float(action[i])))
            # 上面那个是TRE的
            limit_now = int(max(0, (self.available[0][i]) * float(action[i])))
            action_table = real_act[i]

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
        
        for vid in range(len(self.car)):
            # 目标区域发生变化 → 说明是新订单
            if self.car[vid][0] != prev_car[vid][0]:
                # === 【新增修复 2026.1.7】 车辆离开，释放原区域的充电桩 ===
                old_zone = prev_car[vid][0]
                old_charger_id = prev_car[vid][3]
                
                if old_charger_id != -1:
                    self.release_charger(old_zone, old_charger_id)
                # ===============================================
                
                # 调度 / OD 订单 → 非充电订单
                self.assign_parking_slot(vid, is_charging_order=False)    # is_charging_order=False表示表示充电订单 规划调度和od需求后停车的具体位置
        
        total_demand = np.sum(demand)##0729 计算trb论文指标加入
        self._last_svc = np.sum(real_reward) / total_demand if total_demand > 0 else 0#0729 计算trb论文指标加入
        self._last_srd = np.sum(real_reward) / total_demand if total_demand > 0 else 0  # 修改
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

        # =========2026.2.2修改，新奖励函数==========
        # 1. 服务订单收益（已在real_reward中）
        service_revenue = sum(real_reward)

        # 2. 充电加分：统计正在充电的车辆数
        charging_bonus = 0
        charging_count = 0
        for i in range(self.CAR_NUM):
            # 判断车辆是否在充电：charger_id != -1 且不在移动中
            if self.car[i][3] != -1 and self.car[i][2] == 0:
                charging_count += 1
        charging_bonus = 50 * charging_count

        # 3. 低电量惩罚：分段惩罚
        soc_penalty = 0
        for i in range(self.CAR_NUM):
            soc = self.car[i][1]
            if soc >= 600:
                penalty = 0
            elif soc >= 200:
                penalty = 0.5 * (600 - soc)
            else:
                penalty = 0.5 * 400 + 5.0 * (200 - soc)
            soc_penalty += penalty

        # 4. 调度成本（已在real_reward中，但需要单独计算）
        # 注意：第一次solveproblem的real_reward包含调度成本（负值）
        # 这里不需要额外减去，因为已经在real_reward中了

        # 5. 固定惩罚
        fixed_penalty = 500

        # 新奖励函数（添加缩放因子/10以控制波动）
        reward_value = (service_revenue + charging_bonus - soc_penalty) / 10 - fixed_penalty
        # ======修改结束

        info = {            ##0729 计算trb论文指标加入
            "rld": self._last_rld,          ##0729 计算trb论文指标加入
            "svc": self._last_svc,
            "srd": self._last_srd,  # 修改
            "dispatch_orders": rel_dispatch,  # 调度订单矩阵
            "demand_orders": rel_demand  # 需求订单矩阵
        }           ##0729 计算trb论文指标加入

        if self.terminal:
            # 如果采用平均reward，是否需要计算还剩下的油费和租费
            # =========2026.2.2修改，使用新奖励函数==========
            # 计算新奖励函数的各个部分
            service_revenue = sum(real_reward)

            charging_count = 0
            for i in range(self.CAR_NUM):
                if self.car[i][3] != -1 and self.car[i][2] == 0:
                    charging_count += 1
            charging_bonus = 50 * charging_count

            soc_penalty = 0
            for i in range(self.CAR_NUM):
                soc = self.car[i][1]
                if soc >= 600:
                    penalty = 0
                elif soc >= 200:
                    penalty = 0.5 * (600 - soc)
                else:
                    penalty = 0.5 * 400 + 5.0 * (200 - soc)
                soc_penalty += penalty

            fixed_penalty = 500
            final_reward = (service_revenue + charging_bonus - soc_penalty) / 10 - fixed_penalty
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
        self.training = training

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

