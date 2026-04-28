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


# 鐩稿璺緞鍐欐硶    褰撳墠鏂囦欢鎵€鍦ㄧ洰褰曪細specific/
current_dir = os.path.dirname(os.path.abspath(__file__))

# config.json 灏卞湪鍚屼竴鐩綍
config_path = os.path.join(current_dir, 'config.json')

#print('[DEBUG] Loading config from:', config_path)

with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
    
#-------------1---------------------------
# script_dir = os.path.dirname(os.path.abspath('F:\\AAAA缁勪細\\浠ｇ爜澶囦唤\\2501007鍔犱簡conj鍙傛暟鍚庡浠絓\dclily_torch_ddpg_demo鍩轰簬0425淇敼\\specific'))
# config_path = os.path.join(script_dir, 'specific', 'config.json')
#
# with open(config_path, 'r') as file:
#     config = json.load(file)

# 鍔犺浇JSON鏂囦欢-----22222222222-------------
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

# =========2026.2.19淇敼锛屽垏鎹㈠埌SIP澶ц妯℃暟鎹泦==========
# 寮哄埗瑕嗙洊閰嶇疆鍙傛暟浠ヤ娇鐢⊿IP鏁版嵁闆嗭紙50鍖哄煙锛?00杞﹁締锛?
ZONE_NUM = 50
CAR_NUM = 200
TIME_NUM = 26
TIME_LENGTH = 30
TIME_SLICE = 1
OD_MAX = 13
DIST_MAX = 200
# ======淇敼缁撴潫

car_speed_min = config['car_speed_min']
manzu_min = config['manzu_min']
diaodu_min = config['diaodu_min']
elec_min = config['elec_min']
BATTERY_CAPACITY = config['BATTERY_CAPACITY']

# =========2026.2.19淇敼锛屽垏鎹㈠埌SIP鏁版嵁闆嗚矾寰?=========
path = './data/SIP/'
training_set_number = 30
BATTERY_CAPACITY = 3000  # SIP鏁版嵁闆嗕娇鐢?000瀹归噺
# ======淇敼缁撴潫
CHARGER_STATES = config['CHARGER_STATES']  # 鍏呯數鍖哄煙
#NUM_CHARGERS = config['NUM_CHARGERS']  # 鍏呯數妗╂暟閲忥紝寮冪敤
#NO_CHARGER_ZONES = config['NO_CHARGER_ZONES']

# =========2026.2.19淇敼锛屽姞杞絊IP鏁版嵁闆嗙殑ZONES閰嶇疆==========
import json as json_module
sip_config_path = os.path.join(current_dir, '..', 'data', 'SIP', 'config_sip.json')
with open(sip_config_path, 'r', encoding='utf-8') as f:
    sip_config = json_module.load(f)
ZONES = sip_config['ZONES']
CHARGER_ZONES = sip_config['CHARGER_ZONES']
# ======淇敼缁撴潫


# Convert CHARGER_ZONES to list if it's not already
if isinstance(CHARGER_ZONES, int):
    CHARGER_ZONES = [CHARGER_ZONES]

#251007鍔犲叆
# =========2026.2.19淇敼锛屼娇鐢⊿IP閰嶇疆鐢熸垚CHARGER_PER_ZONE==========
if "ZONES" in sip_config:
    NUM_CHARGERS = sum(len(zone.get("charging_slots", [])) for zone in sip_config["ZONES"])   # 鍏呯數妗╂暟閲?
    CHARGER_PER_ZONE = [zone.get("charging_slots", []) for zone in sip_config["ZONES"]]  #鍏呯數妗╀簩缁村垪琛?
    # CHARGER_ZONES宸茬粡浠巗ip_config鍔犺浇锛屼笉闇€瑕侀噸澶?
    #NO_CHARGER_ZONES = [zone["zone_id"] for zone in sip_config["ZONES"] if not zone.get("charging_slots")] #寮冪疆锛屾棤鐢ㄥ彉閲?
# ======淇敼缁撴潫
else:
    # 鍚戝悗鍏煎鏃ч厤缃?
    NUM_CHARGERS = config.get('NUM_CHARGERS', 0)
    CHARGER_PER_ZONE = config.get('CHARGER_PER_ZONE', [0]*ZONE_NUM)
    CHARGER_ZONES = config.get('CHARGER_ZONES', [])
#    NO_CHARGER_ZONES = config.get('NO_CHARGER_ZONES', [])

# 瀹氫箟鍔ㄤ綔銆佺姸鎬?

ACTION_SPACE = spaces.Box(low=0, high=1, shape=(3 * ZONE_NUM,))


# =========2026.2.20淇敼锛岀粺涓€鐘舵€佺淮搴﹀绾?=========
def _expected_state_dim():
    return 3 * TIME_SLICE * ZONE_NUM + 2 * ZONE_NUM + 1


def _build_observation_space():
    nvec = (
        (TIME_SLICE * ZONE_NUM) * [OD_MAX, OD_MAX, CHARGER_STATES]
        + ZONE_NUM * [10]
        + ZONE_NUM * [20]
        + [TIME_NUM]
    )
    return spaces.MultiDiscrete(nvec)
# ======淇敼缁撴潫



# 褰撳墠鍔ㄤ綔涓猴細鍙戣溅鍊惧悜銆佺洰鐨勫湴鍊惧悜銆佸厖鐢靛€惧悜

# =========2026.2.19淇敼锛孲IP鏁版嵁闆嗭細鏇存柊鐘舵€佺┖闂村畾涔夛紙v5锛?51缁达級==========
# 鐘舵€佺┖闂?v5 (251 dimensions for SIP dataset):
#   - OD闇€姹?In): TIME_SLICE * ZONE_NUM 缁?(50)
#   - OD闇€姹?Out): TIME_SLICE * ZONE_NUM 缁?(50)
#   - 鍏呯數妗╃姸鎬? TIME_SLICE * ZONE_NUM 缁?(50)
#   - 鍖哄煙骞冲潎SOC: ZONE_NUM 缁?(50)
#   - 鍖哄煙杞﹁締鏁伴噺: ZONE_NUM 缁?(50)
#   - 褰撳墠鏃堕棿姝? 1 缁?
# 鎬昏锛?0 + 50 + 50 + 50 + 50 + 1 = 251缁?
OBS_SPACE = _build_observation_space()
# ======淇敼缁撴潫


# 褰撳墠鐘舵€佷负锛歰d_in涓巓d_out锛岃瀵熸湭鏉IME_SLICE涓椂闂寸墖锛屽缁寸鏁ｇ姸鎬佺┖闂?

class dclily(gym.Env):
    def __init__(self):
        super(dclily, self).__init__()

        # 瀹氫箟鍔ㄤ綔绌洪棿鍜岃娴嬬┖闂?
        self.config = config  #1011澧炲己rl閫昏緫鏃?鏂板

        self.training = 1  # 1鏄惁澶勪簬璁粌妯″紡鐨勬爣蹇?
        self.action_space = ACTION_SPACE  # 1鐜鐨勫姩浣滅┖闂?
        self.observation_space = OBS_SPACE  # 1涓€涓娴嬬┖闂寸殑瀵硅薄

        # =========2026.2.7淇敼锛屾柟妗圖2锛氭坊鍔爀poch杩借釜==========
        self.current_epoch = 0  # 褰撳墠璁粌杞暟锛堢敤浜庢帰绱㈢瓥鐣ヨ“鍑忥級
        # ======淇敼缁撴潫

        # =========2026.2.20淇敼锛屽疄楠屼笌濂栧姳寮€鍏冲彲閰嶇疆==========
        self.exp_variant = str(self.config.get('exp_variant', 'a1')).lower()
        self.reward_mode = str(self.config.get('reward_mode', 'base')).lower()
        if self.reward_mode not in ('base', 'shaping'):
            self.reward_mode = 'base'
        self.reward_weights = {
            'charging_bonus': float(self.config.get('reward_w_charging_bonus', 1.0)),
            'charging_maintenance': float(self.config.get('reward_w_charging_maintenance', 1.0)),
            'soc_maintain': float(self.config.get('reward_w_soc_maintain', 1.0)),
            'service_rate_penalty': float(self.config.get('reward_w_service_rate_penalty', 1.0)),
        }
        # ======淇敼缁撴潫

        # 鑷瀹氫箟

        self._last_episode_timestep = 0  # 1涓婁竴鍥炲悎鐨勬椂闂存鏁?
        self.episode_timestep = 0  # 1褰撳墠鍥炲悎鐨勬椂闂存鏁?
        self.n_episode = 0  # 1鍥炲悎璁℃暟鍣?

        self.dist = np.array(pd.read_csv(path + 'od_demand_dist.csv', header=None))  # 1''閲岄潰鏄痗sv鏂囦欢鐨勫悕瀛?2 np.array鏄浆鎹㈡暟缁?
        self.od_demand_left = np.array([])
        self.od_demand_now = np.array([])
        self.available = np.array([])
        self.mem_od_demand = []  # 淇濆瓨鏁版嵁
        self.mem_available = np.array([])
        self.car = np.array([])  # 锛?瀛樺偍杞﹁締淇℃伅锛夌涓€浣嶆暟锛氬綋鍓嶇洰鐨勫湴锛岀浜屼綅鏁帮細鐩墠鐢甸噺鍙椹跺叕閲屾暟锛涚涓変綅鏁帮細璺濈鐩殑鍦拌窛绂籸emain锛?>0 琛ㄧず鍦ㄨ矾涓娿€傜數閲忎細鎻愬墠鎵ｉ櫎  2025.12.8 绗洓浣嶆暟瑙勫垝鍋滆溅浣嶇紪鍙凤紙remain==0 鏃舵墠鏄墿鐞嗗崰鐢級 charger_id锛岋紙涓嶈兘鍏呯數灏辨槸 -1锛?
        self.car_mem = np.array([[-1, 0, 5] for _ in range(
            len(self.car))])  # vehicle priority(浣犲簲璇ョ敤涓嶄笂) 绗竴浣嶆暟锛氳皟搴one锛涚浜屼綅鏁帮細璋冨害鏃堕棿鐗?绗笁浣嶆暟锛氭槸鍚﹁璋冨害锛涙槸涓?锛屽惁涓?(鏀逛负鍏朵粬璁℃暟鏂规硶)
        self.new = 1  # 濡傛灉涓嶆槸鏂扮殑鐜锛屼笉闇€瑕侀噸澶嶈浇鍏ユ暟鎹?
        self.cong = np.array(pd.read_csv(path + 'cong.csv'))
        # self.pla_num=PL_NUM
        # self.dd=np.ones((ZONE_NUM,ZONE_NUM))*0
        # self.sorted_dist=[]
        self.od_in = np.array([])
        self.od_out = np.array([])
        self.mem_od_in = []
        self.mem_od_out = []
        self.vehicle_logs = []  # 姣忔璁板綍杞﹁締鐘舵€?
        self.ZONE_NUM = ZONE_NUM
        self.PL_NUM = PL_NUM
        self.CAR_NUM = CAR_NUM
        self.TIME_NUM = TIME_NUM
        self.TIME_SLICE = TIME_SLICE
        self.expected_state_dim = _expected_state_dim()
        
        #self.charging_usage = [0] * ZONE_NUM  #鍙鍚勫尯鍩熶娇鐢ㄧ殑鍏呯數妗╄鏁?
        self.charging_stations = {
            zone_id: [
                {"charger_id": charger_id, "type": "fast", "occupied_by": None, "power": 60}
                for charger_id in zone_cfg["charging_slots"]
            ]
            for zone_id, zone_cfg in enumerate(ZONES)
        }      #鍏呯數妗╁璞″垪琛紝鐩墠鏈夊厖鐢垫々缂栧彿銆佸厖鐢垫々绫诲瀷銆佽鍗犱簡娌℃湁锛岀數閲?
        
        # Log extraction initialization
        self.log_data = []
        self.current_step_logs = {}
        self.log_file_path = "vehicle_action_log.csv"
        # Initialize CSV with header
        with open(self.log_file_path, "w") as f:
            f.write("Time,VehicleID,Zone,SOC,Action_ChargePref,Formula_Score,Formula_Prob,Decision_Charge,Final_Behavior\n")

    @staticmethod
    def _compute_service_metrics(total_demand, real_reward, rel_demand):
        total_demand = float(total_demand)
        served_orders = float(np.sum(rel_demand))
        if total_demand > 0:
            service_rate = served_orders / total_demand
            service_reward_density = float(np.sum(real_reward)) / total_demand
        else:
            service_rate = 1.0
            service_reward_density = 0.0
        return served_orders, service_rate, service_reward_density

        #self.vehicle_logs.append(log)
        
        '''
        self.slot_status = {
            0: [False, True, False, True],  # zone 0 鐨勭0~3涓溅浣嶇姸鎬侊紙True 琛ㄧず宸插崰鐢級
            1: [False, False, False, False],
            ...
        }#0727鍏呯數妗?
        '''

        # self.summ=0
        # self.odin=[0]*ZONE_NUM
        # self.odout=[0]*ZONE_NUM
        # self.training_set_number = training_set_number #璁粌闆?

        # 鍐欏叆鍒濆OD淇℃伅锛岀洰鍓嶈繖涓唬鐮佸啓鐨勬瘮杈冪儌锛岃姹俤emand.csv鐨勬牸寮忔瘮杈冪璋?
        # 鎴戝彲浠ュぇ姒傜粰浣犺В閲婁綘涓嶆噦鐨勫摢涓€琛岋紝浣嗘槸鎴戝缓璁綘閲嶅啓
    
    def print_vehicle_status(self, t):
        print(f"\n=== 杞﹁締鐘舵€侊紙鏃堕棿鐗?{t}锛?==")
        for vid in range(self.CAR_NUM):
            zone = self.car[vid][0]
            soc = self.car[vid][1]
            remain = self.car[vid][2]
            
            # 鍒ゆ柇杞﹁締鐘舵€?
            if remain > 0:
                state = f"鎵ц浠诲姟 (鍓╀綑 {remain} )"
            else:
                state = "绌洪棽"
            
            print(f"杞﹁締 {vid:2d} | 鍖哄煙={zone} | SOC={soc:.2f} | 鐘舵€?{state}")
        print("====================================\n")
    
    def log_vehicle_state(t, vehicles, filename="vehicle_log.csv"):
        with open(filename, "a") as f:
            for v in vehicles:
                f.write(f"{t},{v.id},{v.origin},{v.destination},{v.soc:.2f},"
                        f"{v.current_zone},{v.target_zone},{v.distance_to_target:.2f}\n")
    
    def write_od_left(self, new):
        if self.training:  # 鏍规嵁 self.training 鐨勬爣蹇楀垽鏂綋鍓嶆ā寮忔槸璁粌妯″紡杩樻槸娴嬭瘯妯″紡锛岀劧鍚庨€夋嫨涓嶅悓鐨勬枃浠舵潵鍔犺浇 OD锛圤rigin-Destination锛夐渶姹傛暟鎹紝骞惰绠楁瘡涓尯鍩熺殑 OD 娴侀噺锛堟祦鍏ュ拰娴佸嚭閲忥級
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
                        od_in.append(np.array([sum(row[:, ins]) for row in df_od]))  # 瀛樿繖涓?
                        od_out.append(np.array([sum(row[ins, :]) for row in df_od]))  # 瀛樿繖涓?
                    self.mem_od_in.append(od_in)
                    self.mem_od_out.append(od_out)

            a = random.randint(0, training_set_number - 1)
            OD_demand = self.mem_od_demand[a]
            df_od = np.array(OD_demand)
            od_in = self.mem_od_in[a]
            od_out = self.mem_od_out[a]
        else:
            # 鐩墠榛樿鍙湁涓€涓祴璇曞満鏅?
            OD_demand = pd.read_csv(path + 'od_demand_test.csv')
            df_od = np.array([[[0] * ZONE_NUM] * ZONE_NUM] * (OD_demand.shape[1] - 2))
            for i in range(OD_demand.shape[0]):
                for j in range(2, OD_demand.shape[1]):
                    df_od[j - 2][OD_demand.iloc[i]['O']][OD_demand.iloc[i]['D']] = OD_demand.iloc[i][j]
            df_od = np.array(df_od)
            od_in = []
            od_out = []
            for ins in range(ZONE_NUM):
                od_in.append(np.array([sum(row[:, ins]) for row in df_od]))  # 瀛樿繖涓?
                od_out.append(np.array([sum(row[ins, :]) for row in df_od]))  # 瀛樿繖涓?

        self.od_demand_left = np.array(df_od)
        self.od_in = np.array(od_in)
        self.od_out = np.array(od_out)

    def update_od_left(self):
        # 鏇存柊褰撳墠OD
        self.od_demand_left = self.od_demand_left[1:, :, :]

    def write_od_now(self):
        # 鑾峰彇褰撳墠OD
        self.od_demand_now = copy.deepcopy(self.od_demand_left[:TIME_SLICE, :, :])
    
    def write_ini_cap(self, new):
        """
        鍒濆鍖栧仠杞﹀閲忋€佽溅杈?self.car銆佸厖鐢垫々 charging_stations锛?
        骞惰嚜鍔ㄥ垎閰嶅垵濮嬭溅杈嗗埌鍖哄煙涓庡厖鐢垫々銆?
        """
        
        # ======================
        # 1) 鍒濆鍖?available
        # ======================
        if new:
            available = pd.read_csv(path + 'capacity.csv')
            
            # available.iloc[0] = 姣忎釜鍖哄煙鍒濆杞﹁締鏁伴噺
            # available.iloc[1] = 姣忎釜鍖哄煙鍙敤鍋滆溅浣嶆暟閲忥紙鎬昏溅浣?- 鍒濆杞﹁締锛?
            for i in range(ZONE_NUM):
                # =========2026.2.20淇敼锛屼慨澶峱andas閾惧紡璧嬪€?=========
                available.loc[1, str(i)] = available.loc[1, str(i)] - available.loc[0, str(i)]
                # ======淇敼缁撴潫
            
            self.available = np.array(available)
            self.mem_available = copy.deepcopy(self.available)
        
        else:
            # 涓嶆槸鏂?episode锛屽氨鎭㈠瀛樻。
            self.available = copy.deepcopy(self.mem_available)
        
        # 鍒濆鍖哄煙杞﹁締鏁伴噺锛堢敤浜庢瀯閫?self.car锛?
        zone_init_car = self.available[0]  # 绗竴琛屾槸鍒濆杞﹁締鏁伴噺
        total_cars = int(sum(zone_init_car))
        
        # ======================
        # 2) 鍒濆鍖栬溅杈?self.car
        # car[i] = [zone, SOC, remain_dist, charger_id]
        # ======================
        self.car = []
        BAT = BATTERY_CAPACITY
        
        for zone_id in range(ZONE_NUM):
            count = int(zone_init_car[zone_id])
            for _ in range(count):
                self.car.append([zone_id, BAT, 0, -1])  # 鍒濆鏃?charger_id
        
        # 杞垚 numpy 鏁扮粍锛堜究浜庝綘鍚庣画浣跨敤锛?
        self.car = np.array(self.car, dtype=int)
        
        # ======================
        # 3) 鍒濆鍖?charging_stations锛堟瀯寤哄厖鐢垫々缁撴瀯锛?
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
        # 4) 鑷姩璁╁尯鍩熼噷閮ㄥ垎杞﹁締鍗犵敤鍏呯數妗?
        #
        # 鎸夌収鍖哄煙杞﹁締鏁伴噺 鈫?灏藉彲鑳戒娇鐢?charger
        #
        # 淇濊瘉 car 鈫?station 鍙屽悜涓€鑷?
        # ======================

        # =========2026.2.13淇敼锛屼慨澶嶅垵濮嬪寲鍏呯數妗╁崰鐢ㄩ棶棰?=========
        # 闂锛氬垵濮嬪寲鏃惰溅杈嗗崰鐢ㄥ厖鐢垫々锛屼絾宸茬粡婊＄數锛屽鑷村厖鐢垫々琚暱鏈熷崰鐢?
        # 瑙ｅ喅锛氭墍鏈夎溅杈嗗垵濮嬫椂閮戒笉鍗犵敤鍏呯數妗╋紝璁㏑L鑷富鍐冲畾浣曟椂鍏呯數
        # car_index = 0
        #
        # for zone_id in range(ZONE_NUM):
        #     zone_car_cnt = int(zone_init_car[zone_id])
        #     chargers = self.charging_stations[zone_id]
        #     num_chargers = len(chargers)
        #
        #     # 鑳戒娇鐢ㄧ殑鍏呯數妗╂暟閲?
        #     use = min(zone_car_cnt, num_chargers)
        #
        #     # 鈶?鍓?use 涓溅鍗犵敤鍏呯數妗?
        #     for j in range(use):
        #         charger = chargers[j]
        #         charger["occupied_by"] = car_index
        #         self.car[car_index][3] = charger["charger_id"]
        #         car_index += 1
        #
        #     # 鈶?鍓╀綑杞﹁締鍙仠鍦ㄥ仠杞︿綅锛屼笉鍗犳々
        #     for j in range(zone_car_cnt - use):
        #         self.car[car_index][3] = -1
        #         car_index += 1
        # ======淇敼缁撴潫
        
        # ======================
        # 瀹屾垚鍒濆鍖?
        # ======================
       # print(">>> 鍒濆鍖栧畬鎴? self.car 涓?charging_stations 宸插悓姝ャ€?)
    
    # SOC鍏呯數鍑芥暟

    def f(self, t):  # 鍏呯數鍑芥暟
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
           璁＄畻杞﹁締鍦ㄧ壒瀹氱數閲?battery 鍜屽凡鍏呯數鏃堕棿 minute 涓嬬殑鍙厖鍏ョ數閲忥紙鍗曚綅锛氱數閲忓€硷級

           鍙傛暟
           ----
           battery : float
               褰撳墠鐢垫睜鍓╀綑鐢甸噺锛堝崟浣嶄笌 BATTERY_CAPACITY 涓€鑷达紝濡?Wh 鎴?kWh锛?

           minute : float
               褰撳墠宸插厖鐢电殑鏃堕棿锛堝崟浣嶏細鍒嗛挓锛?

           杩斿洖
           ----
           delta_battery : float
               鏈鍏呯數鍙幏寰楃殑鐢甸噺锛堝崟浣嶄笌 battery 涓€鑷达級

           鍑芥暟閫昏緫
           -------
           - 浣跨敤 self.g() 瀵瑰綋鍓嶇數閲忓綊涓€鍖栧悗鐨勫€艰繘琛岄潪绾挎€у彉鎹紝寰楀埌寤惰繜鍥犲瓙
           - 璁＄畻瀹為檯鐨勨€滄湁鏁堝厖鐢垫椂闂粹€?real_min
           - 鑻ュ厖鐢垫椂闂磋秴鍑轰笂闄?107 鍒嗛挓锛堝彲鑳芥槸绯荤粺绾﹀畾鐨勬渶澶ф椂闂寸獥锛夛紝鍒欒涓虹洿鎺ュ厖婊?
           - 鍚﹀垯鏍规嵁 self.f() 鐨勯潪绾挎€у嚱鏁颁及绠楀厖鐢垫晥鐜囨洸绾匡紝寰楀埌瀹為檯鍙厖鐢甸噺
           """
        
        max_capacity = BATTERY_CAPACITY  # 鐢垫睜鏈€澶у閲?
        real_min = minute + self.g(battery / max_capacity)
        if real_min > 107:
            # 鑻ュ疄闄呭厖鐢靛垎閽熸暟瓒呰繃鏌愪釜闃堝€硷紙107鍒嗛挓锛夛紝鐩存帴杩斿洖鍏呮弧鎵€闇€鐢甸噺 +1
            # 鍔? 鍙兘鏄伩鍏嶆诞鐐硅宸娇寰楀€煎皬浜?max_capacity
            return max_capacity - battery + 1
        else:
            # 浣跨敤 f() 鍑芥暟鑾峰彇 real_min 瀵瑰簲鐨勨€滅疮璁″厖鐢垫瘮渚嬧€濓紝鍐嶅噺鍘诲綋鍓嶆瘮渚?
            # 寰楀埌鏈鍙互鍏呭叆鐨勭數閲?
            return (self.f(real_min) - battery / max_capacity) * max_capacity
    
    def get_micro_state(self):
        # 1) OD in/out锛岃鍒?0..OD_MAX-1锛屽苟杞?int
        # 淇锛氱‘淇漞pisode_timestep涓嶈秴鍑簅d_in/od_out鐨勮寖鍥?
        timestep = min(self.episode_timestep, self.od_in.shape[1] - TIME_SLICE)
        timestep = max(0, timestep)  # 纭繚涓嶄负璐熸暟

        od_in2 = np.array(self.od_in[:, timestep:timestep + TIME_SLICE]).T
        od_out2 = np.array(self.od_out[:, timestep:timestep + TIME_SLICE]).T
        od_part = np.concatenate((od_in2, od_out2), axis=1).ravel()
        od_part = np.clip(od_part, 0, OD_MAX - 1).astype(np.int64)
        
        # 2) 鍏呯數鐘舵€?(绗?娈?锛氭妸鍗犵敤鐜囨槧灏勬垚 0..CHARGER_STATES-1
        charger_states = []
        for zone in range(ZONE_NUM):
            # 鍗犵敤鐜?-> 鐘舵€佹。浣?
            ratio = self.available[0][zone] / (self.mem_available[0][zone] + 1e-5)
            val = int(ratio * CHARGER_STATES)
            val = max(0, min(CHARGER_STATES - 1, val))
            charger_states.extend([val] * TIME_SLICE)
        charger_states = np.array(charger_states, dtype=np.int64)
        
        # =========2026.2.15淇敼锛岀姸鎬佺┖闂翠紭鍖栵細绉婚櫎num_chargers鐨凾IME_SLICE閲嶅==========
        # 3) 鍏呯數妗╂暟閲?(绗?娈?锛氭瘡鍖鸿閰嶇殑妗╂暟锛宑lip 鍒?0..NUM_CHARGERS-1
        # 浼樺寲锛氱Щ闄IME_SLICE閲嶅锛屽洜涓哄厖鐢垫々鏁伴噺鏄潤鎬侀厤缃?
        num_chargers_feat = []
        for zone in range(ZONE_NUM):
            installed = 0
            if zone < len(CHARGER_PER_ZONE) and isinstance(CHARGER_PER_ZONE[zone], (list, tuple)):
                installed = len(CHARGER_PER_ZONE[zone])
            # 鑻ユ兂鏀?鍙敤鍏呯數浣嶆暟"锛屽彲鍦ㄦ鐢ㄥ崰鐢ㄤ俊鎭绠楋紱杩欓噷鍏堟斁瑁呴厤鏁伴噺
            if NUM_CHARGERS > 0:
                installed = max(0, min(NUM_CHARGERS - 1, installed))
            else:
                installed = 0
            num_chargers_feat.append(installed)  # 绉婚櫎TIME_SLICE閲嶅
        num_chargers_feat = np.array(num_chargers_feat, dtype=np.int64)
        # ======淇敼缁撴潫

        # =========2026.2.6淇敼锛岀姸鎬佺┖闂村寮猴細娣诲姞鍖哄煙骞冲潎SOC==========
        # 4) 鍖哄煙骞冲潎 SOC (鏂板)
        avg_soc_per_zone = []
        for zone in range(ZONE_NUM):
            # 璁＄畻璇ュ尯鍩熸墍鏈夐潤姝㈣溅杈嗙殑骞冲潎 SOC
            zone_vehicles = [self.car[i][1] for i in range(len(self.car))
                             if self.car[i][0] == zone and self.car[i][2] == 0]
            if zone_vehicles:
                avg_soc = np.mean(zone_vehicles) / BATTERY_CAPACITY  # 褰掍竴鍖栧埌 0-1
            else:
                avg_soc = 0.5  # 榛樿鍊硷紙鍖哄煙鏃犻潤姝㈣溅杈嗘椂锛?

            # 绂绘暎鍖栦负 10 妗?(0-9)
            soc_discrete = int(np.clip(avg_soc * 10, 0, 9))
            avg_soc_per_zone.append(soc_discrete)

        avg_soc_per_zone = np.array(avg_soc_per_zone, dtype=np.int64)
        # ======淇敼缁撴潫

        # =========2026.2.17淇敼锛孮uick Improvement 3锛氭坊鍔犺溅杈嗗垎甯冨埌鐘舵€佺┖闂?=========
        # 4) 姣忎釜鍖哄煙鐨勮溅杈嗘暟閲忥紙鏂板锛?
        # 缁熻姣忎釜鍖哄煙涓彲鐢ㄨ溅杈嗘暟閲忥紙涓嶅湪鏈嶅姟涓€佷笉鍦ㄥ厖鐢典腑锛?
        vehicle_counts_per_zone = []
        for zone in range(ZONE_NUM):
            count = sum(1 for i in range(len(self.car))
                       if self.car[i][0] == zone and self.car[i][2] <= 0 and self.car[i][3] == -1)
            # 绂绘暎鍖栦负 20 妗?[0-19]
            # 0-9: 鐩存帴鏄犲皠
            # 10+: 姣?杈嗚溅涓€妗?(10-14鈫?0, 15-19鈫?1, ..., 190+鈫?9)
            if count <= 9:
                count_discrete = count
            else:
                count_discrete = min(19, 9 + (count - 9) // 5)
            vehicle_counts_per_zone.append(count_discrete)
        vehicle_counts_per_zone = np.array(vehicle_counts_per_zone, dtype=np.int64)
        # ======淇敼缁撴潫

        # =========2026.2.13淇敼锛岀姸鎬佺┖闂村寮猴細娣诲姞鏃堕棿姝ヤ俊鎭?=========
        # =========2026.2.15淇敼锛岀姸鎬佺┖闂翠紭鍖栵細绉婚櫎remaining_timesteps==========
        # 5) 鏃堕棿姝ョ壒寰侊紙鏂板锛?
        # 淇锛氱‘淇濇椂闂存鍊煎湪鏈夋晥鑼冨洿[0, TIME_NUM-1]鍐?
        # 浼樺寲锛氱Щ闄emaining_timesteps锛屽洜涓哄畠涓巆urrent_timestep绾挎€х浉鍏?
        current_timestep = np.array([min(self.episode_timestep, TIME_NUM - 1)], dtype=np.int64)  # 褰撳墠鏃堕棿姝?(0-11)
        # ======淇敼缁撴潫

        # =========2026.2.19淇敼锛孲IP鏁版嵁闆嗭細鏇存柊鐘舵€佹嫾鎺ワ紙v5锛?51缁达級==========
        # 6) 鎷兼帴鎴?[OD, OD, CHARGER_STATES, AVG_SOC, VEHICLE_COUNTS, TIMESTEP]
        # 缁村害锛?*TIME_SLICE*ZONE_NUM + ZONE_NUM + ZONE_NUM + 1
        #     = 3*1*50 + 50 + 50 + 1 = 251
        charger_states_array = np.array(charger_states, dtype=np.int64)
        state = np.concatenate([od_part, charger_states_array, avg_soc_per_zone, vehicle_counts_per_zone, current_timestep])
        # 淇濋櫓鏂█锛氶暱搴﹀繀椤绘弧瓒?3 * ZONE_NUM * TIME_SLICE + 2 * ZONE_NUM + 1
        expected_dim = self.expected_state_dim
        assert state.shape[0] == expected_dim, \
            f"state dim {state.shape[0]} != expected_state_dim ({expected_dim})"
        # ======淇敼缁撴潫
        return state
        
        # return np.concatenate([od_part, np.array(charger_states)])

    @property
    def state(self):
        return self.get_micro_state()

    # =========2026.2.12淇敼锛屾柟妗團5锛氱粺涓€鍙皟搴﹁溅杈嗗彛寰?=========
    def _compute_dispatchable_counts(self):
        """Recompute dispatchable idle vehicles by zone."""
        counts = np.zeros(self.ZONE_NUM, dtype=int)
        for i in range(len(self.car)):
            if self.car[i][2] == 0 and self.car[i][3] == -1:
                zone = int(self.car[i][0])
                if 0 <= zone < self.ZONE_NUM:
                    counts[zone] += 1
        return counts

    def _sync_dispatchable_available(self):
        """Synchronize available[0] with dispatchable counts."""
        self.available[0] = self._compute_dispatchable_counts().astype(self.available[0].dtype, copy=False)
    # ======修复结束

    def _sigmoid(self, x):
        """Standard sigmoid activation."""
        return 1.0 / (1.0 + math.exp(-x))

    def _cancel_bidirectional_dispatch(self, matrix):
        """Cancel bidirectional dispatch flows."""
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

    def _get_future_out_demand(self, step_offset=1):
        """Return future outgoing demand vector."""
        idx = int(self.episode_timestep + step_offset)
        if self.od_out.size == 0 or idx < 0 or idx >= self.od_out.shape[1]:
            return np.zeros(self.ZONE_NUM, dtype=float)
        return np.asarray(self.od_out[:, idx], dtype=float)


    
    def handle_vehicle_charging(self, vehicle_id, target_zone):
        """
        鏈嚭鍙戠殑杞﹁締 鈫?鐢?check_idle_vehicles() 鍐冲畾鏄惁鍘诲厖鐢碉紙涓诲姩鍐崇瓥锛?
        鍒氬埌杈剧洰鐨勫湴鐨勮溅杈?鈫?鐢?handle_vehicle_charging() 鍐冲畾涓嬩竴姝ユ槸鍚﹀幓鍏呯數锛堣鍔ㄥ喅绛栵級
        
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
                # === 銆愭柊澧炪€?杩欐槸涓€涓幓鍏呯數鐨勫姩浣滐紝蹇呴』瑙勫垝鍏呯數浣?===
                self.assign_parking_slot(vehicle_id, is_charging_order=True)
                # print(
                #     f"[Charging] 杞﹁締{vehicle_id} 鍒拌揪鍖簕current_zone} 鈫?鍓嶅線鍏呯數鍖?{target_zone}, 璺濈={self.car[vehicle_id][2]}")
                
            
            else:
                nearest_zone, _ = self.find_nearest_charger(current_zone)
                return nearest_zone
        
        #2
        elif has_space and has_charger:
            self.car[vehicle_id][0] = target_zone
            # === 銆愭柊澧炪€?鏃㈢劧鍐冲畾鍦ㄦ湰鍦板厖鐢碉紝绔嬪埢瑙勫垝/鍗犵敤鍏呯數浣?
            self.assign_parking_slot(vehicle_id, is_charging_order=True)
        
        else:
            for alt_zone in range(ZONE_NUM):
                if (self.available[0][alt_zone] < self.mem_available[0][alt_zone] and
                        alt_zone != target_zone):
                    return alt_zone
        
        return target_zone
    
    def find_nearest_charger(self, current_zone):
        """
        鍦ㄥ叏缃戠粶涓鎵捐窛绂?current_zone 鏈€杩戠殑銆愮┖闂插厖鐢垫々銆戙€?

        杩斿洖锛?
        - (zone_id, charger_id)锛氭壘鍒板彲鐢ㄥ厖鐢垫々
        - (-1, -1)锛氭病鏈変换浣曞彲鐢ㄥ厖鐢垫々
    
        娉ㄦ剰锛?
        - 杩欓噷鍙槸鈥滄煡璇⑩€?
        - 涓嶄慨鏀?self.car
        - 涓嶄慨鏀?occupied_by
    """
        min_dist = float('inf')
        best_zone = -1
        best_charger_id = -1
        
        # 閬嶅巻鎵€鏈夊尯鍩?
        for zone_id, chargers in self.charging_stations.items():
            
            # 璺宠繃娌℃湁鍏呯數妗╃殑鍖哄煙
            if not chargers:
                continue
            
            # 閬嶅巻璇ュ尯鍩熺殑姣忎竴涓厖鐢垫々
            for ch in chargers:
                # 鍙€冭檻绌洪棽鐨勫厖鐢垫々
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
            # =========2026.2.20淇敼锛屼慨澶岰HARGER_PER_ZONE瓒婄晫==========
            if zone >= len(CHARGER_PER_ZONE) or len(CHARGER_PER_ZONE[zone]) == 0:
                continue
            # ======淇敼缁撴潫
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
    

    def release_charger(self, zone_id, charger_id):
        """Release charger occupancy record for a vehicle leaving the charger."""
        if zone_id in self.charging_stations:
            for station in self.charging_stations[zone_id]:
                if station["charger_id"] == charger_id:
                    if station["occupied_by"] is not None:
                        station["occupied_by"] = None
                    break

    def print_charging_stations(self):
        """Debug helper kept for reset() compatibility."""
        return None

    def check_idle_vehicles(self, charge_pref, charging_actions=None, epoch=0):
        """Patched docstring."""
        
        # 鎵撳嵃鏁翠綋鏍囬锛屾柟渚垮湪鎺у埗鍙颁腑鍖哄垎涓嶅悓鏃堕棿鐐圭殑杈撳嚭
        #print("\n===== Charging Stations Status =====")
        
        # 閬嶅巻姣忎竴涓尯鍩燂紙zone锛?
        # zone_id: 鍖哄煙缂栧彿
        # chargers: 璇ュ尯鍩熷唴鐨勫厖鐢垫々鍒楄〃
        # for zone_id, chargers in self.charging_stations.items():
        #
        #     # 鎵撳嵃褰撳墠鍖哄煙缂栧彿
        #     print(f"Zone {zone_id}:")
        #
        #     # 濡傛灉璇ュ尯鍩熸病鏈変换浣曞厖鐢垫々锛屾槑纭彁绀?
        #     if len(chargers) == 0:
        #         print("  (No chargers)")
            
            # 閬嶅巻璇ュ尯鍩熷唴鐨勬瘡涓€涓厖鐢垫々
           # for ch in chargers:
                # ch 鏄竴涓瓧鍏革紝缁撴瀯绫讳技锛?
                # {
                #   "charger_id": int,
                #   "type": "fast" / "slow",
                #   "occupied_by": None 鎴?car_id,
                #   "power": 60
                # }
                
                # 鎵撳嵃鍗曚釜鍏呯數妗╃殑璇︾粏鐘舵€?
                # print(
                #     f"  Charger {ch['charger_id']} | "  # 鍏呯數妗╃紪鍙凤紙涓?ZONES 涓繚鎸佷竴鑷达級
                #     f"type={ch['type']} | "  # 鍏呯數妗╃被鍨嬶紙鐩墠鍥哄畾 fast锛?
                #     f"occupied_by={ch['occupied_by']} | "  # 褰撳墠鍗犵敤璇ユ々鐨勮溅杈?ID锛圢one 琛ㄧず绌洪棽锛?
                #     f"power={ch['power']}kW"  # 鍏呯數鍔熺巼锛堝崟浣?kW锛?
                # )
        
        # 鎵撳嵃缁撳熬鍒嗛殧绗︼紝鎻愬崌鍙鎬?
        # print("===================================\n")
    
    def print_charging_stations(self):
        """Debug helper kept for reset() compatibility."""
        return None

    def assign_parking_slot(self, car_id, is_charging_order):
        """
        涓鸿溅杈嗚鍒掔洰鏍囧尯鍩熷唴鐨勫仠杞︿綅绫诲瀷锛堜粎瑙勫垝锛屼笉鍙戠敓鐗╃悊鍗犵敤锛?

        鍙傛暟
        ----
        car_id : int
            杞﹁締缂栧彿
        is_charging_order : bool
            True  -> 鍏呯數璁㈠崟锛屼紭鍏堝垎閰嶅厖鐢垫々
            False -> OD / 璋冨害璁㈠崟锛屼紭鍏堟櫘閫氬仠杞︿綅锛坈harger_id = -1锛?

        瑙勫垯
        ----
        - 浠呭湪銆岃溅杈嗗凡缁忕‘瀹氱洰鏍囧尯鍩燂紝浣嗕粛鍦ㄨ矾涓婏紙remain > 0锛夈€嶆椂璋冪敤
        - 涓嶄慨鏀?charging_stations[*]['occupied_by']
        - charger_id 鍙槸鈥滆鍒掑仠杞︿綅缂栧彿鈥?
        
        -------260107 浜岀紪
        淇鍚庣殑瑙勫垝閫昏緫锛氶槻姝㈤潪鍏呯數杞﹁締鐢变簬杞︿綅涓嶈冻鑰屼骇鐢熲€滃菇鐏靛崰浣嶁€?
        """
        
        # ===============================
        # 1. 璇诲彇杞﹁締鐩爣鍖哄煙
        # ===============================
        target_zone = int(self.car[car_id][0])
        
        # 鑾峰彇璇ュ尯鍩熺殑鍏呯數妗╁垪琛?
        chargers = self.charging_stations.get(target_zone, [])
        num_chargers = len(chargers)
        
        # 鑾峰彇鍖哄煙鎬诲閲忓拰褰撳墠鍦ㄥ尯鍩熷唴鐨勮溅杈嗘暟锛堣繎浼艰绠楋級
        # 娉ㄦ剰锛氳繖閲岄渶瑕佷竴涓噯纭殑鏂规硶鐭ラ亾鏈夊灏戣溅宸茬粡鍦ㄨ鍖哄煙涓旀槸鏅€氬仠杞?
        # 绠€鍖栧鐞嗭細鎴戜滑灏濊瘯鍒嗛厤銆?
        
        # ===============================
        # 閫昏緫鍒嗘敮 A锛氳繖鏄竴涓厖鐢佃鍗?
        # ===============================
        if is_charging_order:
            if num_chargers == 0:
                self.car[car_id][3] = -1
                return
            
            # 浼樺厛鎵剧┖闂叉々
            for ch in chargers:
                if ch["occupied_by"] is None:
                    self.car[car_id][3] = ch["charger_id"]
                    return
            
            # 鍏ㄦ弧浜嗭紝鍙兘鎺掗槦鎴栧仠鏅€氫綅
            self.car[car_id][3] = -1
            return
        
        # ===============================
        # 閫昏緫鍒嗘敮 B锛氳繖鏄竴涓櫘閫氳鍗?(OD/璋冨害)
        # ===============================
        else:
            # 1. 鍏堥粯璁ょ粰鏅€氫綅
            self.car[car_id][3] = -1
            
            # 2. 妫€鏌ラ殣鎮ｏ細鏄惁鏅€氫綅宸叉弧锛屽繀椤诲崰鐢ㄥ厖鐢垫々锛?
            # 璁＄畻璇ュ尯鍩熺洰鍓嶅凡缁忚鍗犵敤鐨勫厖鐢垫々鏁伴噺
            occupied_chargers_count = sum(1 for ch in chargers if ch["occupied_by"] is not None)
            
            # 鑾峰彇璇ュ尯鍩熷綋鍓嶆€昏溅鏁?(available[0] 鏄尯鍩熷綋鍓嶈溅鏁?
            current_total_cars = self.available[0][target_zone]
            total_capacity = self.mem_available[0][target_zone]  # 鍋囪 mem_available 瀛樼殑鏄€诲閲?
            
            # 浼扮畻鍓╀綑鐨勬櫘閫氳溅浣?= (鎬诲閲?- 鍏呯數妗╂暟) - (褰撳墠鎬昏溅鏁?- 褰撳墠鍗犳々杞︽暟)
            # 杩欎釜璁＄畻姣旇緝澶嶆潅锛屽洜涓?current_total_cars 鍖呭惈浜嗗嵆灏嗗埌杈剧殑杞﹀悧锛?
            #
            # 馃挕 鏇寸ǔ鍋ョ殑绛栫暐锛?
            # 濡傛灉鏄櫘閫氳鍗曪紝鎴戜滑灏介噺涓嶅垎閰?charger_id銆?
            # 浣嗘槸锛屽鏋滅郴缁熷彂鐜拌鍖哄煙绌洪棽浣嶇疆鏋佸皯锛屼笖绌洪棽浣嶇疆 < 绌洪棽鍏呯數妗╂暟
            # 杩欐剰鍛崇潃鍓╀綑鐨勭┖浣嶅叏閮芥槸鍏呯數妗╋紒
            
            free_spots_total = self.available[1][target_zone]  # 鍓╀綑鎬荤┖浣?
            free_chargers = sum(1 for ch in chargers if ch["occupied_by"] is None)
            
            # 濡傛灉 鍓╀綑鎬荤┖浣?<= 鍓╀綑鍏呯數妗╂暟,涔熷氨鏄墿浣欐櫘閫氬仠杞︿綅 鈮?0锛岃鏄庢櫘閫氳溅浣嶅凡缁忔病浜嗭紒
            # 杩欒締杞﹀繀椤诲幓鍗犱竴涓厖鐢垫々锛屽惁鍒欏畠灏卞彉鎴愪簡鈥滃菇鐏佃溅鈥?
            if free_spots_total <= free_chargers and free_chargers > 0:
                # 蹇呴』鍗犵敤涓€涓厖鐢垫々锛堝嵆鐕冩补杞﹀崰浣嶏級
                for ch in chargers:
                    if ch["occupied_by"] is None:
                        self.car[car_id][3] = ch["charger_id"]
                        # print(f"璀﹀憡锛氬尯鍩焮target_zone}鏅€氳溅浣嶅凡婊★紝杞﹁締{car_id}琚揩鍗犵敤鍏呯數妗﹞ch['charger_id']}")
                        return
            
            # 鍚﹀垯锛岃繕鏄淮鎸?-1
            self.car[car_id][3] = -1
            return
    
    def step(self, action):

        charge_pref = action[2 * ZONE_NUM: 3 * ZONE_NUM]
        # print(f"\n--- 鏃堕棿鐗?{self.episode_timestep} ---")#20251201
        # print("\n====== 鍔ㄤ綔鍓?self.car ======")
        # for i in range(self.CAR_NUM):
        #     print(f"杞﹁締 {i}: zone={self.car[i][0]}, SOC={self.car[i][1]}, remain={self.car[i][2]}, chargerid={self.car[i][3]}")
        # print("================================\n")
        t = self.episode_timestep

        # Reset logs for current step
        self.current_step_logs = {}

        # =========2026.2.4淇敼锛岃褰曞厖鐢靛姩浣?=========
        # 璁板綍鏈椂闂存鐨勫厖鐢靛姩浣滐紙zone_id: count锛?
        charging_actions = {}
        # ======淇敼缁撴潫

        #self.print_vehicle_status(t)

        # 瀹炵幇涓€姝ョ殑閫昏緫
        e = 1  # 涓€涓猼rick

        action = copy.deepcopy(action)
        rate = 0.6
        reward = [[0] * ZONE_NUM for _ in range(TIME_NUM)]# 鍒濆鍖栦竴涓簩缁村垪琛紝瀛樺偍濂栧姳鍊笺€傚ぇ灏忥細TIME_NUM x ZONE_NUM
        real_reward = np.array([0] * ZONE_NUM)
        j = 0
        pre_avail = copy.deepcopy(self.available[1])
        # 鍏堢Щ鍔紝鍐嶇湅鍔ㄤ綔锛屽啀婊¤冻闇€姹?

        # 绉诲姩

        con = self.cong[self.episode_timestep - 1][0]  # 閬撹矾鎷ュ牭鍙傛暟

        i = 0
        while i < len(self.car):
            #self.car_mem[i][2] = max(5, self.car_mem[i][2] + 1)  # 浼樺厛绾ц绠?
            # === 鏂板浠ｇ爜寮€濮嬶細澶勭悊瀹為檯鍏呯數 ===
            # 鍒ゆ柇鏉′欢锛氳溅杈嗕笉鍦ㄨ矾涓?(remain <= 0) 涓?杞﹁締鍒嗛厤浜嗗厖鐢垫々 (charger_id != -1)
            if self.car[i][2] <= 0 and self.car[i][3] != -1:
                # 璋冪敤浣犵殑 charging 鍑芥暟鎴栬€呯畝鍗曠殑绾挎€у厖鐢靛叕寮?
                # 娉ㄦ剰锛氳繖閲岄渶瑕佺‘淇?charging 鍑芥暟杩斿洖鐨勬槸澧為噺锛屾垨鑰呬綘鐩存帴鍦ㄨ繖閲岃绠?
                # 鍋囪绠€鍗曠殑绾挎€у厖鐢碉細
                charge_speed = 1800  # 鍋囪 60kW 鍔熺巼锛屾垨鑰呮牴鎹?charging_stations[zone][id]['power'] 鑾峰彇
                added_energy = charge_speed * (TIME_LENGTH / 60)  # 杞崲涓?kWh
                
                # 鏇存柊鐢甸噺
                self.car[i][1] = min(BATTERY_CAPACITY, self.car[i][1] + added_energy)

                # =========2026.2.13淇敼锛屼紭鍖栧厖鐢甸噴鏀剧瓥鐣?=========
                # 鍘熺瓥鐣ワ細鍏呮弧鎵嶉噴鏀撅紙闇€瑕?姝ワ級
                # 鏂扮瓥鐣ワ細鍏呭埌80%灏遍噴鏀撅紙绾?姝ワ級锛屾彁楂樺厖鐢垫々鍒╃敤鐜?
                release_threshold = BATTERY_CAPACITY * 0.8  # 1600 SOC
                if self.car[i][1] >= release_threshold:
                    self.car[i][1] = min(BATTERY_CAPACITY, self.car[i][1])  # 纭繚涓嶈秴杩囦笂闄?
                    # 閲婃斁鍏呯數妗?
                    current_zone = self.car[i][0]
                    charger_id = self.car[i][3]
                    self.release_charger(current_zone, charger_id)
                    self.car[i][3] = -1  # 閲婃斁鍏呯數妗╃姸鎬?
                # ======淇敼缁撴潫

            # =========2026.2.13淇敼锛屾坊鍔燬OC娑堣€楅€昏緫==========
            # 杞﹁締绉诲姩鏃舵秷鑰楃數閲?
            if self.car[i][2] > 0:
                # 璁＄畻鏈椂闂存瀹為檯绉诲姩鐨勮窛绂?
                max_distance = car_speed_min * TIME_LENGTH * con
                actual_distance = min(self.car[i][2], max_distance)

                # SOC娑堣€楋細1 SOC per unit distance锛堜笌鏈€浼樿В涓€鑷达級
                soc_consumed = actual_distance * 1.0
                self.car[i][1] = max(0, self.car[i][1] - soc_consumed)
                self.car[i][2] -= actual_distance
                # ======淇敼缁撴潫

                # === 杞﹁締鍒拌揪鏃跺埢 (Arrival) ===
                if self.car[i][2] <= 0:
                    self.car[i][2] = 0
                    current_zone = self.car[i][0]
                    #self.available[0][self.car[i][0]] += 1
                    
                    # 澶勭悊鍏呯數閫昏緫锛岃繖涓€part鏈変粈涔堢敤锛燂紵锛焍ug
                    target_zone1 = self.car[i][0]
                    target_zone = self.handle_vehicle_charging(i, target_zone1)
                    
                    # =================================================
                    # 銆愭柊澧炰唬鐮併€戠墿鐞嗗崰鐢ㄥ悓姝ワ細浜鸿溅鍚堜竴
                    # =================================================
                    planned_charger_id = self.car[i][3]
                    
                    # 濡傛灉杩欒締杞﹁韩涓婃湁棰勫畾鐨勫厖鐢垫々 ID (鏃犺鏄湡鍏呯數锛岃繕鏄杩崰浣?
                    if planned_charger_id != -1:
                        if current_zone in self.charging_stations:
                            for station in self.charging_stations[current_zone]:
                                if station['charger_id'] == planned_charger_id:
                                    # 鏍稿績锛氬皢鐜閲岀殑妗╂爣璁颁负琚綋鍓嶈溅杈?i)鍗犵敤
                                    station['occupied_by'] = i
                                    # print(f"杞﹁締{i} 鐗╃悊钀介攣 鍖哄煙{current_zone} 鍏呯數妗﹞planned_charger_id}")
                                    break
                    # 馃攳 鎵撳嵃鍓?0杈嗚溅褰撳墠鐢甸噺锛岀敤浜庨獙璇佹槸鍚﹀瓨鍦ㄤ綆鐢甸噺杞?
                    #print("馃攱 鍓?0杈嗚溅杈嗗綋鍓嶇殑鐢甸噺鍙椹惰窛绂伙細", [round(self.car[i][1], 1) for i in range(min(10, CAR_NUM))])

                    # =========2026.2.13淇敼锛屽厑璁稿厖鐢佃溅杈嗗弬涓庢湇鍔?=========
                    # 鍘熼€昏緫锛氬厖鐢佃溅杈嗗畬鍏ㄤ笉鍙敤锛坧lanned_charger_id != -1鏃朵笉璁″叆available锛?
                    # 鏂伴€昏緫锛氬厖鐢佃溅杈嗕粛鍙湇鍔★紝鍑忓皯鍏呯數鐨勬満浼氭垚鏈?
                    # 杩欐牱璁㏑L鏇村鏄撳涔犲厖鐢电瓥鐣?
                    # 銆愬叧閿慨鏀广€戝厖鐢佃溅杈嗕篃璁″叆available
                    self.available[0][self.car[i][0]] += 1  # 鎵€鏈夊埌杈捐溅杈嗛兘鍙敤
                    # ======淇敼缁撴潫

                    # # 鍘熸湁閫昏緫锛氬埌杈惧悗鍒ゆ柇涓嬩竴姝ワ紙娉ㄦ剰锛氳繖閫氬父鏄拡瀵瑰垰鎵嶅畬鎴愮Щ鍔ㄧ殑杞︼級
                    # #澶勭悊鍏呯數閫昏緫锛岃繖涓€part鏈変粈涔堢敤锛燂紵锛焍ug
                    # target_zone1 = self.car[i][0]
                    # target_zone = self.handle_vehicle_charging(i, target_zone1)
            
            i += 1

        # =========2026.2.12淇敼锛屾柟妗團5锛氬姩浣滆В鐮佸墠鍚屾鍙皟搴﹁溅杈嗘暟==========
        # 鐩殑锛氳皟搴﹂绠楀簲鍩轰簬杞﹁締鐪熷疄鐘舵€侊紝鑰岄潪鍘嗗彶鎵ｅ噺娈嬪€?
        self._sync_dispatchable_available()
        # ======淇敼缁撴潫

        # =========2026.2.6淇敼锛屾柟妗?B锛氬欢鍚庡厖鐢靛喅绛栵紝閬垮厤鍏呯數-璋冨害鍐茬獊==========
        # 娉ㄩ噴鎺夊師浣嶇疆鐨勫厖鐢靛喅绛栵紙鍦ㄨ皟搴﹀墠锛?
        # 鍘熷洜锛氭鏃惰皟鐢ㄤ細瀵艰嚧绌洪棽杞﹁締浼樺厛琚獹urobi璋冨害璧帮紝鍏呯數鍐崇瓥鏃舵棤杞﹀彲鍏?
        # self.check_idle_vehicles(charge_pref, charging_actions)
        # ======淇敼缁撴潫

        # 澶氬尯鍩烵D鍒嗛厤锛屼负姣忎釜鍖哄煙鐨勮溅杈嗗垎閰嶇洰鐨勫湴

        action_table_true = np.ones((ZONE_NUM, ZONE_NUM)) * 0  # 瀛樺偍鏈€缁堢殑 OD 鍒嗛厤缁撴灉
        i = 0
        j = 0
        real_act = []  # 璋冩暣鍚庣殑 OD 鍔ㄤ綔鍒嗛厤琛?

        # 瀵硅緭鍏ョ殑鍔ㄤ綔杩涜OD鍒嗛厤锛堝姩浣滃墠4浣嶄负杞﹁締鍑哄彂鍋忓ソ锛屽悗鍥涗綅涓哄埌杈惧亸濂斤級
        for i in range(0, ZONE_NUM):
            action_table = copy.deepcopy(action[ZONE_NUM:])
            action_table[i] = 0
            real_act.append(action_table)

            # 鍐茬獊鍔ㄤ綔鐨勫垹闄ゅ鐞?
            for j in range(i):
                if real_act[i][j] > real_act[j][i]:
                    real_act[j][i] = 0


        # =========2026.2.12淇敼锛屾柟妗團5锛氫娇鐢ㄧ粺涓€鍙ｅ緞鐨勫彲璋冨害棰勭畻==========
        dispatchable_available = self._compute_dispatchable_counts()
        # ======淇敼缁撴潫

        # =========2026.2.14淇敼锛屽疄楠孉/B鍔ㄤ綔瑙ｇ爜閫昏緫==========
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
        # ======淇敼缁撴潫

        for i in range(0, ZONE_NUM):
            # 鍏堝鍙彂鍑虹殑杞﹁繘琛屼竴娆￠檺鍒?
            # limit_now=int(max(0,(self.available[0][i]-self.od_out[i][self.episode_timestep]/e)*float(action[i])))
            # 涓婇潰閭ｄ釜鏄疶RE鐨?
            # =========2026.2.14淇敼锛屽疄楠孉/B鍔ㄤ綔瑙ｇ爜閫昏緫==========
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
            # ======淇敼缁撴潫

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

                # print("鍔ㄤ綔%d"%self.episode_timestep)
        # print(action)
        # print(action_table_true)
        
        #1118娉ㄩ噴
        # print(">>> 璋冪敤 solveproblem 鍓嶅昂瀵革細")
        # print("dist", len(self.dist), len(self.dist[0]))
        # #print("demand", len(self.demand), len(self.demand[0]))
        # print("avail0/1", len(self.available[0]), len(self.available[1]))
        # print("pre_avail", len(pre_avail))
        # print("real_reward", len(real_reward))
        
        #print("reward[t]", len(reward[timestep]))
        
        # for i in range(len(self.car)):
        #     # 褰撳墠杞﹁締淇℃伅
        #     current_zone = self.car[i][0]
        #     battery = self.car[i][1]
        #     distance_to_target = self.car[i][2]
        #     # 鑾峰彇鐩爣鍖哄煙锛堝姩浣滃垎閰嶅悗鐨勭洰鏍囷級
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
        
        # 鎴戣繖绡囨枃绔犵敤鐨勬槸浼樺寲鏂规硶锛屼綘鍙互涓嶇敤寮勶紝reward鍗曟櫤鑳戒綋搴旇鐢ㄤ笉涓婏紝鎴戝氨鐩存帴璁╀粬绛夊悓浜巖eal_reward
        
        # =========2026.1.30淇敼锛屾渶浼樿В姹傝В==========
        # 璋冨害鍓嶆墽琛屽弻鍚戝璋冨噣娴佹姷娑?
        action_table_true = self._cancel_bidirectional_dispatch(action_table_true)
        # ======淇敼缁撴潫

        prev_car = copy.deepcopy(self.car)  #瀵规瘮solveproblem 鍓嶅悗鐨勭姸鎬侊紝璇嗗埆鍝簺璁㈠崟鏄繖涓€杞柊浜х敓鐨?
        
        self.car, self.available, self.car_mem, reward, real_reward, rel_dispatch = gur.solveproblem(self.dist, self.available,
                                                                                            action_table_true, self.car,
                                                                                            reward, -diaodu_min,
                                                                                            self.car_mem,
                                                                                            self.episode_timestep,
                                                                                            real_reward, 0, 0,
                                                                                            pre_avail)

        # 婊¤冻闇€姹傦紙婊¤冻闇€姹傛寜鐞嗘潵璇村簲璇ラ『搴忔弧瓒筹紝鎴戣繖杈圭畝鍖栦竴涓嬩篃缁欎紭鍖栧仛浜嗭級
        demand = self.od_demand_now[0]
        
        # 20251202=== DEBUG锛氭墦鍗版湰鏃堕棿鐗囬渶姹傛儏鍐?===
        #print(f"\n[DEBUG] t={self.episode_timestep} OD 闇€姹傝鍗曪細")
        
        total_orders = 0
        for o in range(self.ZONE_NUM):
            for d in range(self.ZONE_NUM):
                if demand[o][d] > 0:
                   # print(f"  OD {o} -> {d} : {demand[o][d]} 鍗?)
                    total_orders += demand[o][d]
        
      #  print(f"  鏈椂闂寸墖璁㈠崟鎬绘暟 = {total_orders}")


        self._last_rld = np.sum(rel_dispatch)  # 淇敼

        self.car, self.available, self.car_mem, reward, real_reward, rel_demand = gur.solveproblem(self.dist, self.available,
                                                                                            demand, self.car, reward,
                                                                                            manzu_min, self.car_mem,
                                                                                            self.episode_timestep,
                                                                                            real_reward, 0, 0,
                                                                                            pre_avail)

        # =========2026.2.6淇敼锛屾柟妗?B锛氬湪璋冨害瀹屾垚鍚庢墽琛屽厖鐢靛喅绛?=========
        # 绉诲埌杩欓噷锛氫袱娆?solveproblem 涔嬪悗锛岃璋冨害鍚庡墿浣欑殑绌洪棽杞﹁締鍘诲厖鐢?
        # 浼樺娍锛氶伩鍏嶅厖鐢?璋冨害鍐茬獊锛岀‘淇濆厖鐢靛喅绛栧熀浜庤皟搴﹀悗鐨勭湡瀹炵┖闂茶溅杈?
        # =========2026.2.7淇敼锛屾柟妗圖2锛氫娇鐢╯elf.current_epoch鑰屼笉鏄弬鏁?=========
        self.check_idle_vehicles(charge_pref, charging_actions, epoch=self.current_epoch)
        self._sync_dispatchable_available()
        # ======淇敼缁撴潫

        for vid in range(len(self.car)):
            # 鐩爣鍖哄煙鍙戠敓鍙樺寲 鈫?璇存槑鏄柊璁㈠崟
            if self.car[vid][0] != prev_car[vid][0]:
                # === 銆愭柊澧炰慨澶?2026.1.7銆?杞﹁締绂诲紑锛岄噴鏀惧師鍖哄煙鐨勫厖鐢垫々 ===
                old_zone = prev_car[vid][0]
                old_charger_id = prev_car[vid][3]
                
                if old_charger_id != -1:
                    self.release_charger(old_zone, old_charger_id)
                # ===============================================
                
                # =========2026.2.12淇敼锛屾柟妗團3锛氶伩鍏嶈鐩栨湰姝ュ凡鍒嗛厤鐨勫厖鐢典换鍔?=========
                # 鑻ヨ溅杈嗗凡缁忔嫢鏈夊厖鐢垫々璁″垝锛屼笉鍐嶆寜鏅€氳鍗曡鐩栧仠杞︿綅瑙勫垝
                if self.car[vid][3] == -1:
                    self.assign_parking_slot(vid, is_charging_order=False)    # is_charging_order=False琛ㄧず琛ㄧず鍏呯數璁㈠崟 瑙勫垝璋冨害鍜宱d闇€姹傚悗鍋滆溅鐨勫叿浣撲綅缃?
                # ======淇敼缁撴潫
        
        total_demand = np.sum(demand)##0729 璁＄畻trb璁烘枃鎸囨爣鍔犲叆
        served_orders, service_rate, service_reward_density = self._compute_service_metrics(
            total_demand, real_reward, rel_demand
        )
        self._last_svc = service_rate
        self._last_srd = service_reward_density

        # =========2026.2.14??????????==========
        # ?????????????? / ????
        service_rate = served_orders / total_demand if total_demand > 0 else 1.0

        # 鏈嶅姟鐜囨儵缃氾細濡傛灉鏈嶅姟鐜囦綆浜庨槇鍊硷紝缁欎簣閫傚害鎯╃綒
        service_rate_penalty = 0
        if service_rate < 0.5:  # 鏈嶅姟鐜囦綆浜?0%鏃跺紑濮嬫儵缃?
            # 鎯╃綒鍔涘害锛?0.5 - service_rate) * 8000
            # 渚嬪锛歴ervice_rate=0鏃讹紝鎯╃綒=4000锛泂ervice_rate=0.3鏃讹紝鎯╃綒=1600
            service_rate_penalty = (0.5 - service_rate) * 8000
        # ======淇敼缁撴潫

        # ====== Debug log start ======
        #print(f"\n[Step {self.episode_timestep}] Action received:")

        # 鎵撳嵃鍑哄彂鍊惧悜
       # print("  鍑哄彂鍊惧悜:")
       # for i in range(ZONE_NUM):
       #     print(f"    鍖哄煙 {i}: {action[i]:.3f}  鍙彂杞︽暟 鈮?{int(self.available[0][i] * action[i])}")

        # 鎵撳嵃鐩殑鍦板€惧悜
       # print("  鐩殑鍦板€惧悜:")
        #for i in range(ZONE_NUM):
        #    print(f"    鍖哄煙 {i}: {action[ZONE_NUM + i]:.3f}")

        #print("  鍏呯數鍊惧悜:")
        #for i in range(ZONE_NUM):
        #    print(f"    鍖哄煙 {i}: {charge_pref[i]:.3f}")

        # 鎵撳嵃OD璋冨害鐭╅樀锛堜粠 i 鍖哄煙 鍒?j 鍖哄煙 鍙戝嚭杞﹁締鏁帮級
        #print("  璋冨害鍐崇瓥鐭╅樀锛坅ction_table_true锛?")
        #for i in range(ZONE_NUM):
         #   row = "    浠庡尯鍩?{:d} 鈫?".format(i)
         #   row += "  ".join(["{:d}".format(int(action_table_true[i][j])) for j in range(ZONE_NUM)])
         #   print(row)
        # ====== Debug log end ======
        self.episode_timestep += 1

        # 鍐欏叆鏂扮殑鐘舵€?

        self.update_od_left()

        self.write_od_now()
        done = bool(self.terminal)##0729 璁＄畻trb璁烘枃鎸囨爣鍔犲叆

        # =========2026.2.5淇敼锛屾柟妗?锛氭坊鍔犲嵆鏃跺厖鐢靛鍔?=========
        # 1. 鏈嶅姟璁㈠崟鏀剁泭锛堝凡鍦╮eal_reward涓級
        service_revenue = sum(real_reward)

        # 2. 鍏呯數鍔犲垎锛氱粺璁℃鍦ㄥ厖鐢电殑杞﹁締鏁?
        charging_bonus = 0
        charging_count = 0
        for i in range(self.CAR_NUM):
            # 鍒ゆ柇杞﹁締鏄惁鍦ㄥ厖鐢碉細charger_id != -1 涓斾笉鍦ㄧЩ鍔ㄤ腑
            if self.car[i][3] != -1 and self.car[i][2] == 0:
                charging_count += 1
        # =========2026.2.12淇敼锛屽疄楠?9锛氬ぇ骞呮彁鍗囧厖鐢靛鍔?=========
        # =========2026.2.13淇敼锛岃繘涓€姝ユ彁鍗囧厖鐢靛鍔?=========
        # =========2026.2.13淇敼锛屽畬鏁寸粍鍚堟柟妗堬細澶у箙鎻愬崌鍏呯數濂栧姳==========
        # 闂锛氬厖鐢靛鍔?000浠嶄笉瓒充互鎶垫秷鏈嶅姟鏀跺叆鐨勫惛寮曞姏
        # 瑙ｅ喅锛氭彁鍗囧埌15000锛岄厤鍚堝厑璁稿厖鐢佃溅杈嗘湇鍔★紝娑堥櫎鏈轰細鎴愭湰
        # 鍗虫椂鍏呯數濂栧姳锛氭瘡杈嗗厖鐢佃溅杈?15000锛堜粠5000鎻愬崌鍒?5000锛?
        charging_bonus = 15000 * charging_count
        # ======淇敼缁撴潫

        # 3. 浣庣數閲忔儵缃氾細鍒嗘鎯╃綒
        # =========2026.2.11淇敼锛屽疄楠?2锛氭笎杩涘紡璁粌锛圕urriculum Learning锛?=========
        # 鏍规嵁褰撳墠璁粌杞暟鍔ㄦ€佽皟鏁碨OC鎯╃綒寮哄害
        epoch = self.current_epoch if hasattr(self, 'current_epoch') else 0

        # =========2026.2.13淇敼锛屼慨姝urriculum Learning绛栫暐==========
        # 闂锛氬悗鏈熸儵缃氳繃浣庯紝瀵艰嚧绛栫暐涓嶇ǔ瀹?
        # 瑙ｅ喅锛氫繚鎸佽緝楂樼殑SOC鎯╃綒锛岀‘淇濈瓥鐣ョǔ瀹氭€?
        # 闃舵1锛?-300杞級锛氶珮鎯╃綒锛屽己鍒跺涔犲厖鐢?
        # 闃舵2锛?00-600杞級锛氫腑绛夋儵缃氾紝浼樺寲绛栫暐
        # 闃舵3锛?00+杞級锛氱ǔ瀹氭儵缃氾紝淇濇寔鎬ц兘
        if epoch < 300:
            # 楂樻儵缃氶樁娈?
            penalty_scale = 1.5
            vehicle_penalty = 750
        elif epoch < 600:
            # 涓瓑鎯╃綒闃舵锛堢嚎鎬ц“鍑忥級
            progress = (epoch - 300) / 300  # 0鍒?
            penalty_scale = 1.5 - 0.3 * progress  # 1.5琛板噺鍒?.2
            vehicle_penalty = 750 - 150 * progress  # 750琛板噺鍒?00
        else:
            # 绋冲畾鎯╃綒闃舵锛堜笉鍐嶉檷浣庯級
            penalty_scale = 1.2
            vehicle_penalty = 600
        # ======淇敼缁撴潫

        soc_penalty = 0
        low_soc_count = 0
        for i in range(self.CAR_NUM):
            soc = self.car[i][1]
            # =========2026.2.13淇敼锛屽畬鏁寸粍鍚堟柟妗堬細娣诲姞鎸囨暟绾OC鎯╃綒==========
            # 褰揝OC<400鏃讹紝鎯╃綒鍔犲€嶏紙3x锛夛紝寮哄寲浣庣數閲忓嵄鏈烘劅
            exponential_multiplier = 3.0 if soc < 400 else 1.0
            # ======淇敼缁撴潫

            if soc >= 600:
                penalty = 0
            elif soc >= 400:
                penalty = 2.0 * penalty_scale * (600 - soc)
            elif soc >= 200:
                penalty = 2.0 * penalty_scale * 200 + 10.0 * penalty_scale * (400 - soc)
                penalty *= exponential_multiplier  # 搴旂敤鎸囨暟鎯╃綒
                low_soc_count += 1
            else:
                penalty = 2.0 * penalty_scale * 200 + 10.0 * penalty_scale * 200 + 50.0 * penalty_scale * (200 - soc)
                penalty *= exponential_multiplier  # 搴旂敤鎸囨暟鎯╃綒
                low_soc_count += 1
            soc_penalty += penalty

        low_soc_vehicle_penalty = low_soc_count * vehicle_penalty
        soc_penalty += low_soc_vehicle_penalty
        # ======淇敼缁撴潫

        # =========2026.2.13淇敼锛屾坊鍔燬OC缁存寔濂栧姳锛圧eward Shaping锛?=========
        # =========2026.2.13淇敼锛屽畬鏁寸粍鍚堟柟妗堬細澧炲己SOC缁存寔濂栧姳==========
        # 鐩殑锛氬鍔变繚鎸侀珮SOC鐨勮涓猴紝鑰屼笉浠呬粎鏄儵缃氫綆SOC
        # 璁＄畻杞﹂槦骞冲潎SOC锛岀粰浜堟鍚戝鍔?
        total_soc = sum(self.car[i][1] for i in range(self.CAR_NUM))
        avg_soc = total_soc / self.CAR_NUM

        # SOC缁存寔濂栧姳锛氬钩鍧嘢OC瓒婇珮锛屽鍔辫秺澶?
        # 鐩爣锛氶紦鍔变繚鎸丼OC鍦?200浠ヤ笂锛?0%锛?
        # 浠?.5鎻愬崌鍒?.0锛屽寮洪珮SOC缁存寔鐨勬縺鍔?
        if avg_soc >= 1200:
            soc_maintain_bonus = (avg_soc - 1200) * 2.0  # 瓒呰繃1200鐨勯儴鍒嗭紝姣忕偣2.0濂栧姳锛堜粠0.5鎻愬崌锛?
        else:
            soc_maintain_bonus = 0

        # =========2026.2.13淇敼锛屽畬鏁寸粍鍚堟柟妗堬細娣诲姞鍏呯數缁存姢濂栧姳==========
        # 鐩殑锛氶澶栧鍔卞厖鐢佃涓猴紝纭繚鍏呯數绛栫暐鐨勭ǔ瀹氭€?
        # 鍏呯數缁存姢濂栧姳锛氭瘡杈嗗厖鐢佃溅杈嗛澶?3000
        # 涓庡厖鐢靛鍔?15000)鍙犲姞锛屾€昏18000/杞?
        charging_maintenance_bonus = 3000 * charging_count
        # ======淇敼缁撴潫

        # 4. 璋冨害鎴愭湰锛堝凡鍦╮eal_reward涓紝浣嗛渶瑕佸崟鐙绠楋級
        # 娉ㄦ剰锛氱涓€娆olveproblem鐨剅eal_reward鍖呭惈璋冨害鎴愭湰锛堣礋鍊硷級
        # 杩欓噷涓嶉渶瑕侀澶栧噺鍘伙紝鍥犱负宸茬粡鍦╮eal_reward涓簡

        # =========2026.2.5淇敼锛屽疄楠?锛氱Щ闄ゅ浐瀹氭儵缃?=========
        # 5. 鍥哄畾鎯╃綒锛堢Щ闄や互閬垮厤鎺╃洊鍏呯數濂栧姳淇″彿锛?
        fixed_penalty = 0  # 鍘?00鏀逛负0
        # ======淇敼缁撴潫

        # =========2026.2.12淇敼锛屽疄楠?9锛氬厖鐢靛鍔变笉缂╂斁==========
        # =========2026.2.13淇敼锛屾坊鍔燬OC缁存寔濂栧姳==========
        # =========2026.2.13淇敼锛屽畬鏁寸粍鍚堟柟妗堬細娣诲姞鍏呯數缁存姢濂栧姳==========
        # =========2026.2.14淇敼锛屾坊鍔犳湇鍔＄巼鎯╃綒==========
        # 鏂板鍔卞嚱鏁帮細鍏呯數濂栧姳鍗曠嫭璁＄畻锛屼笉闄や互10
        # 娣诲姞SOC缁存寔濂栧姳锛岄紦鍔变繚鎸侀珮SOC
        # 娣诲姞鍏呯數缁存姢濂栧姳锛岃繘涓€姝ュ己鍖栧厖鐢垫縺鍔?
        # 娣诲姞鏈嶅姟鐜囨儵缃氾紝榧撳姳鏈嶅姟鏇村璁㈠崟
        # =========2026.2.20淇敼锛屽鍔卞垎灞傦細base涓巗haping鍙垏鎹?=========
        weighted_service_rate_penalty = self.reward_weights['service_rate_penalty'] * service_rate_penalty
        base_reward = (service_revenue - soc_penalty) / 10 - fixed_penalty - weighted_service_rate_penalty
        if self.reward_mode == 'shaping':
            reward_value = (
                base_reward
                + self.reward_weights['charging_bonus'] * charging_bonus
                + self.reward_weights['charging_maintenance'] * charging_maintenance_bonus
                + self.reward_weights['soc_maintain'] * soc_maintain_bonus
            )
        else:
            reward_value = base_reward
        # ======淇敼缁撴潫
        # ======淇敼缁撴潫

        # =========2026.2.4淇敼锛屾坊鍔犲厖鐢靛姩浣滃拰璋冨害鐭╅樀鍒癷nfo==========
        parking_available_by_zone = np.zeros(self.ZONE_NUM, dtype=float)
        parking_used_by_zone = np.zeros(self.ZONE_NUM, dtype=float)
        charger_available_by_zone = np.zeros(self.ZONE_NUM, dtype=float)
        charger_used_by_zone = np.zeros(self.ZONE_NUM, dtype=float)

        for zi in range(self.ZONE_NUM):
            if self.available is not None and len(self.available) > 1 and len(self.available[1]) > zi:
                free_parking = float(self.available[1][zi])
            else:
                free_parking = 0.0

            total_parking = float(ZONES[zi].get("num_parking_slots", 0)) if zi < len(ZONES) else 0.0
            used_parking = max(0.0, total_parking - free_parking)
            parking_available_by_zone[zi] = max(0.0, free_parking)
            parking_used_by_zone[zi] = used_parking

            chargers = self.charging_stations.get(zi, []) if hasattr(self, 'charging_stations') else []
            total_chargers = float(len(chargers))
            used_chargers = float(sum(1 for ch in chargers if ch.get("occupied_by") is not None))
            free_chargers = max(0.0, total_chargers - used_chargers)
            charger_available_by_zone[zi] = free_chargers
            charger_used_by_zone[zi] = used_chargers

        info = {            ##0729 璁＄畻trb璁烘枃鎸囨爣鍔犲叆
            "rld": self._last_rld,          ##0729 璁＄畻trb璁烘枃鎸囨爣鍔犲叆
            "svc": self._last_svc,
            "srd": self._last_srd,  # 淇敼
            "dispatch_orders": rel_dispatch,  # 璋冨害璁㈠崟鐭╅樀
            "demand_orders": rel_demand,  # 闇€姹傝鍗曠煩闃?
            "demand_matrix": demand,  # ??????
            "served_orders": float(served_orders),
            "total_demand": float(total_demand),
            "charging_actions": charging_actions,  # 鍏呯數鍔ㄤ綔璁板綍
            "action_table_true": action_table_true,  # RL璋冨害鍊惧悜鐭╅樀
            "service_rate": float(service_rate),
            "reward_mode": self.reward_mode,
            "parking_available_by_zone": parking_available_by_zone,
            "parking_used_by_zone": parking_used_by_zone,
            "charger_available_by_zone": charger_available_by_zone,
            "charger_used_by_zone": charger_used_by_zone,
        }           ##0729 璁＄畻trb璁烘枃鎸囨爣鍔犲叆
        # ======淇敼缁撴潫

        if self.terminal:
            # 濡傛灉閲囩敤骞冲潎reward锛屾槸鍚﹂渶瑕佽绠楄繕鍓╀笅鐨勬补璐瑰拰绉熻垂
            # =========2026.2.5淇敼锛岀粓姝㈡椂涔熶娇鐢ㄥ嵆鏃跺鍔遍€昏緫==========
            # 璁＄畻鏂板鍔卞嚱鏁扮殑鍚勪釜閮ㄥ垎
            service_revenue = sum(real_reward)

            charging_count = 0
            for i in range(self.CAR_NUM):
                if self.car[i][3] != -1 and self.car[i][2] == 0:
                    charging_count += 1
            # =========2026.2.12淇敼锛屾柟妗團4锛氱粓姝㈡濂栧姳涓庝腑闂存瀹屽叏涓€鑷?=========
            charging_bonus = 3000 * charging_count
            # ======淇敼缁撴潫

            # =========2026.2.11淇敼锛屽疄楠?2锛氱粓姝㈡椂涔熶娇鐢ㄦ笎杩涘紡鎯╃綒==========
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
            # ======淇敼缁撴潫

            # =========2026.2.14淇敼锛岀粓姝㈡椂涔熸坊鍔犳湇鍔＄巼鎯╃綒==========
            # 璁＄畻鏈嶅姟鐜囷細瀹為檯鏈嶅姟鐨勮鍗曟暟 / 鎬婚渶姹傛暟
            served_orders = np.sum(rel_demand)  # 瀹為檯鏈嶅姟鐨勮鍗曟暟
            service_rate = served_orders / total_demand if total_demand > 0 else 1.0

            # 鏈嶅姟鐜囨儵缃氾細濡傛灉鏈嶅姟鐜囦綆浜庨槇鍊硷紝缁欎簣閫傚害鎯╃綒
            service_rate_penalty = 0
            if service_rate < 0.5:  # 鏈嶅姟鐜囦綆浜?0%鏃跺紑濮嬫儵缃?
                # 鎯╃綒鍔涘害锛?0.5 - service_rate) * 8000
                # 渚嬪锛歴ervice_rate=0鏃讹紝鎯╃綒=4000锛泂ervice_rate=0.3鏃讹紝鎯╃綒=1600
                service_rate_penalty = (0.5 - service_rate) * 8000
            # ======淇敼缁撴潫

            # =========2026.2.5淇敼锛屽疄楠?锛氱粓姝㈡椂涔熺Щ闄ゅ浐瀹氭儵缃?=========
            fixed_penalty = 0  # 鍘?00鏀逛负0
            weighted_service_rate_penalty = self.reward_weights['service_rate_penalty'] * service_rate_penalty
            base_reward = (service_revenue - soc_penalty) / 10 - fixed_penalty - weighted_service_rate_penalty
            if self.reward_mode == 'shaping':
                final_reward = base_reward + self.reward_weights['charging_bonus'] * charging_bonus
            else:
                final_reward = base_reward
            # ======淇敼缁撴潫
            # ======淇敼缁撴潫
            return self.state, float(final_reward), bool(self.terminal), info
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
            # -------------------------------- 2026.1.30娉ㄩ噴
            # print("\n====== 鍔ㄤ綔鍚?self.car ======")
            # for i in range(self.CAR_NUM):
            #     print(f"杞﹁締 {i}: zone={self.car[i][0]}, SOC={self.car[i][1]}, remain={self.car[i][2]}")
            # print("================================\n")

            # =========2026.2.2淇敼锛屼娇鐢ㄦ柊濂栧姳鍑芥暟==========
            return self.state, float(reward_value), bool(self.terminal), info
            # ======淇敼缁撴潫

    @property
    def terminal(self):
        return self.is_terminal()

    def is_terminal(self):
        return bool(self.episode_timestep >= TIME_NUM)

    def set_training_mode(self, training):
        """
        璁剧疆璁粌妯″紡
        training = 0: 娴嬭瘯妯″紡锛堜繚瀛樺埌 RL-DDPG_actions.txt锛?
        training = 1: 璁粌妯″紡锛堜笉淇濆瓨锛?
        training = 2: 鎺㈢储鏈熻褰曟ā寮忥紙淇濆瓨鍒?RL-DDPG_actions_exploration.txt锛?
        """
        self.training = training

    # =========2026.2.7 修复：set_epoch==========
    def set_epoch(self, epoch):
        """Set current training epoch."""
        self.current_epoch = epoch
    # ======修复结束

    def set_experiment_variant(self, variant):
        """Set experiment variant key."""
        self.exp_variant = str(variant).lower()

    # =========2026.2.20淇敼锛屽鍔辨ā寮忚缃帴鍙?=========
    # ======淇敼缁撴潫

    def reset(self, *, seed=None, options=None):
        # 閲嶇疆鐜骞惰繑鍥炲垵濮嬬姸鎬?
        if seed is not None:
            self.seed(seed)
        new = self.new
        self.episode_timestep = 0
        self._last_episode_timestep = 0
        self.n_episode += 1
        self.write_od_left(new)
        self.write_od_now()
        self.debug_log = []
        
        self.write_ini_cap(new)
        # 馃敼 2. 绔嬪埢鎵撳嵃鍏呯數妗╃姸鎬侊紙纭鍒濆鍖栨槸鍚︽垚鍔燂級
        self.print_charging_stations()
        #self.print_vehicle_status(0)
       # self.car_mem = [[-1, 0, 5] for _ in range(CAR_NUM)]
        self.car_mem = [[-1, 0, 5] for _ in range(len(self.car))]

        if new:
            self.mem_available = copy.deepcopy(self.available)
            self.new = 0

        self._last_edge = None
        self._last_episode_timestep = self.episode_timestep

        return self.state, {}

    def render(self, mode='human'):
        # 娓叉煋鐜鐨勯€昏緫锛屼竴鑸笉娑夊強锛屽彲瑙嗗寲鍐呭
        pass

    def seed(self, seed=None):
        # 璁剧疆闅忔満绉嶅瓙
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

