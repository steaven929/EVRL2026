目前的代码还不涉及保存模型，有需要的话（肯定是有需要的），你可以自行存在这个目录下面
=======.json
"CHARGER_PER_ZONE":  [
  [0, 1],  // 区域0的第0号位和1号位有充电桩
  [0],     // 区域1的第0号位有充电桩
  [1],     // 区域2的第1号位有充电桩
  []       // 区域3没有充电桩
]

=======测试部分修改前（main.py：
# 复制参数h
os.chdir('.\\specific')
cp("..\\" + cfg['data_folder'])
os.chdir('..')
# 训练（这里加了代码）
env, agent = dclilyenv.env_agent_config(cfg)
res_dic = dclilytrain.train(cfg, env, agent)
plot_rewards(res_dic['rewards'], cfg, tag="train")
# 测试
#res_dic = dclilytrain.test(cfg, env, agent)
#(res_dic['rewards'], cfg, tag="test")  # 画出结果

=======environment.py添加
#--------------------5.11
def manual_action_test(env, custom_action=None):
    print("=== 手动动作测试开始 ===")
    env.reset()

    if custom_action is None:
        Z = env.action_space.shape[0] // 3
        custom_action = np.array([
            1.0, 0.0, 0.0, 0.0,     # 出发倾向
            0.0, 1.0, 0.0, 0.0,     # 目的地倾向
            0.1, 0.9, 0.1, 0.1      # 充电倾向
        ])

    print("\n[Manual Action Input]")
    print("出发倾向:", custom_action[0:env.ZONE_NUM])
    print("目的地倾向:", custom_action[env.ZONE_NUM:2*env.ZONE_NUM])
    print("充电倾向:", custom_action[2*env.ZONE_NUM:3*env.ZONE_NUM])

    obs, reward, done, _ = env.step(custom_action)

    print(f"\n[Step 0 Completed] Reward = {reward:.2f}, Terminal = {done}")
    print("=== 手动动作测试结束 ===\n")

    ==========主函数修改1
    self.check_idle_vehicles()
    修改为
    self.check_idle_vehicles(charge_pref)
    ==========主函数修改2
    def check_idle_vehicles(self):
        """检查空闲车辆状态"""
        for i in range(CAR_NUM):
            # 只检查没有任务的车辆
            if self.car[i][2] == 0:
                current_battery = self.car[i][1]

                # 情况4: 电量高于阈值
                if current_battery >= 0.2 * BATTERY_CAPACITY:
                    continue  # 保持待命状态

                # 情况5: 电量低于阈值
                else:
                    current_zone = self.car[i][0]
                    target_charger = self.find_nearest_charger(current_zone)
                    if target_charger != -1:
                        self.car[i][0] = target_charger
                        self.car[i][2] = self.dist[current_zone][target_charger]
    修改为

    def check_idle_vehicles(self, charge_pref):
        for i in range(CAR_NUM):
            if self.car[i][2] == 0 and self.car[i][1] < 0.2 * BATTERY_CAPACITY:
                current_zone = self.car[i][0]
                target_charger = self.find_preferred_charger(current_zone, charge_pref)
                if target_charger != -1:
                    print(
                        f"  🔋车辆 {i} 电量低（{self.car[i][1]:.1f}），从区域 {current_zone} → 充电区域 {target_charger}（倾向={charge_pref[target_charger]:.2f}）")

                    self.car[i][0] = target_charger
                    self.car[i][2] = self.dist[current_zone][target_charger]

    修改为
        def check_idle_vehicles(self, charge_pref):
        """检查空闲车辆状态"""
        for i in range(CAR_NUM):
            # 只检查没有任务的车辆
            if self.car[i][2] == 0:
                current_battery = self.car[i][1]

                # 情况4: 电量高于阈值
                if current_battery >= 0.2 * BATTERY_CAPACITY:
                    continue  # 保持待命状态

                # 情况5: 电量低于阈值
                else:
                    current_zone = self.car[i][0]
                    target_charger = self.find_preferred_charger(current_zone, charge_pref)
                    if target_charger != -1:
                        print(
                        f"  🔋车辆 {i} 电量低（{self.car[i][1]:.1f}），从区域 {current_zone} → 充电区域 {target_charger}（倾向={charge_pref[target_charger]:.2f}）")

                        self.car[i][0] = target_charger
                        self.car[i][2] = self.dist[current_zone][target_charger]

    =========主函数添加
    def find_preferred_charger(self, current_zone, charge_pref):
    candidates = []
    for zone in CHARGER_ZONES:
        if self.available[0][zone] < self.mem_available[0][zone]:  # 有空位
            dist_score = -self.dist[current_zone][zone]  # 越近越好
            pref_score = charge_pref[zone]               # 越大越偏好
            score = 0.7 * pref_score + 0.3 * dist_score
            candidates.append((zone, score))
    if not candidates:
        return -1
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

    ==========# 打印的调度矩阵（已隐藏）
    ====== Debug log start ======

    =======5.24=====config.json新增
    // ✅ 新增每个区域的充电桩数量（长度必须等于 ZONE_NUM）
    "CHARGER_PER_ZONE": [2, 0, 1, 1]

    =========主函数
    self.car[i]
    索引	含义	举例
0	当前车辆所在区域	例如：3 表示在第3个区域
1	当前电量（电池容量）	例如：200.0 表示有200单位可用电量
2	到达目的地的剩余距离	例如：0 表示空闲，非0表示正在执行任务

    =========修改训练轮数
    util.py
    #训练轮数

    =========
    数组
    时段1，Zone 0的订单需求数OD_MAX
    时段1，Zone 0的可用位置数（包括无充电桩的位置）PLACE_NUM
    时段1，Zone 0的可用充电桩数量NUM_CHARGERS

    =========6.26修改
    1 .json  "CHARGER_PER_ZONE"修改为2维数组，各区域指定位置是充电桩
    2.action space正在修改 看gpt

    =========6.28修改
    加一个动作生成函数 def encode_action(zone_from_all=0, zone_to_all=1, charge_pref_all=1, zone_num=4):
    train.py 修改动作action
    =========7.21修改
    def find_nearest_charger(self, current_zone):
    for zone in list(CHARGER_ZONES)改了

    改成了

        def find_nearest_charger(self, current_zone):
        """找到最近的可用充电桩"""
        min_dist = float('inf')
        nearest_charger = -1

        for zone, charger_positions in enumerate(CHARGER_PER_ZONE):
            for pos in charger_positions:
                # 检查该位置是否还有空位
                if self.available[zone][pos] < self.mem_available[zone][pos]:
                    if self.dist[current_zone][zone] < min_dist:
                        min_dist = self.dist[current_zone][zone]
                        nearest_charger = (zone, pos)

        return nearest_charger
    =========7.23修改
    由于完整代码报错，修改了21号改的这个函数
    ======原版：
        def find_preferred_charger(self, current_zone, charge_pref):
        candidates = []
        for zone in CHARGER_ZONES:
            if self.available[0][zone] < self.mem_available[0][zone]:
                dist_score = -self.dist[current_zone][zone]
                prefer_score = charge_pref[zone]
                score = 0.7 * prefer_score + 0.3 * dist_score
                candidates.append((zone, score))
        if not candidates:
            return -1
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    =======改为：
    def find_nearest_charger(self, current_zone):
    """找到最近的可用充电桩"""
    min_dist = float('inf')
    nearest_charger = -1

    for zone in CHARGER_ZONES:
        charger_count = CHARGER_PER_ZONE[zone]
        for pos in range(charger_count):  # ✅ 正确：range(N) 表示该区域有 N 个充电桩
            if self.available[zone][pos] < self.mem_available[zone][pos]:
                if self.dist[current_zone][zone] < min_dist:
                    min_dist = self.dist[current_zone][zone]
                    nearest_charger = zone  # 可改为 (zone, pos) 如果你有用

    return nearest_charger

    ========改为：
    def find_preferred_charger(self, current_zone, charge_pref):
    """
    在可用充电桩中找到当前 zone 的首选充电区域（考虑距离和充电偏好）。
    charge_pref: dict or list, 可选加入偏好权重（例如优先靠近 or 分散）
    返回：最佳 charger 所在的 zone（如有需要也可返回 pos）
    """
    min_dist = float('inf')
    preferred_zone = -1

    for zone in CHARGER_ZONES:
        charger_count = CHARGER_PER_ZONE[zone]
        for pos in range(charger_count):
            if self.available[zone][pos] < self.mem_available[zone][pos]:
                # ⚙️ 可引入 charge_pref 权重，这里简化为最近优先
                distance = self.dist[current_zone][zone]
                if distance < min_dist:
                    min_dist = distance
                    preferred_zone = zone

    return preferred_zone  # 或 return (preferred_zone, pos) if needed
    =======图像在util.py，训练轮数也在util.py
    def plot_rewards(rewards,cfg,path=None,tag='train'):
    图像title 原来是：
        plt.title(f"{tag}ing curve on {cfg['device']} of {cfg['algo_name']} for {cfg['env_name']}")

        =======plot V2版 （改标题、改背景格子）
        def plot_rewards(rewards,cfg,path=None,tag='train'):
    sns.set()
    plt.figure(facecolor='white')  # 创建一个图形实例，方便同时多画几个图# 设置画布背景为白色
    #plt.savefig(path, facecolor='white')  # path 需要非 None
    ax = plt.gca()
    ax.set_facecolor('white')  # 设置绘图区域背景为白色
    plt.grid(False)# 关闭背景网格线（可选）
    plt.title(f"{tag}ing curve on {cfg['device']} of {cfg['algo_name']} ")
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()
    plt.show()

    =========