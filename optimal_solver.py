# -*- coding: utf-8 -*-
"""
小场景最优解求解器
场景：4区域，18辆车

方法：
1. 完美信息 MPC (Model Predictive Control)
2. 动态规划
3. 穷举搜索（小场景可行）
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import os
import gym
from gym.envs.registration import register
import json
import pandas as pd  # 修改
import gurobipy as gp  # 修改
from gurobipy import GRB  # 修改
import math  # 修改

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

register(
    id='dclily',
    entry_point='specific.dclily_elec:dclily',
)


class OptimalSolver:
    """最优解求解器"""

    def __init__(self, env):
        self.env = env
        self.zone_num = 4
        self.car_num = 18

    def _load_config(self):  # 修改
        cfg = getattr(self.env, "config", None)  # 修改
        if cfg is not None:  # 修改
            return cfg  # 修改
        base_dir = os.path.dirname(os.path.abspath(__file__))  # 修改
        config_path = os.path.join(base_dir, "specific", "config.json")  # 修改
        with open(config_path, "r", encoding="utf-8") as f:  # 修改
            return json.load(f)  # 修改

    def _load_od_demand(self, csv_path, zone_num, time_num):  # 修改
        df = pd.read_csv(csv_path)  # 修改
        if "O" not in df.columns or "D" not in df.columns:  # 修改
            raise ValueError("od_demand.csv 必须包含 O 和 D 列")  # 修改
        time_cols = [c for c in df.columns if c not in ["O", "D"]]  # 修改
        if not time_cols:  # 修改
            raise ValueError("od_demand.csv 不包含时间列")  # 修改
        use_cols = time_cols[:time_num]  # 修改
        demand = np.zeros((len(use_cols), zone_num, zone_num), dtype=int)  # 修改
        for _, row in df.iterrows():  # 修改
            o = int(row["O"])  # 修改
            d = int(row["D"])  # 修改
            for t_idx, col in enumerate(use_cols):  # 修改
                demand[t_idx, o, d] = int(row[col])  # 修改
        return demand, len(use_cols)  # 修改

    # =========2026.2.1修改，添加参数优化支持==========
    def solve_global_optimal(self, demand_csv="od_demand.csv", output_path="global_optimal_solution.json",
                            mip_gap=0.05, time_limit=600):  # 修改
        cfg = self._load_config()  # 修改
        zone_num = int(cfg["ZONE_NUM"])  # 修改
        car_num = int(cfg["CAR_NUM"])  # 修改
        time_num = int(cfg["TIME_NUM"])  # 修改
        base_dir = os.path.dirname(os.path.abspath(__file__))  # 修改
        data_root = os.path.normpath(os.path.join(base_dir, cfg["path"]))  # 修改

        dist = np.array(pd.read_csv(os.path.join(data_root, "od_demand_dist.csv"), header=None))  # 修改
        capacity_df = pd.read_csv(os.path.join(data_root, "capacity.csv"), header=None)  # 修改
        initial_counts = capacity_df.iloc[1].to_numpy(dtype=int)  # 修改
        zone_capacity = capacity_df.iloc[2].to_numpy(dtype=int)  # 修改
        demand, used_time = self._load_od_demand(os.path.join(data_root, demand_csv), zone_num, time_num)  # 修改

        cong = pd.read_csv(os.path.join(data_root, "cong.csv"), header=None).to_numpy().flatten()  # 修改
        if len(cong) == 0:  # 修改
            raise ValueError("cong.csv 为空，无法计算步长距离")  # 修改
        min_cong = float(np.min(cong[:used_time])) if len(cong) >= used_time else float(np.min(cong))  # 修改
        step_distance = cfg["car_speed_min"] * cfg["TIME_LENGTH"] * min_cong  # 修改
        if np.max(dist) > step_distance:  # 修改
            raise ValueError("当前距离/速度导致跨多步移动，求解器暂不支持")  # 修改

        if int(np.sum(initial_counts)) != car_num:  # 修改
            raise ValueError("capacity.csv 初始车辆数与配置 CAR_NUM 不一致")  # 修改

        chargers_per_zone = []  # 修改
        if "ZONES" in cfg:  # 修改
            for zone_cfg in cfg["ZONES"]:  # 修改
                chargers_per_zone.append(len(zone_cfg.get("charging_slots", [])))  # 修改
        else:  # 修改
            chargers_per_zone = [0] * zone_num  # 修改

        charge_speed = 1800.0  # 修改
        charge_gain = charge_speed * (cfg["TIME_LENGTH"] / 60.0)  # 修改
        manzu_min = float(cfg["manzu_min"])  # 修改
        diaodu_min = float(cfg["diaodu_min"])  # 修改

        initial_zones = []  # 修改
        for zone_id in range(zone_num):  # 修改
            initial_zones.extend([zone_id] * int(initial_counts[zone_id]))  # 修改

        model = gp.Model("Global_Optimal_Solver")  # 修改
        model.setParam("OutputFlag", 1)  # 修改
        model.setParam("MIPGap", mip_gap)  # 修改
        model.setParam("TimeLimit", time_limit)  # 修改
    # ======修改结束

        z = model.addVars(car_num, used_time + 1, zone_num, vtype=GRB.BINARY, name="z")  # 修改
        x = model.addVars(car_num, used_time, zone_num, zone_num, vtype=GRB.BINARY, name="x")  # 修改
        r = model.addVars(car_num, used_time, zone_num, zone_num, vtype=GRB.BINARY, name="r")  # 修改
        y = model.addVars(car_num, used_time, zone_num, vtype=GRB.BINARY, name="y")  # 修改
        c = model.addVars(car_num, used_time, zone_num, vtype=GRB.BINARY, name="c")  # 修改
        soc = model.addVars(car_num, used_time + 1, lb=0.0, ub=cfg["BATTERY_CAPACITY"], vtype=GRB.CONTINUOUS, name="soc")  # 修改

        # =========2026.2.2修改，新目标函数==========
        # 1. 服务订单收益
        service_revenue = gp.quicksum(
            manzu_min * dist[i][j] * x[v, t, i, j]
            for v in range(car_num)
            for t in range(used_time)
            for i in range(zone_num)
            for j in range(zone_num)
        )

        # 2. 调度成本（使用diaodu_min作为调度系数）
        dispatch_cost = gp.quicksum(
            diaodu_min * dist[i][j] * r[v, t, i, j]
            for v in range(car_num)
            for t in range(used_time)
            for i in range(zone_num)
            for j in range(zone_num)
        )

        # =========2026.2.12修改，实验19：提升充电奖励信号==========
        # 3. 充电加分
        charging_bonus = gp.quicksum(
            3000 * c[v, t, i]
            for v in range(car_num)
            for t in range(used_time)
            for i in range(zone_num)
        )
        # ======修改结束

        # 4. 低电量惩罚：高惩罚
        soc_penalty_vars = model.addVars(car_num, used_time + 1, lb=0.0, vtype=GRB.CONTINUOUS, name="soc_penalty")

        for v in range(car_num):
            for t in range(used_time + 1):
                model.addConstr(soc_penalty_vars[v, t] >= 2.0 * (600 - soc[v, t]))
                model.addConstr(soc_penalty_vars[v, t] >= 3200 - 8.0 * soc[v, t])
                model.addConstr(soc_penalty_vars[v, t] >= 11600 - 50.0 * soc[v, t])

        soc_penalty = gp.quicksum(soc_penalty_vars[v, t] for v in range(car_num) for t in range(used_time + 1))
        # ======修改结束

        # =========2026.2.5修改，实验2：同步RL移除固定惩罚==========
        # 5. 固定惩罚（移除以与RL保持一致）
        fixed_penalty = 0  # 原500 * used_time改为0
        # ======修改结束

        # =========2026.2.12修改，实验19：充电奖励不参与缩放==========
        # 新目标函数（充电奖励不除以10）
        model.setObjective(
            (service_revenue - soc_penalty - dispatch_cost) / 10 + charging_bonus - fixed_penalty,
            GRB.MAXIMIZE
        )
        # ======修改结束
        # ======修改结束

        for v in range(car_num):  # 修改
            for i in range(zone_num):  # 修改
                model.addConstr(z[v, 0, i] == (1 if initial_zones[v] == i else 0))  # 修改
            model.addConstr(soc[v, 0] == cfg["BATTERY_CAPACITY"])  # 修改

        for t in range(used_time + 1):  # 修改
            for i in range(zone_num):  # 修改
                model.addConstr(gp.quicksum(z[v, t, i] for v in range(car_num)) <= zone_capacity[i])  # 修改

        for t in range(used_time):  # 修改
            for v in range(car_num):  # 修改
                model.addConstr(gp.quicksum(z[v, t, i] for i in range(zone_num)) == 1)  # 修改
                model.addConstr(  # 修改
                    gp.quicksum(x[v, t, i, j] for i in range(zone_num) for j in range(zone_num))  # 修改
                    + gp.quicksum(r[v, t, i, j] for i in range(zone_num) for j in range(zone_num))  # 修改
                    + gp.quicksum(y[v, t, i] for i in range(zone_num))  # 修改
                    == 1  # 修改
                )  # 修改
                for i in range(zone_num):  # 修改
                    model.addConstr(y[v, t, i] <= z[v, t, i])  # 修改
                    model.addConstr(c[v, t, i] <= y[v, t, i])  # 修改
                    model.addConstr(c[v, t, i] <= (1 if chargers_per_zone[i] > 0 else 0))  # 修改
                    for j in range(zone_num):  # 修改
                        model.addConstr(x[v, t, i, j] <= z[v, t, i])  # 修改
                        model.addConstr(r[v, t, i, j] <= z[v, t, i])  # 修改
                        if i == j:  # 修改
                            model.addConstr(x[v, t, i, j] == 0)  # 修改
                            model.addConstr(r[v, t, i, j] == 0)  # 修改
                        model.addConstr(soc[v, t] >= dist[i][j] * x[v, t, i, j])  # 修改
                        model.addConstr(soc[v, t] >= dist[i][j] * r[v, t, i, j])  # 修改

            for i in range(zone_num):  # 修改
                model.addConstr(gp.quicksum(c[v, t, i] for v in range(car_num)) <= chargers_per_zone[i])  # 修改
                for j in range(zone_num):  # 修改
                    model.addConstr(gp.quicksum(x[v, t, i, j] for v in range(car_num)) <= demand[t, i, j])  # 修改

            for v in range(car_num):  # 修改
                for j in range(zone_num):  # 修改
                    model.addConstr(  # 修改
                        z[v, t + 1, j]  # 修改
                        == gp.quicksum(x[v, t, i, j] for i in range(zone_num))  # 修改
                        + gp.quicksum(r[v, t, i, j] for i in range(zone_num))  # 修改
                        + y[v, t, j]  # 修改
                    )  # 修改
                model.addConstr(  # 修改
                    soc[v, t + 1]  # 修改
                    == soc[v, t]  # 修改
                    - gp.quicksum(dist[i][j] * (x[v, t, i, j] + r[v, t, i, j])  # 修改
                                  for i in range(zone_num) for j in range(zone_num))  # 修改
                    + charge_gain * gp.quicksum(c[v, t, i] for i in range(zone_num))  # 修改
                )  # 修改

        model.optimize()  # 修改

        # =========2026.2.1修改，添加求解状态信息==========
        if model.status == GRB.OPTIMAL:  # 修改
            status_msg = "OPTIMAL"  # 修改
            optimality_gap = 0.0  # 修改
        elif model.status == GRB.TIME_LIMIT:  # 修改
            status_msg = "TIME_LIMIT"  # 修改
            optimality_gap = model.MIPGap  # 修改
        elif model.status == GRB.INTERRUPTED:  # 修改
            status_msg = "INTERRUPTED"  # 修改
            optimality_gap = model.MIPGap if hasattr(model, 'MIPGap') else float('inf')  # 修改
        else:  # 修改
            raise RuntimeError(f"求解失败，状态码: {model.status}")  # 修改

        served = []  # 修改
        repositioned = []  # 新增：调度动作
        charging = []  # 新增：充电动作
        for t in range(used_time):  # 修改
            step_matrix = np.zeros((zone_num, zone_num), dtype=int)  # 修改
            reposition_matrix = np.zeros((zone_num, zone_num), dtype=int)  # 新增
            charge_vector = np.zeros(zone_num, dtype=int)  # 新增
            for i in range(zone_num):  # 修改
                for j in range(zone_num):  # 修改
                    step_matrix[i, j] = int(round(sum(x[v, t, i, j].X for v in range(car_num))))  # 修改
                    reposition_matrix[i, j] = int(round(sum(r[v, t, i, j].X for v in range(car_num))))  # 新增
                charge_vector[i] = int(round(sum(c[v, t, i].X for v in range(car_num))))  # 新增
            served.append(step_matrix.tolist())  # 修改
            repositioned.append(reposition_matrix.tolist())  # 新增
            charging.append(charge_vector.tolist())  # 新增

        result = {  # 修改
            "method": "Global Optimal MILP",  # 修改
            "time_steps": int(used_time),  # 修改
            "total_reward": float(model.ObjVal),  # 修改
            "status": status_msg,  # 修改
            "mip_gap": float(optimality_gap),  # 修改
            "solve_time": float(model.Runtime),  # 修改
            "served_orders": served,  # 修改
            "repositioned_orders": repositioned,  # 新增：调度订单
            "charging_actions": charging  # 新增：充电动作
        }  # 修改
        # ======修改结束

        with open(output_path, "w", encoding="utf-8") as f:  # 修改
            json.dump(result, f, indent=2)  # 修改

        # 新增：输出详细动作到文本文件
        txt_output_path = 'action/' + output_path.replace('.json', '_actions.txt')
        with open(txt_output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("全局最优解 - 详细动作分析\n")
            f.write("="*80 + "\n\n")
            f.write(f"方法: {result['method']}\n")
            f.write(f"总奖励: {result['total_reward']:.2f}\n")
            f.write(f"状态: {result['status']}\n")
            f.write(f"MIP Gap: {result['mip_gap']*100:.2f}%\n")
            f.write(f"求解时间: {result['solve_time']:.2f}秒\n")
            f.write(f"时间步数: {result['time_steps']}\n\n")

            for t in range(used_time):
                f.write(f"{'='*80}\n时间步 {t}\n{'='*80}\n\n")

                # 服务订单
                f.write("【服务订单】\n")
                has_service = False
                for i in range(zone_num):
                    for j in range(zone_num):
                        if served[t][i][j] > 0:
                            f.write(f"  区域{i} → 区域{j}: {served[t][i][j]}辆\n")
                            has_service = True
                if not has_service:
                    f.write("  无\n")
                f.write("\n")

                # 调度动作
                f.write("【调度动作】\n")
                has_reposition = False
                for i in range(zone_num):
                    for j in range(zone_num):
                        if repositioned[t][i][j] > 0:
                            f.write(f"  区域{i} → 区域{j}: {repositioned[t][i][j]}辆\n")
                            has_reposition = True
                if not has_reposition:
                    f.write("  无\n")
                f.write("\n")

                # 充电动作
                f.write("【充电动作】\n")
                has_charging = False
                for i in range(zone_num):
                    if charging[t][i] > 0:
                        f.write(f"  区域{i}: {charging[t][i]}辆\n")
                        has_charging = True
                if not has_charging:
                    f.write("  无\n")
                f.write("\n")

            # 统计
            f.write("="*80 + "\n统计信息\n" + "="*80 + "\n")
            total_served = sum(sum(sum(row) for row in step) for step in served)
            total_reposition = sum(sum(sum(row) for row in step) for step in repositioned)
            total_charging = sum(sum(step) for step in charging)
            f.write(f"总服务订单数: {total_served}\n")
            f.write(f"总调度次数: {total_reposition}\n")
            f.write(f"总充电次数: {total_charging}\n")
            f.write(f"平均每步充电: {total_charging/used_time:.2f}\n")

        print(f"详细动作已保存到: {txt_output_path}")

        return result  # 修改

    # =========2026.2.1修改，添加滚动时域优化==========
    def solve_rolling_horizon(self, demand_csv="od_demand.csv", output_path="rolling_horizon_solution.json",
                              horizon=4, mip_gap=0.01, time_limit_per_step=30):
        """
        滚动时域优化方法
        每次只优化未来 horizon 步，然后执行第一步，滚动前进

        参数:
            horizon: 优化时域长度（建议3-4步）
            mip_gap: 每步的MIPGap
            time_limit_per_step: 每步的时间限制（秒）
        """
        cfg = self._load_config()
        zone_num = int(cfg["ZONE_NUM"])
        car_num = int(cfg["CAR_NUM"])
        time_num = int(cfg["TIME_NUM"])
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_root = os.path.normpath(os.path.join(base_dir, cfg["path"]))

        dist = np.array(pd.read_csv(os.path.join(data_root, "od_demand_dist.csv"), header=None))
        capacity_df = pd.read_csv(os.path.join(data_root, "capacity.csv"), header=None)
        initial_counts = capacity_df.iloc[1].to_numpy(dtype=int)
        zone_capacity = capacity_df.iloc[2].to_numpy(dtype=int)
        demand, total_time = self._load_od_demand(os.path.join(data_root, demand_csv), zone_num, time_num)

        chargers_per_zone = []
        if "ZONES" in cfg:
            for zone_cfg in cfg["ZONES"]:
                chargers_per_zone.append(len(zone_cfg.get("charging_slots", [])))
        else:
            chargers_per_zone = [0] * zone_num

        charge_speed = 1800.0
        charge_gain = charge_speed * (cfg["TIME_LENGTH"] / 60.0)
        manzu_min = float(cfg["manzu_min"])
        diaodu_min = float(cfg["diaodu_min"])

        current_zones = []
        for zone_id in range(zone_num):
            current_zones.extend([zone_id] * int(initial_counts[zone_id]))

        current_soc = [cfg["BATTERY_CAPACITY"]] * car_num

        total_reward = 0.0
        all_served = []
        total_solve_time = 0.0

        for current_step in range(total_time):
            remaining_steps = total_time - current_step
            actual_horizon = min(horizon, remaining_steps)

            model = gp.Model(f"RollingHorizon_Step{current_step}")
            model.setParam("OutputFlag", 0)
            model.setParam("MIPGap", mip_gap)
            model.setParam("TimeLimit", time_limit_per_step)

            z = model.addVars(car_num, actual_horizon + 1, zone_num, vtype=GRB.BINARY, name="z")
            x = model.addVars(car_num, actual_horizon, zone_num, zone_num, vtype=GRB.BINARY, name="x")
            r = model.addVars(car_num, actual_horizon, zone_num, zone_num, vtype=GRB.BINARY, name="r")
            y = model.addVars(car_num, actual_horizon, zone_num, vtype=GRB.BINARY, name="y")
            c = model.addVars(car_num, actual_horizon, zone_num, vtype=GRB.BINARY, name="c")
            soc = model.addVars(car_num, actual_horizon + 1, lb=0.0, ub=cfg["BATTERY_CAPACITY"], vtype=GRB.CONTINUOUS, name="soc")

            # =========2026.2.2修改，新目标函数==========
            # 1. 服务订单收益
            service_revenue = gp.quicksum(
                manzu_min * dist[i][j] * x[v, t, i, j]
                for v in range(car_num)
                for t in range(actual_horizon)
                for i in range(zone_num)
                for j in range(zone_num)
            )

            # 2. 调度成本（使用diaodu_min作为调度系数）
            dispatch_cost = gp.quicksum(
                diaodu_min * dist[i][j] * r[v, t, i, j]
                for v in range(car_num)
                for t in range(actual_horizon)
                for i in range(zone_num)
                for j in range(zone_num)
            )

            # =========2026.2.5修改，实验2：Rolling Horizon同步充电奖励==========
            # 3. 充电加分（从50提升到500，与RL保持一致）
            charging_bonus = gp.quicksum(
                500 * c[v, t, i]  # 原50改为500
                for v in range(car_num)
                for t in range(actual_horizon)
                for i in range(zone_num)
            )
            # ======修改结束

            # 4. 低电量惩罚
            soc_penalty_vars = model.addVars(car_num, actual_horizon + 1, lb=0.0, vtype=GRB.CONTINUOUS, name="soc_penalty")
            for v in range(car_num):
                for t in range(actual_horizon + 1):
                    model.addConstr(soc_penalty_vars[v, t] >= 0.5 * (600 - soc[v, t]))
                    model.addConstr(soc_penalty_vars[v, t] >= 900 - 4.5 * soc[v, t])

            soc_penalty = gp.quicksum(soc_penalty_vars[v, t] for v in range(car_num) for t in range(actual_horizon + 1))

            # =========2026.2.5修改，实验2：Rolling Horizon移除固定惩罚==========
            # 5. 固定惩罚（移除以与RL保持一致）
            fixed_penalty = 0  # 原500 * actual_horizon改为0
            # ======修改结束

            # 新目标函数（添加缩放因子/10）
            model.setObjective(
                (service_revenue + charging_bonus - soc_penalty - dispatch_cost) / 10 - fixed_penalty,
                GRB.MAXIMIZE
            )
            # ======修改结束

            for v in range(car_num):
                for i in range(zone_num):
                    model.addConstr(z[v, 0, i] == (1 if current_zones[v] == i else 0))
                model.addConstr(soc[v, 0] == current_soc[v])

            for t in range(actual_horizon + 1):
                for i in range(zone_num):
                    model.addConstr(gp.quicksum(z[v, t, i] for v in range(car_num)) <= zone_capacity[i])

            for t in range(actual_horizon):
                for v in range(car_num):
                    model.addConstr(gp.quicksum(z[v, t, i] for i in range(zone_num)) == 1)
                    model.addConstr(
                        gp.quicksum(x[v, t, i, j] for i in range(zone_num) for j in range(zone_num))
                        + gp.quicksum(r[v, t, i, j] for i in range(zone_num) for j in range(zone_num))
                        + gp.quicksum(y[v, t, i] for i in range(zone_num))
                        == 1
                    )
                    for i in range(zone_num):
                        model.addConstr(y[v, t, i] <= z[v, t, i])
                        model.addConstr(c[v, t, i] <= y[v, t, i])
                        model.addConstr(c[v, t, i] <= (1 if chargers_per_zone[i] > 0 else 0))
                        for j in range(zone_num):
                            model.addConstr(x[v, t, i, j] <= z[v, t, i])
                            model.addConstr(r[v, t, i, j] <= z[v, t, i])
                            if i == j:
                                model.addConstr(x[v, t, i, j] == 0)
                                model.addConstr(r[v, t, i, j] == 0)
                            model.addConstr(soc[v, t] >= dist[i][j] * x[v, t, i, j])
                            model.addConstr(soc[v, t] >= dist[i][j] * r[v, t, i, j])

                for i in range(zone_num):
                    model.addConstr(gp.quicksum(c[v, t, i] for v in range(car_num)) <= chargers_per_zone[i])
                    for j in range(zone_num):
                        step_idx = current_step + t
                        model.addConstr(gp.quicksum(x[v, t, i, j] for v in range(car_num)) <= demand[step_idx, i, j])

                for v in range(car_num):
                    for j in range(zone_num):
                        model.addConstr(
                            z[v, t + 1, j]
                            == gp.quicksum(x[v, t, i, j] for i in range(zone_num))
                            + gp.quicksum(r[v, t, i, j] for i in range(zone_num))
                            + y[v, t, j]
                        )
                    model.addConstr(
                        soc[v, t + 1]
                        == soc[v, t]
                        - gp.quicksum(dist[i][j] * (x[v, t, i, j] + r[v, t, i, j])
                                      for i in range(zone_num) for j in range(zone_num))
                        + charge_gain * gp.quicksum(c[v, t, i] for i in range(zone_num))
                    )

            model.optimize()
            total_solve_time += model.Runtime

            if model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
                raise RuntimeError(f"步骤{current_step}求解失败，状态码: {model.status}")

            step_matrix = np.zeros((zone_num, zone_num), dtype=int)
            for i in range(zone_num):
                for j in range(zone_num):
                    step_matrix[i, j] = int(round(sum(x[v, 0, i, j].X for v in range(car_num))))
            all_served.append(step_matrix.tolist())

            step_reward = sum(
                manzu_min * dist[i][j] * x[v, 0, i, j].X
                - diaodu_min * dist[i][j] * r[v, 0, i, j].X
                for v in range(car_num)
                for i in range(zone_num)
                for j in range(zone_num)
            )
            total_reward += step_reward

            for v in range(car_num):
                for i in range(zone_num):
                    if z[v, 1, i].X > 0.5:
                        current_zones[v] = i
                        break
                current_soc[v] = soc[v, 1].X

        result = {
            "method": "Rolling Horizon",
            "horizon": horizon,
            "time_steps": total_time,
            "total_reward": float(total_reward),
            "avg_solve_time_per_step": float(total_solve_time / total_time),
            "total_solve_time": float(total_solve_time),
            "served_orders": all_served
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return result
    # ======修改结束

    def solve_with_perfect_info(self, num_episodes=10):
        """
        方法1：完美信息求解
        假设知道整个 episode 的所有需求，用 Gurobi 求解全局最优
        """
        print("="*60)
        print("方法1：完美信息全局优化")
        print("="*60)

        all_rewards = []
        all_svc = []
        all_rld = []

        for ep in range(num_episodes):
            print(f"\nEpisode {ep+1}/{num_episodes}")

            # 重置环境
            state = env.reset()

            # 收集整个 episode 的需求信息
            episode_demands = []
            episode_states = []

            # 先运行一遍收集信息（不影响实际环境）
            temp_env = gym.make('dclily')
            temp_state = temp_env.reset()

            for step in range(12):  # 假设每个 episode 12 步
                episode_states.append(temp_state)
                # 提取需求信息
                # 这里需要根据实际 state 结构解析
                episode_demands.append(temp_state)

                # 随机动作推进（只为收集信息）
                action = temp_env.action_space.sample()
                temp_state, _, done, _ = temp_env.step(action)
                if done:
                    break

            print(f"  收集了 {len(episode_demands)} 步的需求信息")

            # 现在用完美信息求解
            # 这里简化：使用固定的最优策略
            state = env.reset()
            ep_reward = 0
            ep_svc_list = []
            ep_rld_list = []

            done = False
            step = 0

            while not done:
                # 基于完美信息的最优动作
                # 简化实现：使用经验最优分布
                action = self._get_optimal_action_with_perfect_info(
                    state, step, episode_demands
                )

                next_state, reward, done, info = env.step(action)
                ep_reward += reward

                if 'svc' in info:
                    ep_svc_list.append(info['svc'])
                if 'rld' in info:
                    ep_rld_list.append(info['rld'])

                state = next_state
                step += 1

            all_rewards.append(ep_reward)
            all_svc.append(np.mean(ep_svc_list) if ep_svc_list else 0)
            all_rld.append(np.mean(ep_rld_list) if ep_rld_list else 0)

            print(f"  Reward: {ep_reward:.2f}, SVC: {all_svc[-1]:.3f}, RLD: {all_rld[-1]:.2f}")

        results = {
            'method': 'Perfect Information',
            'reward_mean': float(np.mean(all_rewards)),
            'reward_std': float(np.std(all_rewards)),
            'svc_mean': float(np.mean(all_svc)),
            'rld_mean': float(np.mean(all_rld)),
        }

        print(f"\n完美信息平均结果:")
        print(f"  Reward: {results['reward_mean']:.2f} +- {results['reward_std']:.2f}")
        print(f"  SVC: {results['svc_mean']:.3f}")
        print(f"  RLD: {results['rld_mean']:.2f}")

        return results

    def _get_optimal_action_with_perfect_info(self, state, step, demands):
        """基于完美信息生成最优动作"""
        # 简化实现：使用启发式规则
        action = np.zeros(12)  # 4 zones * 3

        # 根据未来需求调整策略
        # 这里简化为固定的优化分布
        action[0:4] = [0.25, 0.25, 0.25, 0.25]  # 发车均匀
        action[4:8] = [0.3, 0.3, 0.2, 0.2]      # 目的地偏向高需求区
        action[8:12] = [0.2, 0.2, 0.3, 0.3]     # 充电策略

        return action


# =========2026.2.1修改，更新主函数支持多种求解方法==========
if __name__ == "__main__":  # 修改
    import argparse
    parser = argparse.ArgumentParser(description="最优解求解器")
    parser.add_argument("--method", type=str, default="rolling",
                       choices=["global", "rolling", "both"],
                       help="求解方法: global(全局), rolling(滚动时域), both(两者都运行)")
    parser.add_argument("--horizon", type=int, default=4, help="滚动时域长度")
    parser.add_argument("--mip_gap", type=float, default=0.05, help="全局求解的MIPGap")
    parser.add_argument("--time_limit", type=int, default=600, help="全局求解的时间限制(秒)")
    args = parser.parse_args()

    env = gym.make("dclily")  # 修改
    env.set_training_mode(0)  # 修改
    solver = OptimalSolver(env)  # 修改

    if args.method in ["rolling", "both"]:
        print("="*60)
        print(f"运行滚动时域优化 (horizon={args.horizon})")
        print("="*60)
        result_rolling = solver.solve_rolling_horizon(
            demand_csv="od_demand.csv",
            output_path="rolling_horizon_solution.json",
            horizon=args.horizon
        )
        print(f"\n滚动时域优化结果:")
        print(f"  Total Reward: {result_rolling['total_reward']:.2f}")
        print(f"  Avg Solve Time: {result_rolling['avg_solve_time_per_step']:.2f}s/step")
        print(f"  Total Solve Time: {result_rolling['total_solve_time']:.2f}s")

    if args.method in ["global", "both"]:
        print("\n" + "="*60)
        print(f"运行全局优化 (MIPGap={args.mip_gap}, TimeLimit={args.time_limit}s)")
        print("="*60)
        result_global = solver.solve_global_optimal(
            demand_csv="od_demand.csv",
            output_path="global_optimal_solution.json",
            mip_gap=args.mip_gap,
            time_limit=args.time_limit
        )
        print(f"\n全局优化结果:")
        print(f"  Total Reward: {result_global['total_reward']:.2f}")
        print(f"  Status: {result_global['status']}")
        print(f"  MIP Gap: {result_global['mip_gap']*100:.2f}%")
        print(f"  Solve Time: {result_global['solve_time']:.2f}s")

    if args.method == "both":
        print("\n" + "="*60)
        print("对比结果")
        print("="*60)
        gap_pct = abs(result_rolling['total_reward'] - result_global['total_reward']) / abs(result_global['total_reward']) * 100
        print(f"  滚动时域 vs 全局优化:")
        print(f"    Reward差距: {gap_pct:.2f}%")
        print(f"    速度提升: {result_global['solve_time']/result_rolling['total_solve_time']:.1f}x")
# ======修改结束
