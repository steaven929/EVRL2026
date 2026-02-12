# Project Knowledge Base & Dev Log

## 1. 开发与修改规范 (Development Rules)
* **代码修改标记**: 必须使用标记包裹：`# =========2026.XX.XX修改==========` ... `# ======修改结束`。
* **逻辑保护**:
    * 严禁删除 Gurobi 求解器代码。
    * 修改 Reward 公式前必须进行数学对齐检查（Environment vs Optim Solver）。

## 2. 关键更新日志 (Chronological Updates)

### 2.1 [2026-01-30] 训练指标与超参数优化
**背景**: 本环境为成本最小化问题，Reward 长期为负数。

#### A. 指标理解 (Metrics Deep Dive)
* **Q Value 趋势**:
    * **初期 (探索期)**: 0 → -600。Critic 正在学习真实的负回报环境，这是**正常现象**。
    * **后期 (优化期)**: -600 → -300。**Q 值上升**代表策略改进（累积负 reward 减少），这是**收敛标志**。
* **Actor Loss**:
    * 定义为 `-Mean(Q)`。
    * **Loss 下降** (+600 → +400) 代表 Q 值上升，即策略变好。
* **异常判定**: 若 Q 值持续下降无反弹 (>200轮)，或 Q 值与 Reward 趋势严重背离，意味着 Critic 估计偏差。

#### B. 成功超参数配置
* `critic_lr = 5e-4`: (原 1e-3) 有效减少过拟合。
* `batch_size = 64`: (原 32) 提高梯度估计稳定性。
* `tau = 1e-2`: 软更新步长。
* `actor_lr = 1e-4`。

---

### 2.2 [2026-02-02] 新版奖励函数设计 (Reward Engineering)
**总公式**:
$$Reward = \frac{(\sum \text{Service} + \text{Charge} - \sum \text{SOC\_Penalty} - \sum \text{Dispatch})}{10} - 500$$

**细节组件定义**:
1.  **Service (订单收益)**: $1.5 \times \text{dist} \times \text{orders}$
2.  **Charge (充电奖励)**: 每车 $+50$
3.  **Dispatch (调度成本)**: $0.01 \times \text{dist} \times \text{orders}$ (系数从 0.0001 上调)
4.  **SOC Penalty (分段惩罚)**:
    * $SOC \ge 600$: $0$
    * $200 \le SOC < 600$: $0.5 \times (600 - SOC)$
    * $SOC < 200$: $200 + 5.0 \times (200 - SOC)$
5.  **Scaling & Baseline**:
    * `/10`: 缩放因子，防止梯度爆炸。
    * `-500`: 固定基线惩罚，约占总 Reward 的 25%。

## 3. 性能基准与求解器对比 (Benchmarks)
**场景设置**: 4 区域、18 车辆、12 时间步。

### 3.1 算法性能阶梯
用于判断当前模型是否达标：

| 策略 | 预期 Reward | 说明 |
| :--- | :--- | :--- |
| **Random (未训练)** | ≈ -2940 | 下界参考 |
| **RL-DDPG (700轮)** | ≈ -2294 | 较初始提升 ~22% |
| **Global Optimal** | ≈ -1850 | 理论上界 (MIPGap 0%) |

### 3.2 求解器效率对比
用于选择合适的 Baseline 生成方式：

| 方案 | 命令后缀 | 精度 | 耗时 |
| :--- | :--- | :--- | :--- |
| **Global Optimal** | `--method global --mip_gap 0.05` | 100% (理论上界) | 10 min |
| **Rolling Horizon** | `--method rolling --horizon 4` | 95%-98% (近似解) | 20-30 s |

**已知限制**: 必须满足 `max(dist) < car_speed * TIME_LENGTH`，否则求解器会报错。

## 4. 调试指南索引 (Documentation Map)
详细文档请查阅 `checkpoint/` 目录：
* `reward_tuning_guide.md`: 调整 Reward 缩放因子或惩罚项的指南。
* `debugging_plan.md`: 训练不收敛（Q值发散）时的排查步骤。
* `reward_analysis.md`: 详细的数学公式拆解与 Real Reward 构成。

---

### 2.3 [2026-02-12] 后期动作塌缩修复（实验19后续）
**问题现象**: 后期时间步出现“服务/调度/充电全零”，与最优解逐步有动作不一致。

**代码修复点**:
* `specific/dclily_elec.py`
  1. 调度发车预算从 `int(available * action)` 改为 `round` + 最小保留策略，减少截断归零。
  2. 充电分配从“按区硬阈值映射”改为“全局充电预算 + 低SOC优先 + 跨区回填最近充电桩”。
  3. 充电车辆写入 `current_step_logs`，避免日志长期 `N/A/-1` 难以诊断。
  4. 后处理阶段避免用普通订单停车位覆盖本步已分配的充电任务。
  5. 终止步奖励与中间步统一：`(service_revenue - soc_penalty)/10 + charging_bonus`，并统一 `charging_bonus=3000*count`。

**目标**:
* 降低后期动作塌缩概率。
* 提高充电决策可执行率与日志可观测性。
* 保持训练/终止步奖励一致性，减少学习信号冲突。

### 2.4 [2026-02-12] 后期零动作架构修复（可调度口径统一）
**问题现象**: 后期需求仍存在，但 RL 在若干时间步出现服务/调度/充电全零。

**代码修复点**:
* `specific/dclily_elec.py`
  1. 新增 `_compute_dispatchable_counts()` 与 `_sync_dispatchable_available()`，统一可调度车辆定义为 `remain==0 && charger_id==-1`。
  2. `step()` 动作解码预算改为使用 `_compute_dispatchable_counts()`，不再直接依赖可能漂移的 `available[0]`。
  3. `check_idle_vehicles()` 仅对“可调度且未占桩”车辆分配充电任务，避免已充电车辆重复参与分配。
  4. 在动作解码前与充电分配后同步 `available[0]`，减少预算漂移导致的后期全零。

**目标**:
* 降低后期“有需求但无动作”的结构性概率。
* 避免充电侧对调度预算的重复侵蚀。

### 2.5 [2026-02-12] 充电决策落地一致性修复（动作记账对齐）
**问题现象**: `Decision_Charge=True` 数量较高，但 `charging_actions` 统计偏低或为0。

**代码修复点**:
* `specific/dclily_elec.py`
  1. `find_preferred_charger()` 改为基于 `charging_stations[*].occupied_by` 判断桩可用性，不再用 `available[1] > slot_id` 代理判定。
  2. `check_idle_vehicles()` 中充电分配仅在 `assign_parking_slot(..., True)` 成功拿到 `charger_id` 后计入 `charging_actions` 与 `decision_charge=True`。
  3. 充电桩分配失败时回滚车辆状态，避免“决策为真但动作未落地”。
  4. 车辆充满电后自动释放充电桩（`charger_id=-1`），避免桩长期占用导致后续无法充电。

**目标**:
* 让“充电决策日志”与“充电动作统计”保持一致。
* 提升后期可用充电桩周转率。

### 2.6 [2026-02-12] 训练稳定性修复（奖励收敛）
**问题现象**: 训练奖励长期高波动，Critic 值域过大，收敛慢。

**代码修复点**:
* `core/training.py`
  1. `algo_name == SAC` 时不再叠加 OU 噪声。
  2. `algo_name == SAC` 时禁用前20轮充电维度额外噪声。
  3. 回放入池奖励使用 `train_reward = reward * reward_scale`（默认 0.01）。
* `core/util.py`
  1. 新增 `reward_scale` 参数（默认 0.01）。
  2. 参数解析改为 `parse_args()`，支持命令行覆盖。
* `specific/dclily_elec.py`
  1. 训练步与终止步 SOC 惩罚改为固定参数（`penalty_scale=0.85`, `vehicle_penalty=420`），移除分段 curriculum 的非平稳目标。

**目标**:
* 降低策略目标漂移。
* 缩小 Q/TD 目标尺度，提升训练稳定性。

### 2.7 [2026-02-12] 动作对齐最优解（统计口径 + 发车预算）
**问题现象**: RL 动作统计中的充电数量与最优解口径不一致，且服务/调度总量偏低。

**代码修复点**:
* `specific/dclily_elec.py`
  1. 新增 `_collect_active_charging_actions()`，按“当前真实在充电车辆数（zone聚合）”输出 `charging_actions`，与最优解的每步充电动作口径对齐。
  2. 发车预算从 `available * action` 改为需求感知映射：
     - `depart_ratio = 0.2 + 0.8 * action[i]`
     - `limit_now = max(round(available*depart_ratio), round(0.25*out_demand_now))`
     - 再裁剪到 `[0, available]`

**目标**:
* 提升 RL 与最优解在动作统计上的可比性。
* 提高服务/调度动作激活率，降低欠激活。

### 2.8 [2026-02-12] 服务优先优化（冲刺85%）
**问题现象**: 调度动作接近最优后，服务动作与充电决策日志仍不足，限制最终Reward提升。

**代码修复点**:
* `specific/dclily_elec.py`
  1. 在每个时间步开始时为所有车辆初始化 `current_step_logs`，默认 `decision_charge=False`，避免 `Decision_Charge` 出现 `NaN`。
  2. 奖励函数新增服务优先项：`service_bonus = 1200 * served_count`（`served_count = sum(rel_demand)`）。
  3. 终止步奖励与中间步保持一致，同步加入 `service_bonus`。

**目标**:
* 提升服务订单执行量，缩小 RL 与最优解在服务维度的差距。
* 提高日志可观测性，便于后续多种子筛选。
