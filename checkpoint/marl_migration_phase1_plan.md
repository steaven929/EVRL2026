# 多智能体迁移方案（Phase-1 最小改造）

## 1. 目标与边界
- 目标：在不改动 `specific/optim.py` 的前提下，把当前单智能体全局动作控制改为多智能体控制框架。
- 粒度：采用区域级智能体（`n_agents = ZONE_NUM`），不采用车辆级智能体。
- 训练范式：CTDE（集中训练、分散执行）。
- 阶段范围：先完成可运行骨架与数据流打通，再做算法替换和效果优化。

### 1.1 Phase-1.1 补齐目标（本次新增）
- 明确局部观测定义与维度。
- 明确动作拆分与 joint action 映射。
- 锁定 Phase-1 算法路线：先 IPPO（参数共享）再升级 MAPPO。
- 明确参数共享网络骨架与 agent_id 区分机制。
- 明确训练协调策略与评估指标。
- 明确 replay 容量策略（joint-transition 口径）。

## 2. 现状约束（当前代码）
- 单智能体入口：`core/environment.py` 的 `env_agent_config(cfg)` 只返回一个 `agent`。
- 单动作流：`core/training.py` 中每步仅一次 `agent.sample_action(state)`，然后 `env.step(action)`。
- 经验池结构：`core/buffer.py` 仅支持 `(state, action, reward, next_state, done)`。
- 环境动作：`specific/dclily_elec.py` 使用全局动作向量（`3 * ZONE_NUM`）。
- 奖励形态：`specific/dclily_elec.py` 输出全局 `reward_value`。

## 3. 方案对比与结论

### 3.1 推荐方案（Top-1）
- 方案：Parameter Sharing IPPO（Phase-1）→ MAPPO（Phase-2 升级）。
- 原因：
  - 与当前连续动作和全局协调流程兼容。
  - 训练稳定性通常优于 MADDPG/MATD3 路线。
  - 可先保留现有 `env.step` 聚合逻辑，改动面可控。
  - 先用 IPPO 可显著降低首轮实现复杂度。

### 3.1.1 决策说明（消除歧义）
- **Phase-1 固定选择**：IPPO + centralized critic（参数共享 actor）。
- **Phase-1 不做**：完整 MAPPO 价值归一化、差分奖励、反事实基线。
- **升级触发条件**：当 IPPO 在 50 区指标达到或超过单智能体基线后，再升级 MAPPO。

### 3.2 备选方案（Top-2）
- 方案：MATD3（或 MADDPG）。
- 原因：
  - 与现有 SAC/DDPG 心智模型更接近（off-policy）。
  - 连续动作天然支持。
- 风险：多智能体下非平稳性更明显，调参与排障成本更高。

### 3.3 不优先方案
- QMIX/VDN 不作为第一选择：
  - 主流实现偏离散动作，当前项目动作本质是连续偏好向量。
  - 强制离散化会放大动作设计复杂度，并损失控制精度。

## 4. Phase-1 最小改造步骤（按顺序）

### Step 1：环境增加多智能体视图接口
- 文件：`specific/dclily_elec.py`
- 新增接口：
  - `get_agent_obs(agent_id)`：返回区域级局部观测。
  - `get_global_state()`：返回集中 critic 使用的全局状态。
  - `self.n_agents = ZONE_NUM`。
- 要点：先不删除原有全局状态构造逻辑，保持兼容。

#### Step 1.1：局部观测固定为 7 维（强约束）
- 观测向量定义（每个区域 agent 一致）：
  1) `od_in_local`：本区域当前入需求（标量）
  2) `od_out_local`：本区域当前出需求（标量）
  3) `vehicle_count_local`：本区域可用车辆数（标量）
  4) `avg_soc_local`：本区域平均 SOC（标量，建议归一化）
  5) `charger_occ_local`：本区域充电桩占用率（标量）
  6) `timestep_norm`：当前时间步归一化值（标量）
  7) `agent_id`：区域编号（用于 embedding，不直接归一化）
- 目标形状：`obs_i.shape == (7,)`。
- 备注：邻区信息不进入 Phase-1 局部观测，避免先期维度膨胀。

### Step 2：保持单 `step`，支持 joint action 输入
- 文件：`specific/dclily_elec.py`
- 目标：`step` 支持接收 `[n_agents, agent_action_dim]` 或可等价重排的一维 joint action。
- 要点：内部拼接回当前全局动作解码流程；`solveproblem` 调用链保持不变。

#### Step 2.1：动作映射固定规则（强约束）
- 每个区域 agent 输出 `3` 维连续动作：
  1) `depart_pref_i`
  2) `dest_pref_i`
  3) `charge_pref_i`
- 因此：`agent_action_dim = 3`。
- joint action 组装规则：
  - 输入若为二维：`joint_action.shape == (n_agents, 3)`；
  - 映射为一维全局向量：`global_action = joint_action.reshape(-1)`；
  - 目标全局形状：`(3 * ZONE_NUM,)`，与当前环境动作完全兼容。
- 顺序约定：按 `agent_id` 从小到大拼接，保证可逆映射。

### Step 3：训练循环改为多智能体动作采样
- 文件：`core/training.py`
- 目标：
  - 每步按 agent 采样动作 `act_i`。
  - 组装 `joint_action` 后调用 `env.step(joint_action)`。
- 奖励策略：Phase-1 使用全局 reward 共享给所有 agent（先打通，不先做 credit assignment）。

#### Step 3.1：训练协调策略（新增）
- 探索：每个 agent 采样独立噪声，但噪声方差共享同一调度器（避免区域间探索尺度失衡）。
- 更新：共享 actor 同步更新（单个共享网络），centralized critic 每个 step 更新一次。
- 通信：Phase-1 不引入显式通信通道，依靠全局 critic 隐式协调。
- 回合结束：保持现有 done 逻辑，不引入 agent-level done。

### Step 4：经验池升级为多智能体结构
- 文件：`core/buffer.py`
- 目标 transition 结构：
  - `(obs_n, act_n, global_state, reward, next_obs_n, next_global_state, done)`。
- 要点：保留旧接口兼容单智能体模式（避免一次性切换失败）。

#### Step 4.1：容量策略（修正）
- Phase-1 采用 **joint-transition** 入池：每个环境 step 只入池 1 条联合样本。
- 因此默认 `memory_capacity=64000` 可先保持不变。
- 仅当后续改为“每 agent 单独入池”时，才需要按 `n_agents` 放大容量。

### Step 5：agent 工厂增加 `marl_mode`
- 文件：`core/environment.py`
- 目标：
  - `single`：沿用现有 SAC/DDPG。
  - `multi`：返回多智能体 agent（参数共享 actor + centralized critic）。

#### Step 5.1：参数共享网络骨架（最小实现）
- Shared Actor 输入：`[obs_local(去掉agent_id) + agent_id_embedding]`。
- 建议结构：
  - `Embedding(n_agents, 16)`
  - `MLP(6+16 -> 128 -> 64 -> 3)`
  - 输出 `sigmoid` 到 `[0,1]`。
- Centralized Critic 输入：
  - `global_state` + `joint_action`（或拼接后的 `obs_n, act_n`）。
- 说明：此处是最小可运行骨架，不引入注意力/图网络。

### Step 6：配置参数与主入口兼容
- 文件：`core/util.py`, `main.py`
- 新增参数建议：
  - `--marl_mode`（single/multi）
  - `--share_policy`（bool）
  - `--n_agents`（默认 `ZONE_NUM`）
- 要点：默认仍为 single，确保主线训练可继续跑。

### Step 7：验证与回归
- 阶段验证顺序：
  1) 4 区 `train_eps=1, test_eps=1, max_steps=1` 冒烟。
  2) 50 区同样冒烟。
  3) 再进行短程训练并检查 `rewards/svc/rld` 曲线是否可学习。
- 验证关注点：维度一致性、joint action 解码正确性、无越界与无空样本崩溃。

#### Step 7.1：评估指标补齐（新增）
- 基础性能：`reward_mean`, `svc_mean`, `rld_mean`, `srd_mean`。
- 充电行为：`charging_count`, `full_soc_charging_count`, `low_soc_charging_count`。
- 多智能体特有：区域动作方差、区域策略熵、区域负载均衡度（可先用车辆数方差近似）。
- 对比基线：必须与单智能体最新稳定模型同数据集、同 episode 设置做对照。
- 成功判据（Phase-1）：
  - 不崩溃 + 指标可重复；
  - `svc_mean` 不低于单智能体基线的 90%；
  - 充电黑客指标不劣化（`full_soc_charging_count` 不显著上升）。

## 5. 风险清单
- 风险1：局部观测设计不当导致策略退化（信息不足或冗余过高）。
- 风险2：全局 reward 共享可能引发信用分配困难（Phase-2 再处理）。
- 风险3：多智能体 replay 数据组织错误导致训练不稳定。
- 风险4：模式切换（single/multi）兼容性问题导致历史实验不可复现。

### 5.1 充电黑客前置风险（新增）
- 说明：多智能体不会自动修复奖励函数缺陷。
- 约束：在 MARL 正式长训前，必须先用单智能体完成“充电行为体检”基线记录：
  - `full_soc_charging_count`
  - `charging_count`
  - `svc_mean`
- 若存在“高充电低服务”明显失衡，优先修 reward，再做 MARL 长训。

## 6. Phase-2 预留方向（不在本阶段实现）
- 从全局 reward 共享升级到差分奖励或 counterfactual baseline。
- 引入更稳定的 centralized critic 结构（如双 critic、价值归一化）。
- 评估是否迁移到 PettingZoo/RLlib 生态（在骨架稳定后）。

### 6.1 具体化预留（新增）
- 差分奖励原型：`D_i = R_global - R_without_agent_i`（近似可用局部剔除估计）。
- MAPPO 升级点：价值归一化、GAE 参数单独调优、全局状态归一化。
- 框架迁移触发：仅当自研骨架稳定后再评估 PettingZoo/RLlib，避免双变量同时变更。

## 7. 执行准则
- 优先保障现有训练链路可回滚。
- 每步改动后先做 1-step 冒烟，再进入下一步。
- 仅在 Phase-1 骨架跑通后，才进入算法层超参数优化。
