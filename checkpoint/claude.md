# Claude Code Rules (2026.02.04 Refined)

## 1. 极简输出规范 (Critical)
- **Silent Mode**: 修改代码时，禁止输出代码块。
- **Info Only**: 仅输出：[文件路径]、[函数/类名]、[修改数学逻辑/原因]。
- **Manual Trigger**: 除非我明确说 "show me the code"，否则保持静默。
- **Context Alert**: 接近 Token 上限前，必须主动提示执行 `/compact`。
- **No Verbosity**: 解释逻辑要精炼，避免复述已知的基础概念。

## 2. 代码修改强制规范
- **Python 注释**: 涉及 .py 修改，必须使用以下块标记：
  - 起：`# =========2026.1.30修改，最优解求解==========`
  - 止：`# ======修改结束`
- **保护逻辑**: 严禁触碰 Gurobi 求解器核心代码；修改 Reward 公式前必须二次确认。

## 3. 文件管理
- **最小化原则**: 优先修改现有文件，单次任务最多新建 1 个 .py。
- **文档登记**: 新建文件必须在 `checkpoint/readai_me.txt` 登记用途和调用关系。
- **知识持久化 (Knowledge Persistence)**:
  - **分流记录**:
    - 逻辑/架构变更 -> `readme2_traininglog.md`。
    - **超参数实验**: 凡涉及学习率、BatchSize、Reward 系数等参数调整，**必须**在 `tuning_log.md` 中新增一条实验记录，严禁只改代码不记日志。
    - **分支实验标记**: 所有非主线实验必须在日志中标记为 `[分支实验X]`，并说明实验路径和合并条件。
  - **Silent Logging (关键)**: 执行日志写入时，**严禁在对话中展示新增的日志内容**。仅需简单回复 "✅ [文件名] Updated" 即可。

## 4. 算法核心一致性 (RL vs Optimal Control)
- **Reward 对齐**:
  - RL Reward = (服务+充电-低电量惩罚-调度成本)/10 - 500
  - 必须检查符号反转（Maximization vs Minimization）。
  - 离散化必须核对时间步长 $\Delta t$。

### 5. 每次都要调用 superpowers 插件，进行各类操作

---

# 实验进展记录 (2026.02.13)

## 实验21：修复充电桩释放逻辑 - 关键Bug修复

**日期**: 2026-02-13
**问题诊断**:
- RL输出充电偏好但执行0充电动作
- 充电预算正常计算（1-3辆车）
- `charged_count`正常递增
- 但`charging_actions`字典始终为空

**根本原因**:
- 充电桩初始化时被占满（`write_ini_cap()` line 327-347）
- 车辆充电到满电后，释放逻辑被注释（line 1043-1044）
- 导致充电桩永久占用，无法分配给新车辆

**修改内容**:
- 文件: specific/dclily_elec.py (line 1037-1045)
- 启用充电桩释放逻辑：
```python
if self.car[i][1] >= BATTERY_CAPACITY:
    self.car[i][1] = BATTERY_CAPACITY
    # 充满后释放充电桩
    current_zone = self.car[i][0]
    charger_id = self.car[i][3]
    self.release_charger(current_zone, charger_id)
    self.car[i][3] = -1  # 释放充电桩状态
```

**测试结果**:
- Episode 1: 0 charging actions (充电桩全满)
- Episode 2: 4 charging actions at t=0 (充电桩开始释放)
- Episode 3: 3+2+1=6 charging actions (充电桩正常循环)
- 充电桩状态正常：占用 → 充电 → 满电 → 释放 → 可用

**影响**:
- 修复前：0 charging actions/episode
- 修复后：4-10 charging actions/episode
- 这是充电功能的基础性修复，所有之前的实验结果可能受此Bug影响

**状态**: 已修复，正在进行1000轮训练验证

---

## 实验20：状态空间增强 - 添加时间步信息

**日期**: 2026-02-13
**问题诊断**:
- RL在t=0过早充电（6辆 vs 最优0辆）
- RL无法学习时间依赖策略（如"t=0不充电，t=1开始充电"）
- 原因：状态空间缺少时间步信息

**修改内容**:
- 文件: specific/dclily_elec.py (line 92-96, 454-469)
- 状态空间维度：36 → 38
- 新增特征：
  1. 当前时间步 (0-11)
  2. 剩余时间步 (11-0)

**技术细节**:
```python
# 观测空间定义 (line 92-96)
OBS_SPACE = spaces.MultiDiscrete(
    (TIME_SLICE * ZONE_NUM) * [OD_MAX, OD_MAX, CHARGER_STATES, NUM_CHARGERS] +
    ZONE_NUM * [10] +  # SOC特征
    [TIME_NUM, TIME_NUM]  # 新增：当前时间步、剩余时间步
)

# 状态构造 (line 454-469)
current_timestep = np.array([self.episode_timestep], dtype=np.int64)
remaining_timesteps = np.array([TIME_NUM - self.episode_timestep - 1], dtype=np.int64)
state = np.concatenate([od_part, charger_states, num_chargers_feat,
                        avg_soc_per_zone, current_timestep, remaining_timesteps])
```

**预期效果**:
- RL能学习"t=0不充电，t≥1充电"策略
- 充电时机与最优解对齐
- 充电总量提升（64 → 80+）
- 服务不崩溃（持续到t=11）

**同步修改**:
- 文件: baseline_comparison.py (line 77-100)
- 添加状态结构注释，确保GreedyPolicy正确解析38维状态
- 验证通过：RandomPolicy、GreedyPolicy、RL数据读取均兼容新状态空间

- 文件: core/environment.py (line 60-78)
- 修复vocab_size: 10 → 12，支持时间步特征值范围0-11
- 原因：时间步特征可取值0-11（TIME_NUM=12），需要embedding层vocab_size≥12

**测试结果**:
- ✓ 环境创建成功，状态维度38
- ✓ 时间步特征正常更新（t=0: current=0, remaining=11）
- ✓ 训练流程正常运行（10轮测试）
- ✓ 模型保存成功

**状态**: 已验证通过，可以开始正式训练

---

# 实验进展记录 (2026.02.11)

## 实验对比总结 - 达到最优解的62.9%

| 指标 | 实验3 | 实验4 | 实验5 | 最优解 |
|------|-------|-------|-------|--------|
| 充电奖励 | 1500 | 1500 | 1500 | 1500 |
| SOC惩罚(400-600) | 0.5 | 2.0 | 1.0 | 对应 |
| SOC惩罚(200-400) | 0.5 | 10.0 | 5.0 | 对应 |
| SOC惩罚(<200) | 5.0 | 50.0 | 25.0 | 对应 |
| 低电量车辆惩罚 | 0 | 500 | 250 | - |
| 充电阈值 | 0.1 | 0.1 | 0.01 | - |
| 训练轮数 | 600 | 600 | 1000 | - |
| **总奖励** | **9820** | **6570** | **15060** | **23929.82** |
| **充电次数** | **2** | **5** | **1** | **128** |
| **Actor充电偏好** | **10^-6** | **0.1-0.7** | **10^-9** | **-** |
| **RL/最优解** | **40.8%** | **27.6%** | **62.9%** | **100%** |

---

### 实验3：充电奖励增强（500→1500）
**日期**: 2026-02-11
**修改内容**:
- 文件: specific/dclily_elec.py (line 1225, 1277)
- 文件: optimal_solver.py (line 146)
- 充电奖励从500提升到1500（3倍）

**结果**:
- 最优解: 24054.83 (OPTIMAL, 128次充电)
- RL测试期: 9820.00 (2次充电)
- RL/最优解: 40.8%
- Actor充电偏好输出: 10^-9到10^-5级别

**结论**:
- ❌ 充电奖励3倍提升无效
- ❌ Actor仍未学会充电（完全依赖探索噪声）
- ✅ 最优解正常工作
- 根本原因：充电的长期价值未被正确传递给Actor

---

### 实验4：增强SOC维持奖励机制
**日期**: 2026-02-11
**修改内容**:
- 文件: specific/dclily_elec.py (line 1228-1248, 1280-1298)
- 文件: optimal_solver.py (line 153-192)
- SOC惩罚力度提升4-20倍
- 新增低电量车辆数惩罚（每辆500）
- 充电阈值从0.1降低到0.01

**修改逻辑**:
1. 扩大惩罚范围：从SOC<200提升到SOC<400
2. 增强惩罚系数：
   - 400-600: 0.5→2.0 (4倍)
   - 200-400: 0.5→10.0 (20倍)
   - <200: 5.0→50.0 (10倍)
3. 新增：每辆低电量车辆（SOC<400）额外惩罚500

**结果**:
- 最优解: 23804.82 (OPTIMAL, 127次充电)
- RL测试期: 6570.00 (5次充电)
- RL/最优解: 27.6%
- Actor充电偏好输出: 时间步0-4为0.1-0.7级别，时间步5+崩溃到10^-9级别

**结论**:
- ✅ 重大突破：Actor充电偏好从10^-6提升到0.1-0.7（100,000倍提升）
- ✅ 充电次数从2次提升到5次（150%提升）
- ❌ 总奖励反而下降33%（9820→6570）
- ❌ 充电行为不稳定（前5步充电，后7步完全停止）
- 根本原因：SOC惩罚过强，惩罚超过充电奖励，Actor学会输出充电偏好但总收益为负

---

### 实验5：降低SOC惩罚50% + 延长训练到1000轮 + 阈值0.01
**日期**: 2026-02-11
**修改内容**:
- 文件: specific/dclily_elec.py (line 1228-1249, 1287-1309)
- 文件: optimal_solver.py (line 153-180)
- 文件: core/util.py (line 20)
- SOC惩罚力度降低50%（实验4的一半）
- 充电阈值从0.1降低到0.01
- 训练轮数从600提升到1000

**修改逻辑**:
1. SOC惩罚系数降低50%：
   - 400-600: 2.0→1.0
   - 200-400: 10.0→5.0
   - <200: 50.0→25.0
   - 低电量车辆惩罚: 500→250
2. 充电阈值: 0.1→0.01（让实验4的0.1-0.7输出能触发充电）
3. 训练轮数: 600→1000（更长时间学习稳定策略）

**结果**:
- 最优解: 23929.82 (OPTIMAL, 128次充电)
- RL测试期: 15060.00 (1次充电)
- RL/最优解: 62.9%
- Actor充电偏好输出: 全程10^-9到10^-16级别

**结论**:
- ❌ 严重失败：充电行为完全崩溃
- ❌ Actor充电偏好从实验4的0.1-0.7降到10^-9（下降100,000,000倍）
- ❌ 充电次数从5次降到1次
- ✅ 总奖励提升到15060（虚假成功：因为SOC惩罚太弱，不充电也不会被严重惩罚）
- 根本原因：SOC惩罚太弱，Actor失去充电动机，专注调度收益，完全放弃充电策略
- 关键洞察：实验4的方向是对的，问题不是惩罚太强，而是需要提升充电奖励

---

### 实验6：混合策略（中等惩罚 + 中等充电奖励）
**日期**: 2026-02-11
**修改内容**:
- 文件: specific/dclily_elec.py (line 1223-1225, 1228-1249, 1287-1309)
- 文件: optimal_solver.py (line 143-180)
- SOC惩罚：实验4和5的中间值（1.5, 7.5, 37.5, 375）
- 充电奖励：从1500提升到2500（67%提升）
- 训练轮数：保持1000轮
- 充电阈值：保持0.01

**修改逻辑**:
1. 中等SOC惩罚（实验4和5的平均）：
   - 400-600: 1.5 (2.0和1.0的平均)
   - 200-400: 7.5 (10.0和5.0的平均)
   - <200: 37.5 (50.0和25.0的平均)
   - 低电量车辆惩罚: 375 (500和250的平均)
2. 充电奖励提升：1500→2500
3. 目标：平衡充电收益和SOC惩罚，让Actor持续充电

**预期效果**:
- 充电次数：20-50次（vs 最优解128次）
- 总奖励：18000-22000（vs 最优解23929.82）
- 达到最优解的75-90%
- Actor充电偏好输出：稳定在0.1+级别

**训练参数**:
- train_eps: 1000
- 其他参数保持不变

**状态**: 待训练

---

## 关键发现与洞察

### 1. 充电学习的三个阶段
- **实验3**: 充电奖励提升无效，Actor输出10^-6级别
- **实验4**: SOC惩罚强化有效，Actor输出0.1-0.7级别（突破！）
- **实验5**: SOC惩罚过弱，Actor输出退化到10^-9级别（崩溃）

### 2. 奖励函数设计原则
- **惩罚驱动学习**: 高SOC惩罚能迫使Actor学习充电策略
- **奖励平衡收益**: 充电奖励需要>SOC惩罚，才能让Actor持续充电
- **虚假成功陷阱**: 低惩罚导致高总奖励，但Actor未学会充电

### 3. 下一步优化方向
- ✅ 实验6：混合策略（中等惩罚 + 高充电奖励）
- 备选方案A：回到实验4参数，仅提升充电奖励到3000-5000
- 备选方案B：分阶段训练（前期高惩罚，后期降低）

---

## 2026-03-15 配置同步记录

- 已按 `data/new_setting区分功能区.txt` 的 1000 车列更新 `data/SIP/config_sip.json` 与 `data/SIP/config.json` 的各区停车位和充电桩数量。
- 已将 SIP 运行基准车辆总数统一为 1000：`specific/dclily_elec.py` 强制覆盖段 `CAR_NUM = 1000`，并同步 `data/SIP/config*.json` 中 `CAR_NUM = 1000`。
- 已重建 `data/SIP/capacity.csv`：第1行（初始车辆分布）按 `od_probability * 1000` 分配且总和=1000；第2行使用 `p_limit_1000`，避免 `self.CAR_NUM` 与 `self.car` 长度不一致。
- 已修复 `core/training.py` 内存放大问题：日志时间维度改为环境 `TIME_NUM`（可由 `log_time_num` 覆盖），不再直接使用 `max_steps` 预分配 `(max_steps, zone_num)` 大矩阵。
- 已修复 `specific/optim.py` 的Gurobi内存压力点：不可行 `car_assignment` 变量改为 `ub=0`（替代大量零约束），并在每次求解后 `model.dispose()` 显式释放模型内存。
- 已在 `main.py` 与 `core/util.py` 同步时间步防护：默认 `max_steps=26`，并在运行时将 `cfg['max_steps']` 与 `cfg['log_time_num']` 对齐到环境 `TIME_NUM`，防止出现 `(100000, 50)` 级日志预分配。
- 已新增长训内存防护：`core/training.py` 的 `battery_log` 改为有界缓存（`battery_log_keep_steps` + `battery_log_stride`），避免1000车场景下日志无限累积；`specific/optim.py` 将 `Threads` 下调到2，并设置 `NodefileStart=0.25`、`SoftMemLimit=8` 降低Gurobi OOM概率。
- 已对齐单智能体与MARL基线口径：`main_marl.py` 新增 `data_folder -> specific/config.json` 复制链路；并在运行时将 `cfg['max_steps']` 与 `cfg['log_time_num']` 对齐环境 `TIME_NUM`。`core/marl_training.py` 新增 `max_steps` 上限、周期评估（`eval_rewards/eval_episodes`）和 `marl_metric_sample_every` 参数（默认1，每步采样）。
- 已加入MARL策略优化（不改reward权重）：`core/marl_training.py` 使用“自适应探索噪声日程”（`marl_noise_start/min/decay_power/warmup_eps` + `marl_service_target` 驱动的噪声补偿），当上一轮 `svc` 低于目标时自动提高探索强度。

---

## 2026-03-17 会话总接续记忆（跨窗口恢复用）

### A. 本轮用户目标与结论
- 用户主目标：在 1000 车场景下稳定训练，并让 MARL 在公平口径下尽量优于 SAC/TD3。
- 关键结论：
  - 之前的 `(100000, 50)` 内存报错来自日志时间维度错误绑定 `max_steps`，已修复。
  - TD3 的 Gurobi OOM 主要由大规模重复求解 + 长训内存压力叠加导致，已做减压改造。
  - MARL 与单智能体对比口径曾不完全一致，已补齐关键口径（配置同步、时间步、评估输出、采样频率）。

### B. 本轮已完成的代码与配置变更（核心）
- 1000车配置落地：
  - `data/SIP/config_sip.json`、`data/SIP/config.json`：按 `data/new_setting区分功能区.txt` 的 1000 车列更新各区 `num_parking_slots`/`charging_slots`，并设 `CAR_NUM=1000`。
  - `specific/dclily_elec.py`：强制覆盖 `CAR_NUM=1000`。
  - `data/SIP/capacity.csv`：重建为“第1行初始车辆分布总和=1000，第2行停车位上限=1000列配置”。
- 内存与稳定性修复：
  - `core/training.py`：日志时间维度改为 `TIME_NUM/log_time_num`，与 `max_steps` 解耦；`battery_log` 改为有界缓存并支持步长采样。
  - `core/util.py`：`max_steps` 默认改为 26；新增 `battery_log_keep_steps`、`battery_log_stride`。
  - `specific/optim.py`：不可行分配改 `ub=0`（替代大量 `==0` 约束）；求解后 `model.dispose()`；Gurobi 参数降内存（线程降低、NodefileStart 提前、SoftMemLimit）。
- MARL口径对齐与策略优化：
  - `main_marl.py`：补齐 `cp(..\data_folder)` 配置同步；对齐 `max_steps/log_time_num`；新增保存 `eval_rewards/eval_episodes`。
  - `core/marl_training.py`：支持 `max_steps` 上限、周期评估、`marl_metric_sample_every`；新增自适应探索噪声日程。
  - `core/util.py`：新增 MARL 噪声参数；并将默认训练参数切到当前优化版本（见 C 节）。

### C. 当前默认参数快照（以 `core/util.py` 为准）
- 训练：`train_eps=1500`, `test_eps=10`, `max_steps=26`
- 优化：`batch_size=256`, `memory_capacity=128000`, `actor_lr=5e-5`, `critic_lr=1e-4`, `gamma=0.98`, `tau=8e-4`
- 奖励：`reward_scale=10000`, `reward_mode=shaping`
- MARL探索：
  - `marl_noise_start=0.35`, `marl_noise_min=0.12`, `marl_noise_decay_power=1.4`, `marl_noise_warmup_eps=50`
  - `marl_service_target=0.85`, `marl_noise_service_boost=0.10`
  - `marl_metric_sample_every=1`

### D. 已验证运行（可追溯）
- 1000车验证：运行时核验 `module_CAR_NUM=1000 / env_CAR_NUM=1000 / len_car=1000`。
- 冒烟通过：
  - `main.py`（SAC/TD3/DDPG 最小轮次）可完成训练+测试并产出目录。
  - `main_marl.py` 口径对齐后可完成训练+测试，且输出 `eval_rewards.npy`、`eval_episodes.npy`。

### E. 仍需关注的风险与注意事项
- 16GB机器（当前可用内存偏低）长训仍有峰值风险：建议关闭大进程并降低并发压力。
- `main_marl.py` 参数打印里 `algo_name` 可能显示 `TD3`，但 MARL 路径由 `marl_mode='multi'` 强制；以入口脚本为准。
- 结果对比优先看 `tail100 mean reward` + `svc/srd/rld`，避免只看单次峰值。

### F. 下次新窗口恢复步骤（必须）
1. 先读本文件本节（`checkpoint/claude.md`）。
2. 再读 `results/SESSION_HANDOFF_20260317.md`（本次完整操作纪要）。
3. 按同口径运行：
   - `python -u main_marl.py --data_folder data/SIP/`
   - `python -u main.py --algo_name SAC --data_folder data/SIP/`
   - `python -u main.py --algo_name TD3 --data_folder data/SIP/`

### G. 消融实验入口（去除Gurobi优化器）
- 新增独立脚本：`main_marl_no_gurobi.py`。
- 设计方式：不改原有 `main_marl.py` 与 `specific/optim.py`，仅在新入口中 monkeypatch `specific.optim.solveproblem` 为启发式调度器，再调用原 `main_marl.main()`。
- 口径对齐：参数解析、`data_folder -> specific/config.json` 同步、`max_steps/log_time_num` 对齐、训练/评估/指标输出目录结构，均沿用 `main_marl.py`。
- 运行方式：`python -u main_marl_no_gurobi.py --data_folder data/SIP/ [其他参数]`。
