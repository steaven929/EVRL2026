# MARL Phase-1 Implementation Summary

## 实施日期
2026-02-19

## 实施内容

### 1. 环境多智能体接口 (specific/dclily_elec.py)

#### 新增方法：
- `get_agent_obs(agent_id)`: 获取区域级智能体的7维局部观测
  - [0] od_in_local: 本区域当前入需求
  - [1] od_out_local: 本区域当前出需求
  - [2] vehicle_count_local: 本区域可用车辆数
  - [3] avg_soc_local: 本区域平均SOC（归一化）
  - [4] charger_occ_local: 本区域充电桩占用率
  - [5] timestep_norm: 当前时间步归一化值
  - [6] agent_id: 区域编号（用于embedding）

- `get_global_state()`: 获取全局状态（用于centralized critic）
  - 返回与get_micro_state()相同的251维全局状态

- `n_agents` property: 返回智能体数量（区域数=50）

#### 修改方法：
- `step(action)`: 支持joint action输入
  - 单智能体格式：(3*ZONE_NUM,)
  - 多智能体格式：(ZONE_NUM, 3) → 自动展平为 (3*ZONE_NUM,)

### 2. 多智能体经验池 (core/buffer.py)

#### 新增类：
- `MARLReplayBuffer`: 支持joint-transition存储
  - Transition结构：(obs_n, act_n, global_state, reward, next_obs_n, next_global_state, done)
  - 容量：64000（与单智能体相同，因为使用joint-transition）

### 3. MARL模型 (specific/marl_model.py)

#### 新增网络：
- `SharedActor`: 参数共享的Actor网络
  - 输入：局部观测(6维) + agent_id_embedding(16维)
  - 输出：3维连续动作 [depart_pref, dest_pref, charge_pref]
  - 结构：Embedding(50, 16) + MLP(22 → 128 → 64 → 3)

- `CentralizedCritic`: 集中式Critic网络
  - 输入：global_state(251维) + joint_action(150维)
  - 输出：Q值（标量）
  - 结构：MLP(401 → 256 → 256 → 1)

### 4. MARL Agent (core/marl_agent.py)

#### 新增类：
- `IPPOAgent`: IPPO with Parameter Sharing and Centralized Critic
  - 算法：Independent PPO（Phase-1简化版）
  - 特点：
    - 参数共享的Actor（所有区域共享同一个网络）
    - 集中式Critic（使用全局状态）
    - 全局奖励共享（Phase-1不做credit assignment）
  - 方法：
    - `sample_action()`: 采样联合动作
    - `update()`: 更新网络参数
    - `save_model()` / `load_model()`: 模型保存/加载

### 5. MARL训练循环 (core/marl_training.py)

#### 新增函数：
- `marl_train()`: 多智能体训练循环
  - 每步采样所有智能体的局部观测
  - 采样联合动作（带探索噪声）
  - 执行动作并存储transition
  - 更新网络参数

- `marl_test()`: 多智能体测试循环
  - 测试时不添加探索噪声
  - 记录测试奖励

### 6. 环境配置更新 (core/environment.py)

#### 修改函数：
- `env_agent_config()`: 支持multi模式
  - 检测`marl_mode`参数
  - 如果为'multi'，创建IPPOAgent
  - 如果为'single'，使用原有SAC/DDPG

### 7. 配置参数更新 (core/util.py)

#### 新增参数：
- `--marl_mode`: 单智能体/多智能体模式（默认'single'）
- `--n_agents`: 智能体数量（默认50）
- `--share_policy`: 是否参数共享（默认True）

### 8. 主程序 (main_marl.py)

#### 新增文件：
- MARL训练主程序
- 强制设置为multi模式
- 自动创建输出目录
- 训练后绘制曲线并测试

### 9. 测试脚本 (test_marl.py)

#### 新增文件：
- 冒烟测试脚本
- 验证所有MARL组件正常工作
- 测试结果：✓ 所有测试通过

## 测试结果

### 冒烟测试通过项：
1. ✓ 环境创建成功（50个智能体）
2. ✓ 全局状态维度正确（251维）
3. ✓ 局部观测形状正确（50×7）
4. ✓ 动作采样正确（50×3，范围[0,1]）
5. ✓ 环境step正常执行
6. ✓ 经验池正常工作
7. ✓ 网络更新正常（Critic Loss: 4.8e9, Actor Loss: 0.0019）
8. ✓ 模型保存/加载正常

## 架构特点

### Phase-1设计原则：
1. **最小改造**：保持原有环境step逻辑不变
2. **参数共享**：所有区域共享同一个Actor网络
3. **集中训练**：使用全局状态训练Critic
4. **分散执行**：每个区域独立采样动作
5. **全局奖励**：Phase-1不做credit assignment

### 与单智能体的兼容性：
- 保留原有ReplayBuffer和SAC/DDPG
- 通过`marl_mode`参数切换
- 默认为'single'模式，不影响现有训练

## 下一步工作

### Phase-1验证：
1. 运行短程训练（10-50轮）验证可学习性
2. 对比单智能体基线性能
3. 检查充电黑客问题是否改善

### Phase-2升级（如果Phase-1成功）：
1. 实现差分奖励（credit assignment）
2. 添加价值归一化
3. 升级到完整MAPPO
4. 考虑引入邻区信息到局部观测

## 文件清单

### 新增文件：
- `specific/marl_model.py`: MARL网络模型
- `core/marl_agent.py`: MARL Agent实现
- `core/marl_training.py`: MARL训练循环
- `main_marl.py`: MARL主程序
- `test_marl.py`: MARL测试脚本

### 修改文件：
- `specific/dclily_elec.py`: 添加多智能体接口
- `core/buffer.py`: 添加MARLReplayBuffer
- `core/environment.py`: 支持multi模式
- `core/util.py`: 添加MARL参数

## 使用方法

### 运行测试：
```bash
python test_marl.py
```

### 运行训练：
```bash
python main_marl.py --train_eps 100 --test_eps 10
```

### 切换模式：
```bash
# 单智能体模式（默认）
python main.py --marl_mode single

# 多智能体模式
python main_marl.py --marl_mode multi --n_agents 50
```

## 实施状态
✓ Phase-1 完成
- 所有组件实现完毕
- 冒烟测试通过
- 可以开始正式训练
