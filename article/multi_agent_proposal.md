# 多智能体RL改造方案

## 当前架构问题回顾

**单智能体架构：**
- 1个智能体控制50个区域的所有决策
- 动作空间：150维（50区域 × 3决策）
- 状态空间：251维（全局状态）
- **核心问题**：智能体学会"充电黑客"策略，所有区域都执行充电

## 方案对比

### 方案1：车辆级多智能体 (Vehicle-Level MARL)

**架构：**
```
200个独立智能体（每辆车一个）
  ↓
每个智能体观察：
  - 自身状态（位置、SOC、服务状态）
  - 局部环境（当前区域的OD需求、充电桩状态）
  - 邻近车辆信息（可选）
  ↓
每个智能体输出：
  - 是否接单（0/1）
  - 目标区域（50选1）
  - 是否充电（0/1）
```

**优点：**
- ✓ 最真实的建模方式
- ✓ 高度可扩展（车辆数量可变）
- ✓ 自然的分布式决策
- ✓ 每辆车可以学习个性化策略

**缺点：**
- ❌ 训练极其困难（200个智能体同时学习）
- ❌ 需要复杂的通信机制
- ❌ 状态空间爆炸（如果包含其他车辆信息）
- ❌ 信用分配问题严重（哪辆车的贡献？）
- ❌ 收敛速度极慢

**适用场景：**
- 研究型项目
- 有大量计算资源
- 需要高度真实性

**实现难度：** ⭐⭐⭐⭐⭐ (5/5)

---

### 方案2：区域级多智能体 (Zone-Level MARL)

**架构：**
```
50个独立智能体（每个区域一个）
  ↓
每个智能体观察：
  - 本区域状态（OD需求、车辆数、充电桩状态、平均SOC）
  - 邻近区域信息（可选）
  - 全局时间步
  ↓
每个智能体输出：
  - 发车倾向（0-1）
  - 目标区域分布（50维概率分布）
  - 充电倾向（0-1）
```

**优点：**
- ✓ 复杂度适中（50个智能体）
- ✓ 自然的空间分解
- ✓ 可以学习区域特性（商业区 vs 住宅区）
- ✓ 相对容易训练
- ✓ 可以使用参数共享减少复杂度

**缺点：**
- ❌ 仍需协调机制
- ❌ 区域边界问题（车辆跨区域）
- ❌ 训练时间较长

**适用场景：**
- 平衡性能和复杂度
- 区域特性明显的场景
- 中等规模项目

**实现难度：** ⭐⭐⭐ (3/5)

---

### 方案3：混合架构 (Hybrid: Centralized Dispatch + Decentralized Charging)

**架构：**
```
1个全局派车智能体
  ↓
观察全局状态 → 输出派车决策（100维）

+

50个区域充电智能体
  ↓
每个观察本区域状态 → 输出充电决策（1维）
```

**优点：**
- ✓ 针对性解决"充电黑客"问题
- ✓ 保留派车的全局优化能力
- ✓ 充电决策分散，避免全局充电
- ✓ 实现难度适中
- ✓ 训练相对简单

**缺点：**
- ❌ 不是完全的多智能体
- ❌ 派车仍是中心化

**适用场景：**
- 快速解决当前问题
- 保留全局优化能力
- 工程实践项目

**实现难度：** ⭐⭐ (2/5)

---

### 方案4：分层多智能体 (Hierarchical MARL)

**架构：**
```
高层智能体（1个）
  ↓
观察全局状态 → 输出宏观策略（区域目标、充电配额）
  ↓
中层智能体（50个区域）
  ↓
根据宏观策略 + 本地状态 → 输出区域决策
  ↓
低层执行（200辆车）
  ↓
根据区域决策执行具体动作
```

**优点：**
- ✓ 结合全局优化和局部灵活性
- ✓ 层次清晰，易于理解
- ✓ 可以处理不同时间尺度的决策
- ✓ 符合实际调度系统架构

**缺点：**
- ❌ 架构复杂
- ❌ 需要设计层间通信协议
- ❌ 训练需要分阶段进行

**适用场景：**
- 大规模系统
- 需要多时间尺度决策
- 研究型项目

**实现难度：** ⭐⭐⭐⭐ (4/5)

---

## 推荐方案

### 🏆 推荐：方案3（混合架构）+ 方案2（区域级）的渐进式改造

**阶段1：混合架构（快速修复）**

立即实施混合架构解决充电黑客问题：

**实现步骤：**

1. **保留全局派车智能体**
   - 状态：OD需求(100) + 车辆分布(50) + 时间步(1) = 151维
   - 动作：派车决策(100维)
   - 网络：原有Actor-Critic

2. **新增50个区域充电智能体**
   - 状态（每个区域）：
     - 本区域车辆数（1）
     - 本区域平均SOC（1）
     - 本区域充电桩占用率（1）
     - 本区域OD需求（2）
     - 时间步（1）
     - **总计：6维**
   - 动作（每个区域）：充电倾向（1维，0-1）
   - 网络：50个小型Actor-Critic（参数共享）

3. **参数共享策略**
   - 50个区域智能体共享同一套网络参数
   - 输入区域ID作为embedding
   - 大幅减少参数量

**优势：**
- ✓ 快速实施（1-2周）
- ✓ 直接解决充电黑客问题
- ✓ 保留派车的全局优化
- ✓ 训练难度适中

**阶段2：完全区域级多智能体（长期优化）**

在混合架构稳定后，逐步迁移到完全区域级多智能体：

1. 将派车决策也分散到50个区域智能体
2. 使用通信机制（如Graph Neural Network）协调区域间决策
3. 使用QMIX/MAPPO等多智能体算法

---

## 技术实现细节

### 1. 多智能体算法选择

**推荐算法：**

| 算法 | 适用方案 | 优点 | 缺点 |
|------|---------|------|------|
| **MADDPG** | 方案2, 4 | 支持连续动作，中心化训练分散执行 | 训练不稳定 |
| **QMIX** | 方案2, 3 | 价值分解，训练稳定 | 仅支持离散动作 |
| **MAPPO** | 方案1, 2, 4 | 训练稳定，效果好 | 需要大量样本 |
| **CommNet** | 方案1, 2 | 支持智能体通信 | 实现复杂 |
| **参数共享SAC** | 方案3 | 简单有效 | 智能体同质化 |

**针对方案3（混合架构）推荐：参数共享SAC**
- 50个区域充电智能体共享网络参数
- 输入区域ID embedding区分不同区域
- 使用原有SAC框架，改动最小

### 2. 状态空间设计

**方案3（混合架构）状态设计：**

**全局派车智能体：**
```python
state_dispatch = [
    od_in,              # 50维
    od_out,             # 50维
    vehicle_counts,     # 50维（每区域车辆数）
    timestep            # 1维
]  # 总计151维
```

**区域充电智能体（每个）：**
```python
state_charging_zone_i = [
    zone_id,            # 1维（embedding）
    num_vehicles,       # 1维（本区域车辆数）
    avg_soc,            # 1维（本区域平均SOC）
    charger_occupancy,  # 1维（充电桩占用率）
    od_in,              # 1维（本区域入需求）
    od_out,             # 1维（本区域出需求）
    timestep            # 1维
]  # 总计7维
```

### 3. 奖励函数设计

**关键改进：分离派车和充电奖励**

**派车智能体奖励：**
```python
reward_dispatch = (
    service_revenue * 10 +           # 放大服务收入
    distribution_balance_bonus -     # 车辆分布均衡
    fixed_penalty
)
```

**充电智能体奖励（每个区域）：**
```python
reward_charging_zone_i = (
    -charging_cost +                 # 充电成本
    soc_maintain_bonus +             # SOC维持奖励
    -full_battery_charging_penalty + # 满电充电惩罚（新增）
    -charger_waste_penalty           # 充电桩浪费惩罚（新增）
)

# 满电充电惩罚
if avg_soc > 0.9 and charging_action > 0.5:
    full_battery_charging_penalty = 1000

# 充电桩浪费惩罚
if charger_occupancy > 0.8 and num_vehicles < 5:
    charger_waste_penalty = 500
```

### 4. 网络架构

**方案3网络架构：**

```python
# 派车智能体（保持原有）
class DispatchActor(nn.Module):
    def __init__(self, state_dim=151, action_dim=100):
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(state_dim * embed_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_dim)

# 充电智能体（新增，参数共享）
class ChargingActor(nn.Module):
    def __init__(self, state_dim=7, action_dim=1):
        self.zone_embedding = nn.Embedding(50, 16)  # 区域ID embedding
        self.fc1 = nn.Linear(state_dim - 1 + 16, 64)  # 6个特征 + 16维embedding
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)

    def forward(self, state):
        zone_id = state[:, 0].long()
        features = state[:, 1:]
        zone_emb = self.zone_embedding(zone_id)
        x = torch.cat([zone_emb, features], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = torch.sigmoid(self.fc3(x))
        return action
```

### 5. 训练策略

**方案3训练流程：**

```python
# 伪代码
for episode in range(num_episodes):
    state = env.reset()

    for timestep in range(26):
        # 1. 派车决策（全局智能体）
        dispatch_state = extract_dispatch_state(state)
        dispatch_action = dispatch_agent.select_action(dispatch_state)

        # 2. 充电决策（50个区域智能体）
        charging_actions = []
        for zone_id in range(50):
            charging_state = extract_charging_state(state, zone_id)
            charging_action = charging_agent.select_action(charging_state)
            charging_actions.append(charging_action)

        # 3. 合并动作
        full_action = np.concatenate([dispatch_action, charging_actions])

        # 4. 环境交互
        next_state, reward, done, info = env.step(full_action)

        # 5. 分离奖励
        dispatch_reward = calculate_dispatch_reward(info)
        charging_rewards = calculate_charging_rewards(info)  # 50个

        # 6. 存储经验
        dispatch_buffer.push(dispatch_state, dispatch_action, dispatch_reward, ...)
        for zone_id in range(50):
            charging_buffer.push(charging_state[zone_id], charging_actions[zone_id],
                                charging_rewards[zone_id], ...)

        # 7. 更新网络
        if len(dispatch_buffer) > batch_size:
            dispatch_agent.update()
        if len(charging_buffer) > batch_size:
            charging_agent.update()  # 参数共享，一次更新
```

---

## 实施计划

### 阶段1：混合架构实施（2-3周）

**Week 1: 架构搭建**
- [ ] 设计充电智能体网络结构
- [ ] 实现参数共享机制
- [ ] 修改环境接口（分离派车和充电动作）
- [ ] 设计分离的奖励函数

**Week 2: 训练调试**
- [ ] 实现充电智能体训练循环
- [ ] 调试多智能体交互
- [ ] 调整奖励权重
- [ ] 初步训练测试

**Week 3: 优化验证**
- [ ] 对比单智能体性能
- [ ] 分析充电策略是否改善
- [ ] 调优超参数
- [ ] 文档和代码整理

### 阶段2：完全区域级多智能体（1-2个月）

**Month 1: 算法迁移**
- [ ] 研究MADDPG/MAPPO算法
- [ ] 实现区域级智能体
- [ ] 设计通信机制
- [ ] 初步训练

**Month 2: 性能优化**
- [ ] 对比混合架构性能
- [ ] 优化训练效率
- [ ] 大规模训练
- [ ] 性能评估

---

## 预期效果

### 方案3（混合架构）预期：

**服务率：**
- 当前：1.94%
- 预期：15-30%
- 提升：8-15倍

**充电策略：**
- 当前：所有车辆持续充电（即使满电）
- 预期：按需充电，SOC < 50%时充电

**车队利用率：**
- 当前：接近0%（都在充电）
- 预期：40-60%

**奖励值：**
- 当前：15,000,000（虚高）
- 预期：5,000,000-8,000,000（真实服务收入）

---

## 风险和挑战

### 技术风险：

1. **训练不稳定**
   - 多智能体训练比单智能体更不稳定
   - 缓解：使用参数共享、经验回放、目标网络

2. **收敛速度慢**
   - 多智能体需要更多样本
   - 缓解：并行训练、课程学习

3. **协调问题**
   - 派车和充电可能冲突
   - 缓解：设计合理的奖励函数、通信机制

### 工程风险：

1. **代码复杂度增加**
   - 需要重构大量代码
   - 缓解：模块化设计、充分测试

2. **调试困难**
   - 多智能体行为难以分析
   - 缓解：详细日志、可视化工具

---

## 总结

**最佳路径：**
1. **立即实施方案3（混合架构）** - 快速解决充电黑客问题
2. **验证效果后考虑方案2（区域级）** - 长期优化

**关键成功因素：**
- ✓ 合理的奖励函数设计
- ✓ 参数共享减少复杂度
- ✓ 渐进式改造降低风险
- ✓ 充分的测试和验证
