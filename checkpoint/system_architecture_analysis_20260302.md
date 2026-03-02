# MARL系统架构分析与调整方案 (2026-03-02)

## 一、问题诊断总结

### 训练结果 (marl_20260302-004802, 1000 episodes)
- **Service Rate**: 2.06% ⚠️ **极低**
- **Vehicle Utilization**: 6.05% ⚠️ **极低**
- **Fleet Health**: 97.16% ✓ **过高**
- **Charging Efficiency**: 49.05% ⚠️ **低**
- **Response Readiness**: 91.95% ✓ **过高**

### 核心问题
**Agent学会了"躺平"策略**：保持高电量、均衡分布，但拒绝服务订单

## 二、系统架构分析

### 2.1 Reward函数架构 (specific/dclily_elec.py:1640-1670)

```python
# 当前Reward函数结构
base_reward = amplified_service_revenue - effective_soc_penalty - effective_service_rate_penalty + service_rate_bonus

if reward_mode == 'shaping':
    reward_value = (
        base_reward
        + reward_weights['charging_bonus'] * charging_bonus
        + reward_weights['charging_maintenance'] * charging_maintenance_bonus
        + reward_weights['soc_maintain'] * soc_maintain_bonus
    )
```

#### 当前权重配置
| 组件 | 数值 | 问题 |
|------|------|------|
| **服务收益** | service_revenue × 10 | ✓ 已放大 |
| **充电奖励** | 1500/车 | ⚠️ **过高** |
| **充电维护** | 300/车 | ⚠️ **过高** |
| **SOC维持** | (avg_soc-1200) × 0.04 | ⚠️ 鼓励高SOC |
| **SOC惩罚** | 前500轮=0, 后期×0.1 | ⚠️ **几乎无效** |
| **服务率奖励** | 10000-20000 | ✓ 但不够 |
| **服务率惩罚** | 前500轮=0, 后期×0.1 | ⚠️ **几乎无效** |

### 2.2 问题根源分析

#### 问题1: Reward失衡 - 充电收益 >> 服务收益
**计算示例** (单个timestep):
```
服务1个订单收益: ~100 × 10 = 1000
充电1辆车收益: 1500 + 300 = 1800
SOC维持奖励: (2900-1200) × 0.04 × 200车 = 13600

总结：充电+维持SOC >> 服务订单
```

**Agent学习结果**：
- 最优策略 = 充电 + 保持高SOC + 不服务
- 符合reward最大化目标，但违背业务目标

#### 问题2: 惩罚机制失效
**前500轮** (Episode 1-500):
- SOC惩罚 = 0
- 服务率惩罚 = 0
- **结果**: Agent学会"躺平"，无压力服务

**后500轮** (Episode 501-1000):
- SOC惩罚 × 0.1 = 几乎无效
- 服务率惩罚 × 0.1 = 几乎无效
- **结果**: 已形成的"躺平"策略难以改变

#### 问题3: Curriculum Learning策略错误
**原设计意图**: 前期宽松→后期严格
**实际效果**: 前期学会躺平→后期惩罚太弱无法纠正

#### 问题4: 充电效率低 (49%)
**原因**:
- 充电奖励不区分SOC状态（已修复但不够）
- SOC>95%仍有充电倾向
- 满电惩罚(-500)远小于充电奖励(1800)

### 2.3 MARL架构分析

#### 当前架构 (IPPO with Parameter Sharing)
```
50个Zone-level Agents
├── Shared Actor Network (局部观测 7-dim → 动作 3-dim)
├── Centralized Critic (全局状态 251-dim → Q值)
└── Global Reward Sharing (所有agent获得相同reward)
```

**问题**:
- Global Reward无法区分各agent贡献
- 无法惩罚"搭便车"行为（某些agent躺平）
- 缺少Credit Assignment机制

## 三、调整方案

### 方案A: 激进Reward重塑 (推荐，立即实施)

#### A1. 大幅降低充电相关奖励
```python
# 修改 specific/dclily_elec.py:1541, 1617

# 当前
charging_bonus = 1500  # 每辆车
charging_maintenance_bonus = 300 * charging_count

# 修改为
charging_bonus = 300  # 降低80%
charging_maintenance_bonus = 50 * charging_count  # 降低83%
```

**理由**: 充电是手段不是目的，不应过度奖励

#### A2. 移除SOC维持奖励
```python
# 修改 specific/dclily_elec.py:1608-1612

# 当前
if avg_soc >= 1200:
    soc_maintain_bonus = (avg_soc - 1200) * 0.04

# 修改为
soc_maintain_bonus = 0  # 完全移除
```

**理由**: 鼓励高SOC导致车辆不愿服务（怕消耗电量）

#### A3. 恢复并增强SOC惩罚
```python
# 修改 specific/dclily_elec.py:1646-1653

# 当前
if epoch < 500:
    effective_soc_penalty = 0
else:
    effective_soc_penalty = soc_penalty * 0.1

# 修改为
# 全程启用SOC惩罚，但分阶段调整强度
if epoch < 200:
    effective_soc_penalty = soc_penalty * 0.3  # 前期适度
elif epoch < 500:
    effective_soc_penalty = soc_penalty * 0.5  # 中期增强
else:
    effective_soc_penalty = soc_penalty * 1.0  # 后期全力
```

**理由**: 需要持续压力防止电量过低，但不能前期为0

#### A4. 大幅提升服务收益
```python
# 修改 specific/dclily_elec.py:1642

# 当前
amplified_service_revenue = service_revenue * 10.0

# 修改为
amplified_service_revenue = service_revenue * 50.0  # 提升5倍
```

**理由**: 服务是核心目标，必须成为最大收益来源

#### A5. 增强服务率奖励
```python
# 修改 specific/dclily_elec.py:1477-1481

# 当前
if service_rate > 0.1:
    service_rate_bonus = service_rate * 10000
elif service_rate > 0.05:
    service_rate_bonus = service_rate * 20000

# 修改为
if service_rate > 0.2:
    service_rate_bonus = service_rate * 50000  # 大幅提升
elif service_rate > 0.1:
    service_rate_bonus = service_rate * 30000
elif service_rate > 0.05:
    service_rate_bonus = service_rate * 20000
else:
    service_rate_bonus = service_rate * 10000
```

**理由**: 正向激励比惩罚更有效

#### A6. 添加空闲惩罚（新增）
```python
# 在 specific/dclily_elec.py:1654 之前添加

# 新增：空闲车辆惩罚
idle_penalty = 0
for i in range(self.CAR_NUM):
    # 高电量(>60%)、不在服务、不在充电 = 空闲
    if (self.car[i][1] > BATTERY_CAPACITY * 0.6 and
        self.car[i][2] == 0 and
        self.car[i][3] == -1):
        idle_penalty += 200  # 每辆空闲车惩罚200

base_reward = amplified_service_revenue - effective_soc_penalty - idle_penalty + service_rate_bonus
```

**理由**: 直接惩罚"躺平"行为

#### A7. 修复充电效率问题
```python
# 修改 specific/dclily_elec.py:1537-1541

# 当前
if soc >= BATTERY_CAPACITY * 0.95:
    full_battery_penalty -= 500
else:
    charging_count += 1
    charging_bonus += 300  # 已按A1修改

# 修改为
if soc >= BATTERY_CAPACITY * 0.80:  # 降低阈值到80%
    full_battery_penalty -= 1000  # 加大惩罚
elif soc >= BATTERY_CAPACITY * 0.60:
    # 60-80%：适度充电，小奖励
    charging_count += 1
    charging_bonus += 150
else:
    # <60%：需要充电，正常奖励
    charging_count += 1
    charging_bonus += 300
```

**理由**: 分层充电策略，避免过度充电

### 方案B: 架构升级 (中期，1-2周)

#### B1. 实现Credit Assignment
```python
# 新增 specific/marl_model.py 中的 LocalCritic

class LocalCritic(nn.Module):
    """每个agent的局部Critic，评估自己的贡献"""
    def __init__(self, local_obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(local_obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, local_obs, action):
        return self.net(torch.cat([local_obs, action], dim=-1))
```

#### B2. 实现Local Reward Shaping
```python
# 修改 core/marl_training.py

def compute_local_rewards(env, global_reward, agent_actions):
    """根据agent行为分配局部奖励"""
    local_rewards = []

    for agent_id in range(env.n_agents):
        # 基础：全局奖励均分
        base_reward = global_reward / env.n_agents

        # 加成：该agent的服务贡献
        zone_service = count_zone_service(env, agent_id)
        service_bonus = zone_service * 100

        # 惩罚：该agent的空闲时间
        if is_idle(env, agent_id):
            idle_penalty = -50
        else:
            idle_penalty = 0

        local_reward = base_reward + service_bonus + idle_penalty
        local_rewards.append(local_reward)

    return local_rewards
```

#### B3. 升级到MAPPO
- 使用PPO替代DDPG
- 添加Advantage函数
- 实现GAE (Generalized Advantage Estimation)

### 方案C: 环境诊断 (立即，并行)

#### C1. 验证服务逻辑
```python
# 新增 test_service_logic.py

def test_service_availability():
    """测试：为什么服务率这么低？"""
    env = gym.make('dclily')
    env.reset()

    for t in range(26):
        # 统计可服务订单数
        available_orders = count_available_orders(env)
        # 统计可用车辆数
        available_vehicles = count_available_vehicles(env)
        # 统计实际服务数
        actual_service = count_actual_service(env)

        print(f"t={t}: Orders={available_orders}, Vehicles={available_vehicles}, Served={actual_service}")

        # 执行随机动作
        action = env.action_space.sample()
        env.step(action)
```

#### C2. 分析Action分布
```python
# 新增 analyze_actions.py

def analyze_action_distribution(model_path):
    """分析训练好的模型的action分布"""
    agent = load_agent(model_path)
    env = gym.make('dclily')

    dispatch_actions = []
    charging_actions = []

    for episode in range(10):
        obs = env.reset()
        for t in range(26):
            action = agent.select_action(obs, deterministic=True)

            # 分析dispatch倾向
            dispatch_pref = action[:50]
            dispatch_actions.append(dispatch_pref)

            # 分析charging倾向
            charging_pref = action[100:150]
            charging_actions.append(charging_pref)

            obs, _, _, _ = env.step(action)

    # 统计分析
    print(f"Dispatch mean: {np.mean(dispatch_actions):.4f}")
    print(f"Dispatch std: {np.std(dispatch_actions):.4f}")
    print(f"Charging mean: {np.mean(charging_actions):.4f}")
    print(f"Charging std: {np.std(charging_actions):.4f}")
```

## 四、实施计划

### Phase 1: 紧急修复 (1-2天)

**目标**: 修复reward失衡，提升服务率到10%+

**步骤**:
1. ✅ 实施方案A (A1-A7)
2. ✅ 运行100轮快速验证
3. ✅ 如果服务率>5%，继续训练到500轮
4. ✅ 如果服务率<5%，执行方案C诊断

**预期结果**:
- Service Rate: 2% → 10-15%
- Vehicle Utilization: 6% → 20-30%
- Fleet Health: 97% → 70-80%

### Phase 2: 深度优化 (3-5天)

**目标**: 服务率提升到30%+，接近最优解

**步骤**:
1. 分析Phase 1结果
2. 微调reward权重
3. 实施方案C的诊断工具
4. 训练1000轮完整模型

**预期结果**:
- Service Rate: 15% → 30-40%
- 接近或超过单agent baseline

### Phase 3: 架构升级 (1-2周)

**目标**: 实现真正的MARL优势

**步骤**:
1. 实施方案B (Credit Assignment)
2. 升级到MAPPO
3. 对比IPPO vs MAPPO性能

**预期结果**:
- Service Rate: 40% → 50-60%
- 超越单agent baseline
- 展现MARL的协调优势

## 五、风险与应对

### 风险1: Reward调整后训练不稳定
**应对**:
- 保存每100轮checkpoint
- 如果崩溃，回退到上一个稳定版本
- 逐步调整权重，不要一次改太多

### 风险2: 服务率仍然很低
**应对**:
- 执行方案C诊断
- 检查环境的服务逻辑是否有bug
- 检查action到实际执行的映射

### 风险3: 充电和服务无法平衡
**应对**:
- 使用Multi-objective RL
- 分别优化充电和服务两个目标
- 使用Pareto优化

## 六、监控指标

### 训练过程监控
- **每10轮**: Service Rate, Vehicle Utilization, Fleet Health
- **每50轮**: Reward分解（service/charging/penalty各占比）
- **每100轮**: Action分布统计

### 关键阈值
- Service Rate < 5% (Episode 100): ⚠️ 立即停止，执行诊断
- Service Rate < 10% (Episode 500): ⚠️ 调整reward权重
- Service Rate > 20% (Episode 500): ✓ 继续训练
- Service Rate > 30% (Episode 1000): ✓ 成功

## 七、代码修改清单

### 立即修改 (方案A)
1. `specific/dclily_elec.py:1541` - 降低charging_bonus (1500→300)
2. `specific/dclily_elec.py:1617` - 降低charging_maintenance (300→50)
3. `specific/dclily_elec.py:1608-1612` - 移除soc_maintain_bonus
4. `specific/dclily_elec.py:1646-1653` - 恢复SOC惩罚
5. `specific/dclily_elec.py:1642` - 提升服务收益 (×10→×50)
6. `specific/dclily_elec.py:1477-1481` - 增强服务率奖励
7. `specific/dclily_elec.py:1654前` - 添加空闲惩罚
8. `specific/dclily_elec.py:1537-1541` - 修复充电效率

### 新增文件 (方案C)
1. `test_service_logic.py` - 服务逻辑测试
2. `analyze_actions.py` - Action分布分析

### 后续修改 (方案B)
1. `specific/marl_model.py` - 添加LocalCritic
2. `core/marl_training.py` - 实现local reward shaping
3. `core/marl_agent.py` - 升级到MAPPO

## 八、总结

**核心问题**: Reward函数严重失衡，充电收益远大于服务收益

**根本原因**:
1. 充电奖励过高 (1800/车)
2. SOC维持奖励鼓励高电量 (13600/timestep)
3. 前500轮无惩罚，学会"躺平"
4. 服务收益相对太低

**解决方案**:
1. **立即**: 大幅降低充电奖励，提升服务收益，添加空闲惩罚
2. **中期**: 实现Credit Assignment，区分各agent贡献
3. **长期**: 升级到MAPPO，实现真正的MARL优势

**预期效果**:
- Phase 1: Service Rate 2% → 15%
- Phase 2: Service Rate 15% → 35%
- Phase 3: Service Rate 35% → 55%

**关键成功因素**:
1. Reward权重调整要大胆（当前失衡太严重）
2. 持续监控训练过程，及时调整
3. 如果仍然失败，必须检查环境逻辑
