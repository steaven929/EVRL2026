# 规模扩展问题分析 (2026-02-18)

## 配置对比

| 参数 | Toy Network (小) | SIP Dataset (大) | 比例 |
|------|------------------|------------------|------|
| **区域数** | 4 | 50 | 12.5x |
| **车辆数** | 18 | 200 | 11.1x |
| **时间步** | 12 | 26 | 2.2x |
| **电池容量** | 2000 | 3000 | 1.5x |
| **状态维度** | 29 | 251 | 8.7x |
| **动作维度** | 12 | 150 | 12.5x |
| **停车位总数** | 24 | ~280 | 11.7x |
| **充电桩总数** | ~10 | ~200 | 20x |

## 关键发现

### 1. 电池容量不匹配 ⚠️

**问题**：
- SIP配置：BATTERY_CAPACITY = 3000
- 训练日志：车辆SOC = 2000
- **结论**：环境可能使用了错误的配置

**验证**：
```python
# dclily_elec.py line 67
BATTERY_CAPACITY = config['BATTERY_CAPACITY']  # 应该是3000

# 但日志显示：
24,100,48,2000.00,...  # SOC=2000，不是3000
```

### 2. 奖励规模不匹配 🔴

**问题**：充电奖励是绝对值，不随规模调整

**Toy Network**：
- 18辆车，充电奖励1500/车
- 假设充电3辆：+4500
- 服务收益：~10000/10 = 1000
- 充电/服务比：4.5:1

**SIP Dataset**：
- 200辆车，充电奖励1500/车
- 假设充电30辆：+45000
- 服务收益：~100000/10 = 10000
- 充电/服务比：4.5:1

**但实际情况**：
- SIP的服务收益可能更低（分散在50个区域）
- 充电奖励变得相对更有吸引力
- 导致过度充电

### 3. 状态空间爆炸 🔴

**Toy Network**: 29维
- 网络容量：hidden=512, embed=32
- 参数量：适中

**SIP Dataset**: 251维 (8.7x)
- 网络容量：hidden=768, embed=48 (仅增加50%)
- 参数量增长：不足以匹配状态空间增长
- **结论**：网络容量相对不足

### 4. 探索策略不适配 🟡

**SAC Alpha = 0.10**：
- Toy: 4个区域，12个动作维度
- SIP: 50个区域，150个动作维度
- 相同的探索强度在高维空间中效果不同

### 5. 充电桩密度差异 ⚠️

**Toy Network**：
- 10个充电桩 / 18辆车 = 0.56
- 充电桩相对稀缺

**SIP Dataset**：
- 200个充电桩 / 200辆车 = 1.0
- 充电桩充足，甚至过剩
- **结论**：充电变得"太容易"，RL倾向于过度使用

## 根本原因

### 主要问题：奖励函数未按规模调整

**当前奖励函数**：
```python
reward = (service_revenue - soc_penalty) / 10 +
         charging_bonus + charging_maintenance_bonus +
         soc_maintain_bonus + distribution_balance_bonus -
         fixed_penalty
```

**问题**：
1. `charging_bonus = 1500 * num_charged` - 绝对值
2. `service_revenue / 10` - 固定除数
3. `fixed_penalty = 500` - 绝对值

**在大规模场景下**：
- 充电数量可以很大（30-50辆）
- 充电奖励：1500 * 40 = 60000
- 服务收益：可能只有10000-20000
- **充电奖励主导了总奖励**

### 次要问题：网络容量不足

**状态维度增长8.7x，但网络容量只增长50%**：
- 小样本：29维 → hidden=512
- 大样本：251维 → hidden=768
- 维度/容量比：29/512 = 0.057 vs 251/768 = 0.327
- **大样本的比例高5.7倍，网络压力更大**

## 解决方案

### 方案1：规模自适应奖励函数 (推荐)

**核心思想**：奖励按车队规模归一化

```python
# =========2026.2.18修改，规模自适应奖励==========
# 归一化因子
scale_factor = CAR_NUM / 18  # 以toy network为基准

# 充电奖励归一化
charging_bonus_per_vehicle = 1500 / scale_factor  # 200车时：1500/11.1=135
charging_bonus = charging_bonus_per_vehicle * num_charged

# 或者：充电奖励按比例计算
charging_ratio = num_charged / CAR_NUM
charging_bonus = 1500 * charging_ratio * 18  # 归一化到18辆车的规模

# 服务收益权重调整
service_weight = 10 / scale_factor  # 200车时：10/11.1=0.9
reward = (service_revenue - soc_penalty) / service_weight + ...
# ======修改结束
```

### 方案2：相对奖励而非绝对奖励

```python
# =========2026.2.18修改，相对奖励==========
# 充电奖励：按充电比例而非绝对数量
charging_ratio = num_charged / max(1, num_idle_vehicles)
charging_bonus = 1500 * charging_ratio  # 最大1500

# 服务奖励：按服务率而非绝对收益
service_rate = num_served / max(1, num_orders)
service_bonus = 10000 * service_rate  # 最大10000

reward = service_bonus + charging_bonus + ...
# ======修改结束
```

### 方案3：增大网络容量（配合方案1/2）

```python
# core/util.py
# 对于251维状态，需要更大的网络
parser.add_argument('--critic_hidden_dim',default=1024,type=int)  # 768→1024
parser.add_argument('--actor_hidden_dim',default=1024,type=int)   # 768→1024
parser.add_argument('--critic_embed_dim',default=64,type=int)     # 48→64
parser.add_argument('--actor_embed_dim',default=64,type=int)      # 48→64
```

### 方案4：验证BATTERY_CAPACITY配置

```python
# 检查环境是否正确加载SIP配置
# dclily_elec.py
print(f"[Config Check] BATTERY_CAPACITY: {BATTERY_CAPACITY}")
print(f"[Config Check] ZONE_NUM: {ZONE_NUM}")
print(f"[Config Check] CAR_NUM: {CAR_NUM}")

# 确保使用SIP配置
assert BATTERY_CAPACITY == 3000, "Should use SIP config!"
assert ZONE_NUM == 50, "Should use SIP config!"
```

## 实施优先级

### 立即执行（Critical）：

1. **验证配置加载**
   - 确认BATTERY_CAPACITY=3000
   - 确认使用SIP配置而非toy_network

2. **实施方案2（相对奖励）**
   - 最简单，最有效
   - 不需要调整超参数
   - 自动适应任何规模

### 高优先级（High）：

3. **增大网络容量（方案3）**
   - hidden: 768→1024
   - embed: 48→64
   - 给251维状态更多表示能力

### 中优先级（Medium）：

4. **调整reward_scale**
   - 当前2500可能仍然太高
   - 建议：1500-2000

## 预期效果

**实施方案2（相对奖励）后**：
- 充电奖励：最大1500（而非60000）
- 服务奖励：最大10000
- 服务/充电比：6.7:1（合理）
- 服务率：0.63% → 20-40%
- 总奖励：1,156,881 → 1,800,000+

**配合方案3（大网络）后**：
- 更稳定的学习
- 更好的收敛
- 接近最优解的85-90%

## 关键洞察

1. **规模不变性**：奖励函数应该对问题规模不敏感
2. **相对 > 绝对**：使用比例而非绝对数量
3. **网络容量要匹配**：状态维度增长8x，网络也应该增长相应倍数
4. **小样本成功 ≠ 大样本成功**：需要规模自适应设计
