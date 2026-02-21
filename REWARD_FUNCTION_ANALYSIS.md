# 当前奖励函数分析

## 奖励函数结构

### Base模式（reward_mode='base'）
```python
base_reward = (service_revenue - soc_penalty) / 10 - fixed_penalty - weighted_service_rate_penalty
reward_value = base_reward
```

### Shaping模式（reward_mode='shaping'）
```python
base_reward = (service_revenue - soc_penalty) / 10 - fixed_penalty - weighted_service_rate_penalty
reward_value = base_reward + charging_bonus + charging_maintenance_bonus + soc_maintain_bonus
```

## 各项详细说明

### 1. service_revenue（服务收益）
- 来源：real_reward（Gurobi求解器返回的服务订单收益）
- 数值：正值，取决于完成的订单数
- 缩放：除以10

### 2. soc_penalty（低电量惩罚）
- **分段惩罚**：
  - SOC >= 600: 0
  - 400 <= SOC < 600: 2.0 × penalty_scale × (600 - SOC)
  - 200 <= SOC < 400: 2.0 × penalty_scale × 200 + 10.0 × penalty_scale × (400 - SOC)
    - 额外：× 3.0（指数惩罚）
  - SOC < 200: 2.0 × penalty_scale × 200 + 10.0 × penalty_scale × 200 + 50.0 × penalty_scale × (200 - SOC)
    - 额外：× 3.0（指数惩罚）

- **Curriculum Learning（渐进式训练）**：
  - Epoch 0-300: penalty_scale=1.5, vehicle_penalty=750
  - Epoch 300-600: penalty_scale=1.5→1.2, vehicle_penalty=750→600
  - Epoch 600+: penalty_scale=1.2, vehicle_penalty=600

- **低电量车辆额外惩罚**：
  - low_soc_vehicle_penalty = low_soc_count × vehicle_penalty
  - 当SOC < 400时，车辆计入low_soc_count

- **总SOC惩罚**：
  - soc_penalty = Σ(每辆车的penalty) + low_soc_vehicle_penalty

### 3. fixed_penalty（固定惩罚）
- **当前值：0**（已移除）

### 4. service_rate_penalty（服务率惩罚）
- **计算**：
  - service_rate = served_orders / total_demand
  - 如果 service_rate < 0.5:
    - service_rate_penalty = (0.5 - service_rate) × 8000
  - 否则：service_rate_penalty = 0

- **权重**：
  - weighted_service_rate_penalty = reward_weights['service_rate_penalty'] × service_rate_penalty
  - 默认权重：1.0

- **示例**：
  - service_rate = 0: penalty = 4000
  - service_rate = 0.3: penalty = 1600
  - service_rate = 0.5: penalty = 0

### 5. charging_bonus（充电奖励，仅shaping模式）
- **计算**：15000 × charging_count
- charging_count：正在充电的车辆数（charger_id != -1 且 remain == 0）
- **权重**：reward_weights['charging_bonus']，默认1.0

### 6. charging_maintenance_bonus（充电维护奖励，仅shaping模式）
- **计算**：3000 × charging_count
- **权重**：reward_weights['charging_maintenance']，默认1.0

### 7. soc_maintain_bonus（SOC维持奖励，仅shaping模式）
- **计算**：
  - avg_soc = Σ(car[i][1]) / CAR_NUM
  - 如果 avg_soc >= 1200:
    - soc_maintain_bonus = (avg_soc - 1200) × 2.0
  - 否则：soc_maintain_bonus = 0
- **权重**：reward_weights['soc_maintain']，默认1.0

## 当前配置（MARL训练）

根据代码，MARL训练使用的是：
- **reward_mode**: 'base'（默认）
- **reward_weights**: 全部为1.0（默认）

因此，实际奖励函数为：
```python
reward_value = (service_revenue - soc_penalty) / 10 - 0 - service_rate_penalty
             = (service_revenue - soc_penalty) / 10 - service_rate_penalty
```

## 负奖励原因分析

### 训练结果：平均奖励 = -254

假设一个典型回合（26步）：
- service_revenue: 假设每步完成1个订单，收益约100，总计2600
- soc_penalty:
  - 200辆车，假设平均SOC=1500（50%）
  - 每辆车penalty ≈ 2.0 × 1.5 × (600-1500) = 0（SOC>600无惩罚）
  - 但如果有车辆SOC<600，惩罚会很大
  - 假设有50辆车SOC在400-600，平均SOC=500
    - 每辆：2.0 × 1.5 × (600-500) = 300
    - 总计：50 × 300 = 15,000
  - 假设有10辆车SOC<400，平均SOC=300
    - 每辆：(2.0×1.5×200 + 10.0×1.5×100) × 3.0 = (600+1500)×3 = 6,300
    - 总计：10 × 6,300 = 63,000
    - 额外vehicle_penalty：10 × 750 = 7,500
  - **总SOC惩罚：15,000 + 63,000 + 7,500 = 85,500**

- service_rate_penalty:
  - 假设service_rate = 0.02（2%）
  - penalty = (0.5 - 0.02) × 8000 = 3,840

- **总奖励**：
  - reward = (2,600 - 85,500) / 10 - 3,840
  - reward = -8,290 - 3,840
  - reward = **-12,130**

### 结论

**负奖励的主要原因：**

1. **SOC惩罚过大**：
   - 低电量车辆的惩罚非常重（指数级×3）
   - 即使少量低电量车辆也会导致巨大惩罚
   - 当前epoch=0-100，penalty_scale=1.5（最高惩罚阶段）

2. **服务率惩罚**：
   - 服务率2%时，惩罚3,840
   - 这个惩罚相当于抵消260个订单的收益

3. **服务收益被缩放**：
   - service_revenue除以10，大幅降低了服务的激励

4. **Shaping奖励未启用**：
   - 当前使用base模式，没有charging_bonus等正向奖励
   - 如果启用shaping模式，充电奖励可以部分抵消惩罚

## 建议

### 短期修复（保持当前架构）：
1. **启用shaping模式**：reward_mode='shaping'
2. **降低SOC惩罚**：penalty_scale从1.5降到0.5
3. **降低服务率惩罚**：从8000降到2000
4. **增加服务收益权重**：不除以10，或只除以2

### 长期优化：
1. **重新设计奖励平衡**：确保正常运营时奖励为正
2. **移除过度惩罚**：指数级×3的惩罚太严厉
3. **简化奖励函数**：当前过于复杂，难以调试
