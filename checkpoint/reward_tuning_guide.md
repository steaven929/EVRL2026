# 奖励函数调整指南  # 修改

## 当前奖励函数  # 修改

**位置：** `specific/dclily_elec.py:1075, 1123`  # 修改

```python  # 修改
reward = sum(real_reward) / 10 - 500  # 修改
```  # 修改

**组成：**  # 修改
- `real_reward`: Gurobi 优化器返回的实际收益（订单收益 - 调度成本 - SOC惩罚）  # 修改
- `/10`: 缩放因子  # 修改
- `-500`: 固定惩罚项（可能代表固定成本）  # 修改

**当前表现：**  # 修改
- 700轮训练后：Reward 从 -2683 改进到 -2294 (+14.5%)  # 修改
- Reward 一直为负数  # 修改

---

## 调整策略  # 修改

### 策略1：移除或减小固定惩罚（推荐首先尝试）  # 修改

**目的：** 让 Reward 更接近实际收益，便于观察改进  # 修改

**修改方案：**  # 修改

```python  # 修改
# 方案1A：完全移除固定惩罚  # 修改
reward = sum(real_reward) / 10  # 修改

# 方案1B：减小固定惩罚  # 修改
reward = sum(real_reward) / 10 - 100  # 从 -500 改为 -100  # 修改

# 方案1C：改为相对惩罚  # 修改
reward = sum(real_reward) / 10 - len(self.car) * 5  # 按车辆数惩罚  # 修改
```  # 修改

**预期效果：**  # 修改
- Reward 可能变为正数或接近0  # 修改
- 更容易观察策略改进  # 修改
- 不改变优化方向，只改变数值范围  # 修改

---

### 策略2：调整缩放因子  # 修改

**目的：** 让 Reward 的数值范围更合理  # 修改

**修改方案：**  # 修改

```python  # 修改
# 方案2A：增大缩放（让 Reward 更小）  # 修改
reward = sum(real_reward) / 100 - 50  # 修改

# 方案2B：减小缩放（让 Reward 更大）  # 修改
reward = sum(real_reward) / 5 - 500  # 修改

# 方案2C：归一化到 [-1, 1]  # 修改
raw_reward = sum(real_reward)  # 修改
reward = np.tanh(raw_reward / 1000)  # 使用 tanh 归一化  # 修改
```  # 修改

**注意：** 缩放因子不改变优化方向，但会影响 Q 值的数值范围  # 修改

---

### 策略3：添加额外奖励项（谨慎使用）  # 修改

**目的：** 引导 Agent 学习特定行为  # 修改

**修改方案：**  # 修改

```python  # 修改
# 方案3A：奖励高服务率  # 修改
base_reward = sum(real_reward) / 10 - 500  # 修改
service_bonus = self._last_svc * 100  # SVC 越高奖励越多  # 修改
reward = base_reward + service_bonus  # 修改

# 方案3B：惩罚调度损失  # 修改
base_reward = sum(real_reward) / 10 - 500  # 修改
rld_penalty = self._last_rld * (-10)  # RLD 越高惩罚越多  # 修改
reward = base_reward + rld_penalty  # 修改

# 方案3C：奖励电量管理  # 修改
base_reward = sum(real_reward) / 10 - 500  # 修改
avg_soc = np.mean([car[1] for car in self.car])  # 平均电量  # 修改
soc_bonus = (avg_soc - 300) / 10  # 电量高于300有奖励  # 修改
reward = base_reward + soc_bonus  # 修改

# 方案3D：综合奖励  # 修改
base_reward = sum(real_reward) / 10  # 移除固定惩罚  # 修改
service_bonus = self._last_svc * 50  # 修改
rld_penalty = self._last_rld * (-5)  # 修改
reward = base_reward + service_bonus + rld_penalty  # 修改
```  # 修改

**警告：** 添加额外项会改变优化目标，需要仔细设计权重  # 修改

---

### 策略4：Reward Shaping（高级）  # 修改

**目的：** 加速学习，提供更密集的反馈  # 修改

```python  # 修改
# 方案4A：增量奖励（奖励改进）  # 修改
if not hasattr(self, '_last_real_reward'):  # 修改
    self._last_real_reward = 0  # 修改
current_reward = sum(real_reward)  # 修改
improvement = current_reward - self._last_real_reward  # 修改
reward = improvement / 10  # 奖励改进量  # 修改
self._last_real_reward = current_reward  # 修改

# 方案4B：潜力奖励（Potential-based shaping）  # 修改
def potential(state):  # 修改
    # 定义状态的"潜力"，例如可用车辆数  # 修改
    return sum(state['available'])  # 修改

current_potential = potential(self.state)  # 修改
next_potential = potential(next_state)  # 修改
shaping = self.gamma * next_potential - current_potential  # 修改
reward = sum(real_reward) / 10 - 500 + shaping  # 修改
```  # 修改

---

## 推荐的调整流程  # 修改

### 第1步：简化奖励函数（最安全）  # 修改

**目标：** 让 Reward 更直观，不改变优化方向  # 修改

```python  # 修改
# 在 dclily_elec.py:1075 和 1123 行修改  # 修改
# 原来：reward = sum(real_reward) / 10 - 500  # 修改
# 修改为：  # 修改
reward = sum(real_reward) / 10  # 移除固定惩罚  # 修改
```  # 修改

**验证：**  # 修改
1. 运行 50-100 轮训练  # 修改
2. 观察 Reward 是否变为正数或接近0  # 修改
3. 检查 Reward 改进趋势是否更明显  # 修改

---

### 第2步：根据结果调整缩放  # 修改

**如果 Reward 数值过大（如 >1000）：**  # 修改
```python  # 修改
reward = sum(real_reward) / 100  # 增大缩放因子  # 修改
```  # 修改

**如果 Reward 数值过小（如 <10）：**  # 修改
```python  # 修改
reward = sum(real_reward) / 5  # 减小缩放因子  # 修改
```  # 修改

**目标：** Reward 在 -100 到 +100 之间比较合适  # 修改

---

### 第3步：添加额外项（可选）  # 修改

**只有在基础奖励工作正常后才考虑**  # 修改

**示例：奖励高服务率**  # 修改
```python  # 修改
base_reward = sum(real_reward) / 10  # 修改
service_bonus = self._last_svc * 50  # 权重需要调整  # 修改
reward = base_reward + service_bonus  # 修改
```  # 修改

**验证：**  # 修改
1. 对比添加前后的训练曲线  # 修改
2. 确认 Agent 确实在优化目标指标（如 SVC）  # 修改
3. 检查是否出现意外行为  # 修改

---

## 注意事项  # 修改

### ⚠️ 修改奖励函数的风险  # 修改

1. **改变优化目标**：添加额外项会改变 Agent 学习的目标  # 修改
2. **需要重新训练**：修改后需要从头训练，不能继续使用旧模型  # 修改
3. **可能变差**：不当的修改可能导致性能下降  # 修改

### ✅ 安全的修改  # 修改

- 移除或调整固定惩罚项  # 修改
- 调整缩放因子  # 修改
- 这些不改变优化方向，只改变数值范围  # 修改

### ⚠️ 需要谨慎的修改  # 修改

- 添加新的奖励项  # 修改
- 改变奖励的组成  # 修改
- 使用 Reward Shaping  # 修改

---

## 实验记录模板  # 修改

每次修改奖励函数后，记录以下信息：  # 修改

```markdown  # 修改
### 实验 X：[简短描述]  # 修改

**日期：** 2026-01-30  # 修改

**修改内容：**  # 修改
- 原公式：`reward = sum(real_reward) / 10 - 500`  # 修改
- 新公式：`reward = sum(real_reward) / 10`  # 修改

**训练配置：**  # 修改
- 轮数：500  # 修改
- critic_lr: 5e-4  # 修改
- batch_size: 64  # 修改

**结果：**  # 修改
- Reward 范围：-50 到 +100  # 修改
- 改进幅度：+150 (30%)  # 修改
- Q Value 趋势：稳定上升  # 修改
- SVC 变化：0.75 → 0.82  # 修改

**结论：**  # 修改
- ✅ 成功：Reward 更直观，改进更明显  # 修改
- 建议：继续使用此配置  # 修改
```  # 修改

---

## 快速参考  # 修改

| 修改类型 | 风险 | 效果 | 推荐度 |  # 修改
|---------|------|------|--------|  # 修改
| 移除固定惩罚 | 低 | 让 Reward 更直观 | ⭐⭐⭐⭐⭐ |  # 修改
| 调整缩放因子 | 低 | 改变数值范围 | ⭐⭐⭐⭐ |  # 修改
| 添加服务率奖励 | 中 | 引导优化 SVC | ⭐⭐⭐ |  # 修改
| 添加 RLD 惩罚 | 中 | 引导减少调度损失 | ⭐⭐⭐ |  # 修改
| Reward Shaping | 高 | 加速学习 | ⭐⭐ |  # 修改
