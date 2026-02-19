# 使用toy_network奖励函数测试SIP数据集

## 目标
验证toy_network的奖励函数是否能在SIP数据集上避免充电黑客问题。

## 背景分析

### toy_network训练结果（正常）
- 配置：4区域，18车辆，12时间步
- 电池容量：2000
- 训练结果：达到最优解的77.9%
- 充电行为：正常（没有充电黑客问题）

### SIP数据集训练结果（异常）
- 配置：50区域，200车辆，26时间步
- 电池容量：3000
- 训练结果：奖励很高（1500万+）
- 充电行为：异常（所有车辆持续充电，服务率仅1.94%）

## 关键问题

**为什么toy_network正常，SIP异常？**

可能原因：
1. **奖励函数尺度问题**
   - toy_network: 4区域 × 充电奖励 = 4倍奖励
   - SIP: 50区域 × 充电奖励 = 50倍奖励（12.5倍差异）
   - 服务收入权重没有相应调整

2. **充电奖励设计问题**
   - 可能存在"满电也能获得充电奖励"的漏洞
   - 在小规模下影响不明显
   - 在大规模下被放大

3. **状态空间信息不足**
   - SIP的251维状态可能缺少关键信息
   - toy_network的29维状态可能更聚焦

## 实验方案

### 方案A：直接复制toy_network奖励函数（快速测试）

**步骤：**
1. 找到toy_network的奖励函数代码
2. 复制到SIP环境
3. 训练10-50轮
4. 观察充电行为

**预期结果：**
- 如果充电行为正常 → 证明是奖励函数尺度问题
- 如果充电行为仍异常 → 可能是其他问题

### 方案B：按比例调整奖励权重（更合理）

**调整原则：**
```python
# toy_network (4区域)
charging_bonus_per_zone = X
total_charging_bonus = 4 * X

# SIP (50区域)
# 保持总奖励尺度一致
charging_bonus_per_zone = X / 12.5  # 或者调整其他权重
```

**具体调整：**
1. 充电奖励 ÷ 12.5（区域数比例）
2. 或者服务收入 × 12.5
3. 或者两者都调整

### 方案C：使用相对奖励（最优）

**设计思路：**
```python
# 不使用绝对值，使用比例
charging_reward = charging_count / total_vehicles * base_reward
service_reward = service_count / total_demand * base_reward
```

这样可以自动适应不同规模。

## 需要的信息

为了帮你实施，我需要知道：

### 问题1：toy_network的奖励函数参数是什么？

请提供toy_network训练时的奖励函数参数：
- 充电奖励（charging_bonus）
- 服务收入权重
- SOC惩罚系数
- 其他奖励/惩罚项

**如何查找：**
```bash
# 查看toy_network的奖励函数
grep -A 20 "def calculate_reward\|reward_value =" specific/dclily_elec.py

# 或者查看历史提交
git log --all --grep="toy_network\|reward" --oneline
```

### 问题2：当前SIP的奖励函数参数是什么？

**当前奖励函数位置：**
- 文件：`specific/dclily_elec.py`
- 大约在：1500-1700行

### 问题3：你想用哪个方案？

- **方案A**：直接复制（最快，1小时）
- **方案B**：按比例调整（合理，2小时）
- **方案C**：重新设计相对奖励（最优，4小时）

## 实施步骤（以方案A为例）

### Step 1：备份当前奖励函数
```bash
cp specific/dclily_elec.py specific/dclily_elec_sip_backup.py
```

### Step 2：查找toy_network奖励函数
```bash
# 如果有git历史
git log --all --oneline | grep toy

# 或者查看备份文件
ls -la specific/ | grep backup
```

### Step 3：替换奖励函数
- 复制toy_network的奖励计算代码
- 替换到当前SIP环境

### Step 4：清除缓存并测试
```bash
# 清除Python缓存
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# 运行10轮测试
python main.py --train_eps 10
```

### Step 5：分析结果
```python
# 检查充电行为
import numpy as np
rewards = np.load('rewards.npy')
svc = np.load('svc_history.npy')

print(f'Average reward: {np.mean(rewards)}')
print(f'Average service rate: {np.mean(svc)}')

# 查看车辆行为日志
head -50 vehicle_action_log.csv
```

## 预期结果与判断

### 如果充电行为正常（成功）
- 服务率 > 10%
- 充电次数适中（不是全部充电）
- 车辆SOC分布合理

**结论：**
- 问题确实是奖励函数尺度
- 可以继续优化这个奖励函数
- 然后再进行MARL迁移

### 如果充电行为仍异常（失败）
- 服务率 < 5%
- 所有车辆仍在充电

**可能原因：**
1. toy_network奖励函数本身有问题（只是在小规模下不明显）
2. SIP数据集有其他问题（数据质量、环境逻辑）
3. 状态空间设计问题

**下一步：**
- 分析toy_network和SIP的详细差异
- 检查环境逻辑是否一致
- 考虑使用方案C（相对奖励）

## 时间估算

- 方案A：1-2小时（查找+替换+测试）
- 方案B：2-3小时（计算比例+调整+测试）
- 方案C：4-6小时（重新设计+实现+测试）

## 下一步行动

请告诉我：
1. 你能找到toy_network的奖励函数参数吗？
2. 你想用哪个方案（A/B/C）？
3. 需要我帮你查找toy_network的历史代码吗？

我可以帮你：
- 查找toy_network的奖励函数
- 对比两个版本的差异
- 实施替换或调整
- 分析测试结果
