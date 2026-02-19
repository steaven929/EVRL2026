# toy_network奖励函数恢复方案

## 当前SIP奖励函数问题诊断

### 问题代码（dclily_elec.py 1503-1598行）
```python
charging_bonus = 15000 * charging_count  # 每辆充电车+15000
charging_maintenance_bonus = 3000 * charging_count  # 每辆充电车+3000
# 总计：18000/车

reward_value = (service_revenue - soc_penalty) / 10 + charging_bonus + charging_maintenance_bonus + soc_maintain_bonus - fixed_penalty
```

### 问题分析
1. **充电奖励过高**：18000/车 × 200车 = 360万（如果全部充电）
2. **服务收入被削弱**：除以10
3. **尺度失衡**：充电奖励 >> 服务收入

### 为什么SIP会出现充电黑客？
```
假设场景：
- 服务1个订单：收入约100-500，除以10后 = 10-50
- 充电1辆车：奖励18000

智能体学习结果：
- 充电1辆车 = 服务360-1800个订单
- 当然选择充电！
```

## 解决方案

### 方案1：简单修复（推荐，最快）

**修改原则：**
- 大幅降低充电奖励
- 移除充电维护奖励
- 增加服务收入权重

**具体修改：**
```python
# 1. 降低充电奖励（从15000降到500）
charging_bonus = 500 * charging_count

# 2. 移除充电维护奖励
# charging_maintenance_bonus = 3000 * charging_count  # 注释掉

# 3. 增加服务收入权重（从/10改为×5）
reward_value = service_revenue * 5 - soc_penalty + charging_bonus + soc_maintain_bonus - fixed_penalty
```

**预期效果：**
- 充电奖励：500/车 × 200车 = 10万（如果全部充电）
- 服务收入：100-500 × 5 = 500-2500/单
- 服务10-20单 = 充电1辆车（更合理）

### 方案2：相对奖励（最优，需要重新设计）

**设计思路：**
```python
# 使用比例而非绝对值
charging_ratio = charging_count / CAR_NUM  # 0-1
service_ratio = service_count / total_demand  # 0-1

charging_reward = charging_ratio * 10000  # 最多10000
service_reward = service_ratio * 50000  # 最多50000

reward_value = service_reward + charging_reward - soc_penalty
```

**优点：**
- 自动适应不同规模（4区域或50区域）
- 奖励尺度稳定
- 鼓励平衡策略

### 方案3：toy_network原始参数（如果能找到）

**需要查找的参数：**
- charging_bonus系数
- service_revenue权重
- soc_penalty系数

**查找方法：**
```bash
# 方法1：查看git历史
git log --all --oneline | grep -E "toy|small|4.*zone"
git show <commit_hash>:specific/dclily_elec.py | grep -A 20 "charging_bonus\|reward_value"

# 方法2：查看备份文件
ls -la specific/ | grep backup
ls -la checkpoint/ | grep backup

# 方法3：查看实验日志
cat checkpoint/CLAUDE.md | grep -A 10 "toy_network\|4.*zone"
```

## 推荐实施步骤

### Step 1：备份当前文件
```bash
cp specific/dclily_elec.py specific/dclily_elec_sip_broken.py
```

### Step 2：应用方案1（简单修复）
修改 `specific/dclily_elec.py` 的1503-1598行：

```python
# =========修改：降低充电奖励，避免充电黑客==========
# 充电奖励：从15000降到500
charging_bonus = 500 * charging_count

# 移除充电维护奖励（注释掉）
# charging_maintenance_bonus = 3000 * charging_count

# SOC维持奖励保持不变
if avg_soc >= 1200:
    soc_maintain_bonus = (avg_soc - 1200) * 2.0
else:
    soc_maintain_bonus = 0

# 新奖励函数：增加服务收入权重
reward_value = service_revenue * 5 - soc_penalty + charging_bonus + soc_maintain_bonus - fixed_penalty
# ======修改结束
```

### Step 3：同步修改终止步奖励（1624行）
```python
# 终止步也使用相同的充电奖励
charging_bonus = 500 * charging_count  # 从3000改为500
```

### Step 4：清除缓存
```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
```

### Step 5：测试训练（10轮）
```bash
python main.py --train_eps 10
```

### Step 6：分析结果
```python
import numpy as np
import pandas as pd

# 检查奖励
rewards = np.load('rewards.npy')
print(f'Average reward: {np.mean(rewards):.0f}')

# 检查服务率
svc = np.load('svc_history.npy')
print(f'Average service rate: {np.mean(svc)*100:.2f}%')

# 检查充电行为
df = pd.read_csv('vehicle_action_log.csv')
print(f'Charging actions: {len(df[df.iloc[:, -1].str.contains("Charging")])}')
print(f'Full SOC charging: {len(df[(df.iloc[:, 3] >= 2900) & df.iloc[:, -1].str.contains("Charging")])}')
```

## 成功标准

### 如果修复成功：
- ✅ 服务率 > 10%
- ✅ 奖励值：500万-1000万（合理范围）
- ✅ 充电行为：按需充电（SOC<50%时充电）
- ✅ 满电充电次数：接近0

### 如果仍有问题：
- ❌ 服务率 < 5%
- ❌ 奖励值 > 1500万（虚高）
- ❌ 所有车辆仍在充电

**下一步：**
- 尝试方案2（相对奖励）
- 或者进一步降低充电奖励（500 → 200）

## 时间估算

- 修改代码：10分钟
- 清除缓存：1分钟
- 训练10轮：10-20分钟
- 分析结果：5分钟

**总计：30-40分钟**

## 需要我帮你做什么？

1. ✅ 我可以直接帮你修改代码（应用方案1）
2. ✅ 我可以帮你查找toy_network的历史参数
3. ✅ 我可以帮你分析测试结果
4. ✅ 我可以设计方案2（相对奖励）

请告诉我你想：
- A. 直接应用方案1（简单修复）
- B. 先查找toy_network原始参数
- C. 设计方案2（相对奖励）
