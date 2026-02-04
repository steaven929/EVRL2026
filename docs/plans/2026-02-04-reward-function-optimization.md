# 奖励函数优化实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 优化奖励函数以改善RL充电策略，缩小与最优解的性能差距

**Architecture:** 调整环境奖励函数中的充电奖励系数和低电量惩罚力度，使RL学习到主动充电策略

**Tech Stack:** Python, NumPy, Gym环境

**当前性能：**
- RL总奖励: -5100.00
- 最优解总奖励: 469.80
- 差距: 5569.80

**问题诊断：**
- RL充电次数: 6次 vs 最优解: 112次
- RL在时间步7-11完全无服务（电量耗尽）
- 充电奖励(50/次)不足以覆盖充电成本

---

### Task 1: 备份当前配置

**Files:**
- Read: `specific/dclily_elec.py:1076-1110`
- Create: `checkpoint/2026_claude/reward_backup_2026-02-04.txt`

**Step 1: 读取当前奖励函数配置**

```bash
# 记录当前参数
charging_bonus = 50 * charging_count
SOC_THRESHOLD = 600.0
PENALTY_FACTOR = 50.0 (在optim.py中)
```

**Step 2: 保存备份**

创建备份文件记录当前配置和性能基线。

**Step 3: 提交备份**

```bash
git add checkpoint/2026_claude/reward_backup_2026-02-04.txt
git commit -m "docs: backup reward function config before optimization

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 2: 增加充电奖励系数

**Files:**
- Modify: `specific/dclily_elec.py:1087`

**Step 1: 修改充电奖励系数**

```python
# =========2026.2.4修改，增加充电奖励系数==========
# 原值: charging_bonus = 50 * charging_count
# 新值: 增加到200，使充电更有吸引力
charging_bonus = 200 * charging_count
# ======修改结束
```

**理由：** 当前50的奖励不足以覆盖充电的机会成本（调度成本+时间成本），增加到200使充电更有吸引力。

**Step 2: 提交修改**

```bash
git add specific/dclily_elec.py
git commit -m "feat: increase charging bonus from 50 to 200

Improve charging incentive to encourage RL agent to charge vehicles
more frequently, addressing the issue of vehicles running out of
battery in later timesteps.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 3: 增强低电量惩罚

**Files:**
- Modify: `specific/dclily_elec.py:1089-1099`

**Step 1: 调整低电量惩罚分段**

```python
# =========2026.2.4修改，增强低电量惩罚==========
# 3. 低电量惩罚：分段惩罚（增强力度）
soc_penalty = 0
for i in range(self.CAR_NUM):
    soc = self.car[i][1]
    if soc >= 600:
        penalty = 0
    elif soc >= 400:  # 修改：从200提高到400
        penalty = 1.0 * (600 - soc)  # 修改：从0.5增加到1.0
    elif soc >= 200:
        penalty = 1.0 * 200 + 3.0 * (400 - soc)  # 新增中间档
    else:
        penalty = 1.0 * 200 + 3.0 * 200 + 10.0 * (200 - soc)  # 修改：从5.0增加到10.0
    soc_penalty += penalty
# ======修改结束
```

**理由：**
- 提高惩罚阈值（200→400），更早触发惩罚
- 增加惩罚力度（0.5→1.0, 5.0→10.0）
- 添加中间档位，使惩罚更平滑

**Step 2: 提交修改**

```bash
git add specific/dclily_elec.py
git commit -m "feat: enhance low battery penalty with stronger gradients

- Raise penalty threshold from 200 to 400
- Increase penalty factors (0.5→1.0, 5.0→10.0)
- Add intermediate tier for smoother penalty curve

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 4: 训练并验证

**Files:**
- Run: `main.py`
- Generate: `rewards.npy`, `trained_model.pth`

**Step 1: 运行训练（短期测试）**

```bash
python main.py --train_eps 100
```

预期：前100轮观察充电行为是否改善

**Step 2: 检查训练指标**

```python
import numpy as np
rewards = np.load('rewards.npy')
print(f"最后10轮平均Reward: {rewards[-10:].mean():.2f}")
```

预期：Reward应高于-5100

**Step 3: 生成RL动作对比**

```bash
python baseline_comparison.py
```

**Step 4: 分析充电行为**

检查 `action/RL-DDPG_actions.txt`：
- 充电次数是否增加（目标：>20次）
- 是否在早期时间步充电
- 后期时间步是否仍有服务

**Step 5: 记录实验结果**

在 `checkpoint/tuning_log.md` 中记录：
- 参数修改
- 训练结果
- 充电行为变化

---

### Task 5: 长期训练（如果短期测试通过）

**Files:**
- Run: `main.py`

**Step 1: 完整训练**

```bash
python main.py --train_eps 700
```

**Step 2: 生成最终对比**

```bash
python baseline_comparison.py
```

**Step 3: 分析性能差距**

对比指标：
- 总奖励：目标接近最优解469.80
- 充电次数：目标>50次
- 服务完成率：目标后期时间步仍有服务

**Step 4: 更新文档**

在 `checkpoint/readme2_traininglog.md` 记录：
- 优化前后对比
- 性能提升幅度
- 剩余差距分析

**Step 5: 提交训练结果**

```bash
git add checkpoint/tuning_log.md checkpoint/readme2_traininglog.md
git commit -m "docs: record reward optimization experiment results

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## 预期结果

**优化目标：**
1. 充电次数：6次 → >50次
2. 总奖励：-5100 → >-1000
3. 后期服务：0订单 → 有服务

**如果效果不佳，备选方案：**
1. 进一步增加充电奖励（200→300）
2. 调整固定惩罚（-500→-300）
3. 添加"未来可服务订单数"作为奖励项

---

## 回滚方案

如果优化失败，恢复原配置：

```python
# 恢复原值
charging_bonus = 50 * charging_count
# 恢复原低电量惩罚逻辑
```

参考备份文件：`checkpoint/2026_claude/reward_backup_2026-02-04.txt`
