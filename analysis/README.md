# 充电偏好值分析工具集

本文件夹包含用于分析RL充电决策的脚本工具和版本管理系统。

## 📋 使用方式说明

### 文档类型

**静态分析文档**（不需要每次更新）：
- `zone_classification_explanation.md` - 区域分类说明（基于OD需求数据）
- `charging_behavior_mapping_analysis.md` - 充电行为映射分析（代码逻辑说明）
- `../docs/charging_preference_root_cause.md` - 根本原因分析（当前问题诊断）

**动态分析结果**（每次训练后重新生成）：
- `charging_preferences_analysis.txt` - 测试期充电偏好值分析
- `exploration_vs_test_comparison.txt` - 探索期vs测试期对比
- `../action/RL-DDPG_actions*.txt` - 详细动作记录

### 推荐工作流程

```bash
# 1. 训练新模型
python main.py --train_eps 50

# 2. 自动运行所有分析（推荐）
python analysis/auto_analyze.py

# 3. 保存当前版本的分析结果
python analysis/version_manager.py save --name "E3_50epochs"

# 4. 查看所有版本
python analysis/version_manager.py list

# 5. 对比两个版本
python analysis/version_manager.py compare --v1 "D2_50epochs" --v2 "E3_50epochs"
```

---

## 🔧 脚本列表

### 核心分析脚本

#### 1. analyze_charging_preferences.py
**功能**：分析测试期的充电偏好值分布

**用法**：
```bash
python analysis/analyze_charging_preferences.py
```

**输出**：
- `analysis/charging_preferences_analysis.txt`

**分析内容**：
- 每个区域的充电偏好值统计（最小值、最大值、平均值、中位数）
- 充电偏好值的数量级分布
- 每个时间步的详细充电偏好值

---

#### 2. save_exploration_actions.py
**功能**：保存探索期（第10轮训练）的详细动作数据

**用法**：
```bash
python analysis/save_exploration_actions.py
```

**前提条件**：
- 需要已训练的模型文件 `trained_model.pth`

**输出**：
- `action/RL-DDPG_actions_exploration.txt`

**记录内容**：
- Actor原始输出（无噪声）
- 添加OU噪声后的动作
- 充电噪声（±0.3）
- 最终动作（添加充电噪声后）

---

#### 3. compare_exploration_test.py
**功能**：对比探索期和测试期的充电偏好值

**用法**：
```bash
python analysis/compare_exploration_test.py
```

**前提条件**：
- 需要 `action/RL-DDPG_actions.txt`（测试期数据）
- 需要 `action/RL-DDPG_actions_exploration.txt`（探索期数据）

**输出**：
- `analysis/exploration_vs_test_comparison.txt`

**对比内容**：
- 每个区域在探索期和测试期的充电偏好值统计
- 探索期平均值 / 测试期平均值的倍数
- 每个时间步的详细对比

---

### 自动化工具

#### 4. auto_analyze.py
**功能**：训练后自动运行所有分析脚本

**用法**：
```bash
python analysis/auto_analyze.py
```

**执行流程**：
1. 检查是否存在训练好的模型
2. 分析测试期充电偏好值
3. 保存探索期动作数据
4. 对比探索期和测试期
5. 显示关键指标快速预览

**优势**：
- 一键运行所有分析
- 自动检查前置条件
- 显示关键指标摘要

---

#### 5. version_manager.py
**功能**：分析结果版本管理

**用法**：

**保存当前版本**：
```bash
python analysis/version_manager.py save --name "E3_50epochs"
```

**列出所有版本**：
```bash
python analysis/version_manager.py list
```

**对比两个版本**：
```bash
python analysis/version_manager.py compare --v1 "D2_50epochs" --v2 "E3_50epochs"
```

**版本存储位置**：
- `analysis/versions/{version_name}/`

**保存内容**：
- 所有分析结果文件
- 元数据（时间戳、奖励值等）

---

## 📊 完整分析流程示例

### 场景1：首次训练后分析

```bash
# 步骤1：训练模型
python main.py --train_eps 50

# 步骤2：运行自动分析
python analysis/auto_analyze.py

# 步骤3：保存版本
python analysis/version_manager.py save --name "baseline_50epochs"

# 步骤4：查看结果
cat analysis/charging_preferences_analysis.txt
cat analysis/exploration_vs_test_comparison.txt
```

### 场景2：改进后对比分析

```bash
# 步骤1：实施改进（例如：降低阈值）
# 修改 dclily_elec.py 中的分段映射阈值

# 步骤2：重新训练
python main.py --train_eps 50

# 步骤3：运行自动分析
python analysis/auto_analyze.py

# 步骤4：保存新版本
python analysis/version_manager.py save --name "lower_threshold_50epochs"

# 步骤5：对比两个版本
python analysis/version_manager.py compare --v1 "baseline_50epochs" --v2 "lower_threshold_50epochs"
```

### 场景3：手动分析（逐步执行）

```bash
# 如果需要单独运行某个分析
python analysis/analyze_charging_preferences.py
python analysis/save_exploration_actions.py
python analysis/compare_exploration_test.py
```

---

## 🎯 关键指标说明

### 测试期充电偏好值（最新数据）
- **区域0（有车辆）**：平均 3.95e-06（极小）
- **区域1（过境区）**：平均 0.251（较大）
- **区域2（过境区）**：平均 5.42e-06（极小）
- **区域3（有车辆）**：平均 6.30e-05（较小）

### 探索期充电偏好值
- **区域0（有车辆）**：平均 0.076（较大）
- **区域1（过境区）**：平均 0.311（较大）
- **区域2（过境区）**：平均 0.046（较大）
- **区域3（有车辆）**：平均 0.165（较大）

### 探索期 vs 测试期倍数
- **区域0**：探索期是测试期的 **19,243倍**
- **区域1**：探索期是测试期的 1.24倍
- **区域2**：探索期是测试期的 8,586倍
- **区域3**：探索期是测试期的 **2,616倍**

---

## 💡 使用建议

### 何时需要重新分析？

**必须重新分析**：
- 修改了充电决策逻辑（dclily_elec.py）
- 修改了奖励函数
- 修改了探索策略
- 重新训练了模型

**不需要重新分析**：
- 只是查看历史数据
- 对比不同版本
- 阅读静态分析文档

### 如何判断改进是否有效？

**关键指标**：
1. **测试期充电偏好值**：是否增大？（目标：> 0.1）
2. **实际充电次数**：是否增加？（目标：接近最优解112次）
3. **训练奖励**：是否提升？
4. **探索期vs测试期倍数**：是否缩小？（说明RL学会了保持充电行为）

**改进成功的标志**：
- 区域0/3的测试期充电偏好值 > 0.1
- 实际充电次数 > 50次
- 训练奖励持续上升
- 探索期和测试期的充电偏好值接近

---

## 📝 相关文档

- **完整根因分析**：`../docs/charging_preference_root_cause.md`
- **实验记录**：`../checkpoint/experiment_*.md`
- **项目说明**：`../checkpoint/readai_me.txt`

---

## 🔧 维护说明

### 添加新的分析脚本

1. 在 `analysis/` 文件夹创建新脚本
2. 更新本 README
3. 如果需要自动运行，修改 `auto_analyze.py`
4. 如果需要版本管理，修改 `version_manager.py`

### 文件命名规范

- 分析脚本：`analyze_*.py`
- 结果文件：`*_analysis.txt` 或 `*_comparison.txt`
- 版本文件夹：`versions/{version_name}/`

---

**创建日期**：2026-02-07
**最后更新**：2026-02-07
**维护者**：Claude Code + Happy
