# RL动作解析实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修改baseline_comparison.py，从环境info字典提取并解析RL详细动作（服务订单、调度、充电），生成与MILP格式一致的动作文件

**Architecture:**
- 利用环境step返回的info字典中的 `dispatch_orders` 和 `demand_orders` 矩阵
- 通过对比两个矩阵来区分服务订单和空驶调度
- 从环境状态中提取充电信息（车辆的charger_id状态）
- 生成与optimal_solver.py相同格式的动作文件

**Tech Stack:** Python, NumPy, Gym环境

---

## Task 1: 验证info字典内容

**Files:**
- Read: `F:\Claude code\EVRL260123\dclily_torch_ddpg_demo0425\specific\dclily_elec.py:1112-1118`
- Test: 临时测试脚本

**Step 1: 创建测试脚本验证info内容**

创建 `test_info_content.py`:

```python
import gym
import numpy as np
from gym.envs.registration import register
from core.util import get_args
from core.environment import env_agent_config

register(id='dclily', entry_point='specific.dclily_elec:dclily')

cfg = get_args()
env = gym.make('dclily')
env.set_training_mode(0)

state = env.reset()
action = env.action_space.sample()
next_state, reward, done, info = env.step(action)

print("Info keys:", info.keys())
print("dispatch_orders shape:", np.array(info['dispatch_orders']).shape)
print("demand_orders shape:", np.array(info['demand_orders']).shape)
print("dispatch_orders:\n", info['dispatch_orders'])
print("demand_orders:\n", info['demand_orders'])
```

**Step 2: 运行测试脚本**

Run: `python test_info_content.py`
Expected: 输出info字典的内容和矩阵形状

**Step 3: 分析输出，确认数据结构**

验证：
- `dispatch_orders` 是否为 4x4 矩阵（ZONE_NUM=4）
- `demand_orders` 是否为 4x4 矩阵
- 矩阵值的含义（车辆数或距离）

---

## Task 2: 修改baseline_comparison.py - 提取info信息

**Files:**
- Modify: `F:\Claude code\EVRL260123\dclily_torch_ddpg_demo0425\baseline_comparison.py:337-451`

**Step 1: 修改动作记录循环，保存info信息**

在baseline_comparison.py的第351-359行，修改为：

```python
while not done:
    action = agent.predict_action(state)

    # 记录step前的充电状态
    pre_charging = {}
    for i in range(len(test_env.car)):
        if test_env.car[i][3] != -1:  # charger_id != -1 表示在充电
            zone = test_env.car[i][0]
            pre_charging[zone] = pre_charging.get(zone, 0) + 1

    next_state, reward, done, info = test_env.step(action)

    # 记录step后的充电状态
    post_charging = {}
    for i in range(len(test_env.car)):
        if test_env.car[i][3] != -1:
            zone = test_env.car[i][0]
            post_charging[zone] = post_charging.get(zone, 0) + 1

    # 提取info中的矩阵信息
    detailed_actions.append({
        'step': step,
        'action': action.tolist() if hasattr(action, 'tolist') else list(action),
        'dispatch_orders': info.get('dispatch_orders', None),
        'demand_orders': info.get('demand_orders', None),
        'pre_charging': pre_charging,
        'post_charging': post_charging,
        'reward': reward
    })

    state = next_state
    step += 1
```

**Step 2: 验证修改**

Run: `python baseline_comparison.py`
Expected: 程序正常运行，不报错

---

## Task 3: 实现动作解析函数

**Files:**
- Modify: `F:\Claude code\EVRL260123\dclily_torch_ddpg_demo0425\baseline_comparison.py:105-140`

**Step 1: 在baseline_comparison.py中添加动作解析函数**

在第105行（load_theoretical_bounds函数之前）添加：

```python
# =========2026.2.4修改，添加动作解析函数==========
def parse_rl_actions(action_data):
    """
    解析RL动作，区分服务订单、调度动作、充电动作

    参数：
    - action_data: 包含dispatch_orders, demand_orders, charging信息的字典

    返回：
    - served_orders: [(from_zone, to_zone, count), ...]
    - reposition_actions: [(from_zone, to_zone, count), ...]
    - charging_actions: {zone: count, ...}
    """
    dispatch_matrix = action_data.get('dispatch_orders')
    demand_matrix = action_data.get('demand_orders')
    post_charging = action_data.get('post_charging', {})

    served_orders = []
    reposition_actions = []
    charging_actions = post_charging.copy()

    if dispatch_matrix is None or demand_matrix is None:
        return served_orders, reposition_actions, charging_actions

    # 转换为numpy数组
    dispatch_matrix = np.array(dispatch_matrix)
    demand_matrix = np.array(demand_matrix)

    zone_num = dispatch_matrix.shape[0]

    # 区分服务订单和调度动作
    # demand_matrix: 满足真实需求的订单（服务订单）
    # dispatch_matrix: RL倾向的调度（可能包含服务+空驶）
    # 逻辑：demand_matrix中的非零值是服务订单，其余是调度

    for i in range(zone_num):
        for j in range(zone_num):
            if i == j:
                continue

            demand_count = int(demand_matrix[i][j]) if demand_matrix[i][j] > 0 else 0
            dispatch_count = int(dispatch_matrix[i][j]) if dispatch_matrix[i][j] > 0 else 0

            # 服务订单：demand_matrix中的值
            if demand_count > 0:
                served_orders.append((i, j, demand_count))

            # 调度动作：dispatch_matrix中超出demand的部分
            reposition_count = max(0, dispatch_count - demand_count)
            if reposition_count > 0:
                reposition_actions.append((i, j, reposition_count))

    return served_orders, reposition_actions, charging_actions
# ======修改结束
```

**Step 2: 验证函数逻辑**

创建单元测试 `test_parse_actions.py`:

```python
import numpy as np
from baseline_comparison import parse_rl_actions

# 测试用例
action_data = {
    'dispatch_orders': np.array([[0, 2, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    'demand_orders': np.array([[0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    'post_charging': {0: 2, 1: 1}
}

served, reposition, charging = parse_rl_actions(action_data)
print("Served orders:", served)
print("Reposition actions:", reposition)
print("Charging actions:", charging)

# 预期输出：
# Served orders: [(0, 1, 1), (0, 3, 1)]
# Reposition actions: [(0, 1, 1), (1, 0, 1)]
# Charging actions: {0: 2, 1: 1}
```

Run: `python test_parse_actions.py`
Expected: 输出符合预期

---

## Task 4: 修改动作文件生成逻辑

**Files:**
- Modify: `F:\Claude code\EVRL260123\dclily_torch_ddpg_demo0425\baseline_comparison.py:361-451`

**Step 1: 使用解析函数生成详细动作文件**

修改第361-451行的文件写入部分：

```python
# 保存动作到文件
action_file = 'action/RL-DDPG_actions.txt'
with open(action_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("RL-DDPG 策略 - 详细动作记录（测试episode）\n")
    f.write("="*80 + "\n\n")

    total_reward = sum([a['reward'] for a in detailed_actions])
    f.write(f"总步数: {len(detailed_actions)}\n")
    f.write(f"总奖励: {total_reward:.2f}\n\n")

    for action_data in detailed_actions:
        f.write(f"{'='*80}\n")
        f.write(f"时间步 {action_data['step']}\n")
        f.write(f"{'='*80}\n\n")

        # 解析动作
        served_orders, reposition_actions, charging_actions = parse_rl_actions(action_data)

        # 写入服务订单
        f.write("【服务订单】\n")
        if served_orders:
            for from_zone, to_zone, count in served_orders:
                f.write(f"  区域{from_zone} → 区域{to_zone}: {count}辆\n")
        else:
            f.write("  无\n")
        f.write("\n")

        # 写入调度动作
        f.write("【调度动作】\n")
        if reposition_actions:
            for from_zone, to_zone, count in reposition_actions:
                f.write(f"  区域{from_zone} → 区域{to_zone}: {count}辆\n")
        else:
            f.write("  无\n")
        f.write("\n")

        # 写入充电动作
        f.write("【充电动作】\n")
        if charging_actions:
            for zone in sorted(charging_actions.keys()):
                f.write(f"  区域{zone}: {charging_actions[zone]}辆\n")
        else:
            f.write("  无\n")
        f.write("\n")

print(f"[INFO] 动作文件已保存到: {action_file}")
```

**Step 2: 运行完整测试**

Run: `python baseline_comparison.py`
Expected: 生成 `action/RL-DDPG_actions.txt`，格式与MILP一致

**Step 3: 对比输出格式**

手动对比：
- `action/RL-DDPG_actions.txt`
- `action/global_optimal_solution_actions.txt`

验证格式一致性

---

## Task 5: 清理和文档更新

**Files:**
- Delete: `test_info_content.py`
- Delete: `test_parse_actions.py`
- Modify: `F:\Claude code\EVRL260123\dclily_torch_ddpg_demo0425\checkpoint\claude.md`

**Step 1: 删除测试文件**

```bash
rm test_info_content.py test_parse_actions.py
```

**Step 2: 更新claude.md文档**

在 `checkpoint/claude.md` 中添加：

```markdown
## 2026.2.4 - RL动作解析优化

**问题：**
- RL动作文件只有动作向量，无法与MILP对比

**解决方案：**
- 从环境info字典提取 `dispatch_orders` 和 `demand_orders` 矩阵
- 通过对比两个矩阵区分服务订单和调度动作
- 从环境状态提取充电信息
- 生成与MILP格式一致的动作文件

**修改文件：**
- `baseline_comparison.py:105-140` - 添加 `parse_rl_actions` 函数
- `baseline_comparison.py:337-451` - 修改动作记录和文件生成逻辑
```

**Step 3: 验证最终结果**

Run: `python baseline_comparison.py`
Expected:
- 程序正常运行
- 生成格式正确的动作文件
- 可以与MILP动作文件对比

---

## 验收标准

1. ✅ `action/RL-DDPG_actions.txt` 包含服务订单、调度动作、充电动作三个部分
2. ✅ 格式与 `action/global_optimal_solution_actions.txt` 一致
3. ✅ 每个时间步的动作信息完整
4. ✅ 代码有清晰的注释说明
5. ✅ 文档已更新

---

## 注意事项

1. **数据结构验证**：Task 1必须先验证info字典的实际内容
2. **矩阵含义**：需要确认dispatch_orders和demand_orders是车辆数还是距离
3. **充电状态**：充电信息从 `test_env.car[i][3]` 获取（charger_id）
4. **边界情况**：处理矩阵为None或空的情况
