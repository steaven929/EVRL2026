# MARL Phase-1 使用指南

## 快速开始

### 1. 运行冒烟测试
验证MARL实现是否正常工作：
```bash
python test_marl.py
```

预期输出：
```
✓ 所有测试通过！MARL实现正常工作。
```

### 2. 运行短程训练（推荐）
先运行10轮训练验证可学习性：
```bash
python main_marl.py --train_eps 10 --test_eps 3
```

### 3. 运行完整训练
如果短程训练正常，运行完整训练：
```bash
python main_marl.py --train_eps 100 --test_eps 10
```

## 配置参数

### MARL专用参数
- `--marl_mode`: 模式选择
  - `single`: 单智能体模式（默认）
  - `multi`: 多智能体模式
- `--n_agents`: 智能体数量（默认50，对应50个区域）
- `--share_policy`: 是否参数共享（默认True）

### 通用参数
- `--train_eps`: 训练轮数（默认100）
- `--test_eps`: 测试轮数（默认10）
- `--batch_size`: 批大小（默认128）
- `--actor_lr`: Actor学习率（默认2e-5）
- `--critic_lr`: Critic学习率（默认3e-5）
- `--gamma`: 折扣因子（默认0.95）

## 输出文件

训练完成后，会在`./outputs/marl_YYYYMMDD-HHMMSS/`目录下生成：
- `marl_model_final.pth`: 最终模型
- `marl_model_ep*.pth`: 定期保存的模型（每100轮）
- `marl_rewards.npy`: 训练奖励
- `marl_ma_rewards.npy`: 移动平均奖励
- `marl_critic_losses.npy`: Critic损失
- `marl_actor_losses.npy`: Actor损失
- `train_rewards.png`: 训练曲线图
- `marl_test_rewards.npy`: 测试奖励

## 与单智能体对比

### 运行单智能体基线
```bash
python main.py --train_eps 100 --test_eps 10
```

### 对比指标
1. **训练奖励**：MARL vs 单智能体
2. **服务率**：检查是否改善
3. **充电行为**：是否解决充电黑客问题
4. **训练稳定性**：损失曲线是否收敛

## 常见问题

### Q1: 训练很慢怎么办？
A: MARL训练比单智能体慢，因为需要采样50个智能体的动作。可以：
- 减少训练轮数（先验证可学习性）
- 减少batch_size（但可能影响稳定性）
- 使用GPU（修改`--device cuda`）

### Q2: 奖励不收敛怎么办？
A: 可能原因：
- 学习率过高/过低：调整`--actor_lr`和`--critic_lr`
- 探索不足：增加噪声衰减周期
- 奖励函数问题：检查是否仍有充电黑客

### Q3: 如何加载已训练模型？
A: 修改`main_marl.py`，在创建agent后添加：
```python
agent.load_model('./outputs/marl_YYYYMMDD-HHMMSS/marl_model_final.pth')
```

### Q4: 如何切换回单智能体？
A: 使用原有的`main.py`：
```bash
python main.py --train_eps 100
```
或者在`main_marl.py`中设置：
```python
cfg['marl_mode'] = 'single'
```

## 调试技巧

### 1. 检查局部观测
在`test_marl.py`中查看第0个智能体的观测：
```python
obs = env.get_agent_obs(0)
print(f"Agent 0 observation: {obs}")
```

### 2. 检查动作分布
在训练循环中添加：
```python
print(f"Action mean: {actions.mean()}, std: {actions.std()}")
```

### 3. 检查损失曲线
训练后绘制损失曲线：
```python
import matplotlib.pyplot as plt
import numpy as np

critic_losses = np.load('./outputs/.../marl_critic_losses.npy')
actor_losses = np.load('./outputs/.../marl_actor_losses.npy')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(critic_losses)
plt.title('Critic Loss')
plt.subplot(1, 2, 2)
plt.plot(actor_losses)
plt.title('Actor Loss')
plt.show()
```

## Phase-2升级路线

如果Phase-1成功（服务率>10%，充电行为正常），可以升级到Phase-2：

### 升级内容
1. **差分奖励**：实现credit assignment
2. **价值归一化**：稳定训练
3. **完整MAPPO**：使用GAE和PPO clip
4. **邻区信息**：扩展局部观测

### 升级触发条件
- IPPO在50区指标达到或超过单智能体基线
- 充电黑客问题得到解决
- 训练稳定收敛

## 技术支持

如有问题，请查看：
- `checkpoint/marl_phase1_implementation_summary.md`: 完整实现文档
- `checkpoint/marl_migration_phase1_plan(1).md`: 原始设计方案
- `test_marl.py`: 测试代码示例
