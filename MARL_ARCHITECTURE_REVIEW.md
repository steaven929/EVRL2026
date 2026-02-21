# MARL架构深度分析报告

## 执行摘要

经过800轮训练，系统表现为：奖励平稳无增长，Critic Loss巨大（1.83e9），Actor Loss异常（负值）。根本原因是**奖励缩放失效**导致Q值过大，学习极慢。

---

## 1. 核心问题：奖励缩放失效

### 问题描述
- **配置值**：`reward_scale=1000.0`（util.py line 36）
- **实际值**：`reward_scale=1.0`（训练时默认值）
- **影响**：奖励未缩放，导致Q值爆炸

### 数据证据
```
原始奖励：~1,000,000 / episode
期望缩放后：~1,000 / episode (÷1000)
实际存储：~1,015,075 / episode (未缩放！)

理论Q值（无缩放）：~575,000
Critic Loss：1.83e9
隐含Q值误差：√(1.83e9) ≈ 42,778
```

### 根本原因
`marl_training.py` line 41:
```python
reward_scale = float(cfg.get('reward_scale', 1.0))  # 默认1.0！
```

如果cfg中没有'reward_scale'键，默认为1.0，导致缩放失效。

### 修复方案
1. **立即修复**：将reward_scale提高到10000或更高
2. **验证传递**：确保cfg字典包含reward_scale参数

---

## 2. 网络架构问题

### 2.1 SharedActor（参数共享Actor）

**设计**：
- 输入：局部观测(6维) + agent_id_embedding(16维)
- 结构：22维 → 128 → 64 → 3维
- 输出：sigmoid激活，范围[0,1]

**问题**：
1. **网络容量不足**：
   - 50个智能体共享一个网络
   - 隐藏层只有128→64，对于50个区域的复杂决策可能不够
   - 建议：256→128或更大

2. **Agent ID Embedding维度小**：
   - 当前16维，需要区分50个区域
   - 建议：32维或64维

3. **无Batch Normalization**：
   - 输入特征尺度可能差异大（OD需求、SOC、车辆数等）
   - 建议：添加BatchNorm或LayerNorm

### 2.2 CentralizedCritic（集中式Critic）

**设计**：
- 输入：global_state(251维) + joint_action(50×3=150维) = 401维
- 结构：401 → 256 → 256 → 1
- 输出：Q值（标量）

**问题**：
1. **输入维度巨大**（401维）：
   - 第一层Linear(401, 256)参数量：401×256 = 102,656
   - 可能导致过拟合或学习困难
   - 建议：增加隐藏层维度到512或1024

2. **无残差连接**：
   - 深度网络（3层）可能梯度消失
   - 建议：添加skip connection

3. **初始化权重过小**（init_w=3e-3）：
   - 最后一层权重初始化在[-0.003, 0.003]
   - Q值初始估计接近0，但实际Q值是575K
   - 导致初期学习极慢
   - 建议：使用Xavier或He初始化

---

## 3. 训练循环问题

### 3.1 更新频率

**当前设计**：
```python
if len(agent.memory) >= cfg['batch_size']:
    critic_loss, actor_loss = agent.update()
```

**问题**：
- 每个timestep都尝试更新（如果buffer够大）
- 26 steps/episode × 800 episodes = 20,800次更新
- 但batch_size=128，前128步才开始更新
- **更新过于频繁**，可能导致过拟合单个episode的经验

**建议**：
- 每N步更新一次（如N=4或N=10）
- 或每个episode结束后更新K次（如K=26）

### 3.2 探索噪声衰减

**当前设计**：
```python
noise_scale = max(0.1, 0.3 * (1 - i_ep / cfg['train_eps']))
```

**问题**：
- 线性衰减，800轮后仍有10%噪声
- 可能导致后期探索过多，无法收敛
- 建议：指数衰减或更快衰减

---

## 4. Loss计算问题

### 4.1 Critic Loss

**当前实现**：
```python
critic_loss = self.critic_criterion(current_q, target_q)
# MSE Loss
```

**问题**：
- MSE对大Q值敏感：loss = (Q_pred - Q_target)²
- 当Q值是575K时，即使1%误差也会导致巨大loss
- 575K × 1% = 5,750，loss = 5,750² = 33M

**建议**：
1. **Huber Loss**：对大误差不敏感
2. **Q值归一化**：除以running std
3. **Gradient Clipping**：限制梯度范数

### 4.2 Actor Loss

**当前实现**：
```python
actor_loss = -self.critic(global_state_batch, joint_action_flat).mean()
```

**问题**：
- Actor loss = -Q值
- 当Q值是575K时，actor_loss = -575K
- 梯度会非常大，导致更新不稳定
- 实际观测：actor_loss = -28K到-54K（异常大）

**建议**：
1. **归一化Q值**：actor_loss = -Q / Q_std
2. **限制梯度**：torch.nn.utils.clip_grad_norm_()
3. **使用advantage**：actor_loss = -(Q - baseline)

---

## 5. 超参数问题

### 5.1 学习率（已修复）

**原始值**：
- critic_lr = 1e-5（太小）
- actor_lr = 5e-6（太小）

**修复后**：
- critic_lr = 1e-4（提高10x）
- actor_lr = 5e-5（提高10x）

**评估**：修复合理，但可能还需要根据reward_scale调整

### 5.2 Batch Size

**当前值**：128

**问题**：
- 对于50个智能体的联合动作空间（150维），128可能偏小
- 梯度估计方差大
- 建议：256或512

### 5.3 Tau（软更新系数）

**当前值**：5e-4

**问题**：
- 太小，target网络更新太慢
- 当主网络学习时，target网络滞后严重
- 建议：1e-3或5e-3

### 5.4 Gamma（折扣因子）

**当前值**：0.95

**评估**：
- 26 timesteps，discount sum = 14.73
- 合理，但可以尝试0.99（更重视长期奖励）

---

## 6. 环境接口问题

### 6.1 局部观测维度

**当前设计**：7维（含agent_id）
```
[od_in, od_out, vehicle_count, avg_soc, charger_occ, timestep_norm, agent_id]
```

**问题**：
1. **信息不足**：
   - 缺少邻近区域信息（空间相关性）
   - 缺少历史信息（时间相关性）
   - 建议：添加邻近区域的OD需求、车辆数

2. **特征尺度差异大**：
   - od_in/od_out：可能0-100
   - vehicle_count：0-20
   - avg_soc：0-1
   - timestep_norm：0-1
   - 建议：标准化所有特征

### 6.2 全局状态维度

**当前设计**：251维
```
[0:100]   OD demand (50 zones × 2)
[100:150] Charger states (50 zones)
[150:200] Avg SOC per zone (50 zones)
[200:250] Vehicle counts per zone (50 zones)
[250]     Current timestep
```

**评估**：设计合理，包含了必要的全局信息

---

## 7. 算法选择问题

### 7.1 当前算法：IPPO + Centralized Critic

**优点**：
- 参数共享，减少参数量
- Centralized Critic利用全局信息

**问题**：
1. **IPPO不是最优选择**：
   - IPPO是on-policy算法，需要大量样本
   - 当前使用replay buffer（off-policy），算法不匹配
   - 建议：改用MADDPG（off-policy）或真正的PPO（去掉replay buffer）

2. **无Credit Assignment**：
   - 所有智能体共享全局奖励
   - 无法区分哪个智能体的贡献
   - 建议：Phase-2实现局部奖励或value decomposition

### 7.2 Target Network更新

**当前实现**：软更新，每次update都更新

**问题**：
- 与DDPG一致，但对于IPPO不是标准做法
- PPO通常不使用target network
- 建议：如果用PPO，去掉target network；如果用DDPG，保持当前设计

---

## 8. 优先级修复建议

### 🔴 P0（立即修复）

1. **奖励缩放**：
   - 将reward_scale提高到10000
   - 验证cfg传递正确

2. **Critic Loss归一化**：
   - 使用Huber Loss替代MSE
   - 或对Q值进行归一化

3. **梯度裁剪**：
   - 添加gradient clipping（max_norm=1.0）

### 🟡 P1（短期优化）

4. **网络容量**：
   - Actor: 128→256, 64→128
   - Critic: 256→512
   - Agent embedding: 16→32

5. **Batch Normalization**：
   - 在Actor和Critic的第一层后添加

6. **更新频率**：
   - 改为每4步更新一次

### 🟢 P2（长期改进）

7. **算法一致性**：
   - 决定使用MADDPG（off-policy）还是PPO（on-policy）
   - 如果PPO，去掉replay buffer，使用GAE

8. **Credit Assignment**：
   - 实现局部奖励或QMIX/VDN

9. **特征工程**：
   - 添加邻近区域信息
   - 标准化所有输入特征

---

## 9. 预期效果

### 修复P0后（reward_scale + loss归一化 + gradient clipping）：

**预期指标**：
- Critic Loss：从1.83e9降到1e4-1e6
- Actor Loss：从-28K降到-10到-100
- 奖励趋势：100轮内开始上升
- 学习速度：提升10-100倍

### 修复P1后（网络容量 + BatchNorm + 更新频率）：

**预期指标**：
- 收敛速度：200-300轮收敛
- 最终性能：接近或超过单智能体baseline
- 服务率：从2%提升到10-30%

### 修复P2后（算法一致性 + Credit Assignment）：

**预期指标**：
- 最终性能：显著超过单智能体baseline
- 服务率：30-50%
- 多智能体协作：明显的区域间协调行为

---

## 10. 测试计划

### 阶段1：验证P0修复（100轮）
1. 修改reward_scale=10000
2. 添加Huber Loss
3. 添加gradient clipping
4. 训练100轮，观察：
   - Critic Loss是否降到1e6以下
   - 奖励是否开始上升
   - Actor Loss是否稳定

### 阶段2：应用P1优化（200轮）
1. 增加网络容量
2. 添加BatchNorm
3. 调整更新频率
4. 训练200轮，观察：
   - 是否收敛
   - 最终性能vs baseline

### 阶段3：长期改进（500轮）
1. 重构算法（MADDPG或真PPO）
2. 实现Credit Assignment
3. 完整训练，评估最终性能

---

## 总结

**根本问题**：奖励缩放失效导致Q值爆炸，学习极慢

**关键修复**：
1. reward_scale: 1000 → 10000
2. Huber Loss替代MSE
3. Gradient Clipping

**架构优化**：
1. 增加网络容量
2. 添加BatchNorm
3. 调整更新频率

**长期方向**：
1. 算法一致性（MADDPG vs PPO）
2. Credit Assignment
3. 特征工程

修复P0后，预期100轮内看到明显学习效果。
