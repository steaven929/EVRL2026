# MARL 多智能体强化学习使用指南

## 创建日期
2026-02-19

## 目录
1. [MARL文件结构](#marl文件结构)
2. [快速开始](#快速开始)
3. [参数调整方法](#参数调整方法)
4. [输出文件说明](#输出文件说明)
5. [与单智能体对比](#与单智能体对比)
6. [常见问题](#常见问题)

---

## MARL文件结构

### 核心文件

#### 1. 主程序
- **main_marl.py**: MARL训练主程序
  - 强制使用multi模式
  - 自动生成完整可视化图像
  - 保存所有训练数据

#### 2. 模型文件
- **specific/marl_model.py**: MARL网络模型
  - `SharedActor`: 参数共享的Actor网络
    - 输入: 6维局部观测 + agent_id embedding(16维)
    - 输出: 3维动作 [depart_pref, dest_pref, charge_pref]
  - `CentralizedCritic`: 集中式Critic网络
    - 输入: 251维全局状态 + 150维联合动作
    - 输出: Q值

#### 3. Agent实现
- **core/marl_agent.py**: MARL Agent类
  - `IPPOAgent`: IPPO算法实现
  - 参数共享 + 集中式Critic
  - 全局奖励共享（Phase-1）

#### 4. 训练循环
- **core/marl_training.py**: 训练和测试函数
  - `marl_train()`: 多智能体训练循环
  - `marl_test()`: 多智能体测试循环
  - 收集RLD、SVC、SRD等指标

#### 5. 经验池
- **core/buffer.py**: 新增MARLReplayBuffer类
  - 支持joint-transition存储
  - 容量: 64000（与单智能体相同）

#### 6. 环境接口
- **specific/dclily_elec.py**: 新增多智能体接口
  - `get_agent_obs(agent_id)`: 获取7维局部观测
  - `get_global_state()`: 获取251维全局状态
  - `step()`: 支持joint action输入

#### 7. 配置文件
- **core/util.py**: 新增MARL参数
  - `--marl_mode`: single/multi
  - `--n_agents`: 智能体数量
  - `--share_policy`: 是否参数共享

#### 8. 测试脚本
- **test_marl.py**: MARL冒烟测试
  - 验证所有组件正常工作

---

## 快速开始

### 1. 冒烟测试（验证安装）
```bash
python test_marl.py
```

**预期输出:**
```
✓ 所有测试通过！MARL实现正常工作。
```

### 2. 快速测试（1轮训练）
```bash
python main_marl.py --train_eps 1 --test_eps 1
```

**用途:** 验证MARL能正常运行，检查是否有错误

### 3. 短程训练（10轮）
```bash
python main_marl.py --train_eps 10 --test_eps 3
```

**用途:** 初步验证学习能力，观察指标变化趋势

### 4. 完整训练（100轮）
```bash
python main_marl.py --train_eps 100 --test_eps 10
```

**用途:** 完整训练，与单智能体基线对比

---

## 参数调整方法

### 方法1: 命令行参数（推荐）

**优点:** 不需要修改代码，灵活方便

#### 基础参数
```bash
# 训练轮数
python main_marl.py --train_eps 50 --test_eps 10

# 批大小
python main_marl.py --train_eps 100 --batch_size 256

# 学习率
python main_marl.py --train_eps 100 --actor_lr 1e-5 --critic_lr 2e-5

# 折扣因子
python main_marl.py --train_eps 100 --gamma 0.99
```

#### 网络结构
```bash
# 增大网络容量
python main_marl.py --train_eps 100 \
  --actor_hidden_dim 512 \
  --critic_hidden_dim 512

# 调整embedding维度
python main_marl.py --train_eps 100 \
  --actor_embed_dim 16 \
  --critic_embed_dim 16
```

#### 组合调整
```bash
# 完整参数调整示例
python main_marl.py \
  --train_eps 150 \
  --test_eps 10 \
  --batch_size 256 \
  --actor_lr 1e-5 \
  --critic_lr 2e-5 \
  --gamma 0.99 \
  --actor_hidden_dim 512 \
  --critic_hidden_dim 512
```

### 方法2: 修改 core/util.py 默认值

**优点:** 永久改变默认值，不需要每次输入

**文件位置:** `core/util.py` 第13-46行

#### 常用参数位置
```python
# 第20行: 训练轮数
parser.add_argument('--train_eps', default=1000, type=int)

# 第21行: 测试轮数
parser.add_argument('--test_eps', default=10, type=int)

# 第27行: 批大小
parser.add_argument('--batch_size', default=128, type=int)

# 第24行: Critic学习率
parser.add_argument('--critic_lr', default=3e-5, type=float)

# 第25行: Actor学习率
parser.add_argument('--actor_lr', default=2e-5, type=float)

# 第23行: 折扣因子
parser.add_argument('--gamma', default=0.95, type=float)

# 第30-31行: 网络隐藏层维度
parser.add_argument('--critic_hidden_dim', default=256, type=int)
parser.add_argument('--actor_hidden_dim', default=256, type=int)

# 第43-45行: MARL专用参数
parser.add_argument('--marl_mode', default='single', type=str)
parser.add_argument('--n_agents', default=50, type=int)
parser.add_argument('--share_policy', default=True, type=bool)
```

#### 修改示例
```python
# 将默认训练轮数改为50
parser.add_argument('--train_eps', default=50, type=int)

# 将默认批大小改为256
parser.add_argument('--batch_size', default=256, type=int)
```

### 所有可调参数列表

| 参数名 | 默认值 | 说明 | 推荐范围 |
|--------|--------|------|----------|
| `--train_eps` | 1000 | 训练轮数 | 10-200 |
| `--test_eps` | 10 | 测试轮数 | 3-10 |
| `--batch_size` | 128 | 批大小 | 64-512 |
| `--actor_lr` | 2e-5 | Actor学习率 | 1e-6 ~ 1e-4 |
| `--critic_lr` | 3e-5 | Critic学习率 | 1e-6 ~ 1e-4 |
| `--gamma` | 0.95 | 折扣因子 | 0.9-0.99 |
| `--tau` | 5e-4 | 软更新参数 | 1e-4 ~ 1e-3 |
| `--actor_hidden_dim` | 256 | Actor隐藏层 | 128-1024 |
| `--critic_hidden_dim` | 256 | Critic隐藏层 | 128-1024 |
| `--actor_embed_dim` | 8 | Actor embedding | 8-32 |
| `--critic_embed_dim` | 8 | Critic embedding | 8-32 |
| `--memory_capacity` | 64000 | 经验池容量 | 10000-100000 |
| `--n_agents` | 50 | 智能体数量 | 固定50（区域数） |
| `--share_policy` | True | 参数共享 | True（Phase-1） |

---

## 输出文件说明

### 训练完成后的输出目录
```
./outputs/marl_YYYYMMDD-HHMMSS/
├── final_metrics.png          # 8个子图的综合指标
├── critic_loss_curve.png      # Critic损失曲线
├── train_rewards.png          # 训练奖励曲线
├── critic_losses.npy          # Critic损失数据
├── actor_losses.npy           # Actor损失数据
├── rewards.npy                # 训练奖励数据
├── ma_rewards.npy             # 移动平均奖励
├── rld_history.npy            # 响应距离历史
├── svc_history.npy            # 服务率历史
├── srd_history.npy            # 服务奖励比历史
├── marl_model_final.pth       # 最终模型
└── marl_model_ep*.pth         # 定期保存的模型（每100轮）
```

### 可视化图像说明

#### final_metrics.png（8个子图）
1. **Critic Loss**: 应该下降，表示网络在学习
2. **Actor Loss (-Q)**: 负的Q值，应该下降（Q值上升）
3. **Q Value**: MARL中标记为N/A（不单独记录）
4. **Target Q Value**: MARL中标记为N/A（不单独记录）
5. **RLD (响应距离)**: 调度距离，越小越好
6. **SVC (服务率)**: 服务完成率，越高越好（目标>10%）
7. **SRD (服务奖励比)**: 服务收益比，越高越好
8. **Episode Reward**: 训练奖励曲线，应该上升

### 数据分析示例
```python
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
rewards = np.load('./outputs/marl_YYYYMMDD-HHMMSS/rewards.npy')
svc_history = np.load('./outputs/marl_YYYYMMDD-HHMMSS/svc_history.npy')
critic_losses = np.load('./outputs/marl_YYYYMMDD-HHMMSS/critic_losses.npy')

# 分析
print(f"平均奖励: {np.mean(rewards):.2f}")
print(f"最终奖励: {rewards[-1]:.2f}")
print(f"平均服务率: {np.mean(svc_history):.4f}")
print(f"最终服务率: {svc_history[-1]:.4f}")

# 绘图
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(rewards)
plt.title('Rewards')
plt.subplot(1, 3, 2)
plt.plot(svc_history)
plt.title('Service Rate')
plt.subplot(1, 3, 3)
plt.plot(critic_losses)
plt.title('Critic Loss')
plt.show()
```

---

## 与单智能体对比

### 运行对比实验

#### 1. 运行单智能体基线
```bash
python main.py --train_eps 100 --test_eps 10
```

**输出:** 当前目录下的 `final_metrics.png`, `rewards.npy` 等

#### 2. 运行MARL
```bash
python main_marl.py --train_eps 100 --test_eps 10
```

**输出:** `./outputs/marl_YYYYMMDD-HHMMSS/` 目录下的文件

#### 3. 对比分析
```python
import numpy as np

# 加载单智能体结果
single_rewards = np.load('rewards.npy')
single_svc = np.load('svc_history.npy')

# 加载MARL结果
marl_rewards = np.load('./outputs/marl_YYYYMMDD-HHMMSS/rewards.npy')
marl_svc = np.load('./outputs/marl_YYYYMMDD-HHMMSS/svc_history.npy')

# 对比
print("="*60)
print("单智能体 vs MARL 对比")
print("="*60)
print(f"平均奖励:")
print(f"  单智能体: {np.mean(single_rewards):.2f}")
print(f"  MARL:     {np.mean(marl_rewards):.2f}")
print(f"  提升:     {(np.mean(marl_rewards)/np.mean(single_rewards)-1)*100:.2f}%")
print(f"\n平均服务率:")
print(f"  单智能体: {np.mean(single_svc):.4f}")
print(f"  MARL:     {np.mean(marl_svc):.4f}")
print(f"  提升:     {(np.mean(marl_svc)/np.mean(single_svc)-1)*100:.2f}%")
print("="*60)
```

### 关键对比指标

| 指标 | 单智能体 | MARL | 说明 |
|------|----------|------|------|
| 服务率 (SVC) | 目标>10% | 目标>10% | 最重要指标 |
| 训练奖励 | 基线 | 对比 | 越高越好 |
| 充电行为 | 检查是否黑客 | 检查是否改善 | 避免过度充电 |
| 训练稳定性 | 检查方差 | 检查方差 | 越稳定越好 |

---

## 常见问题

### Q1: MARL训练很慢怎么办？
**A:** MARL比单智能体慢是正常的，因为需要采样50个智能体的动作。

**解决方案:**
- 先用少量轮数测试（1-10轮）
- 减少batch_size（但可能影响稳定性）
- 使用GPU: `--device cuda`（需要CUDA环境）

### Q2: 服务率仍然是0%怎么办？
**A:** 如果MARL也出现服务率崩溃，说明是奖励函数问题。

**解决方案:**
1. 检查奖励函数设置（specific/dclily_elec.py）
2. 大幅提升服务奖励权重（×20或更高）
3. 降低或移除服务相关惩罚
4. 考虑升级到Phase-2（差分奖励）

### Q3: 如何加载已训练的模型？
**A:** 修改main_marl.py，在创建agent后添加：

```python
# 在 main_marl.py 的 main() 函数中
env, agent = env_agent_config(cfg)

# 加载已训练模型
agent.load_model('./outputs/marl_YYYYMMDD-HHMMSS/marl_model_final.pth')
print("已加载预训练模型")

# 继续训练或测试
```

### Q4: 如何只运行测试（不训练）？
**A:** 设置train_eps=0：

```bash
python main_marl.py --train_eps 0 --test_eps 10
```

**注意:** 需要先加载已训练的模型（见Q3）

### Q5: 训练中断了怎么办？
**A:** MARL会定期保存模型（每100轮）。

**恢复训练:**
1. 找到最近保存的模型: `marl_model_ep100.pth`
2. 加载模型（见Q3）
3. 继续训练剩余轮数

### Q6: 如何切换回单智能体？
**A:** 使用原来的main.py：

```bash
python main.py --train_eps 100
```

或者在main_marl.py中设置：
```python
cfg['marl_mode'] = 'single'
```

### Q7: 内存不足怎么办？
**A:** 减少经验池容量和batch size：

```bash
python main_marl.py --train_eps 100 \
  --memory_capacity 32000 \
  --batch_size 64
```

### Q8: 如何调试MARL？
**A:** 使用test_marl.py查看详细信息：

```python
# 在test_marl.py中添加调试代码
obs = env.get_agent_obs(0)
print(f"Agent 0 observation: {obs}")

actions = agent.sample_action(obs_n, add_noise=False)
print(f"Actions shape: {actions.shape}")
print(f"Actions mean: {actions.mean()}, std: {actions.std()}")
```

---

## 进阶使用

### Phase-2 升级路线

如果Phase-1成功（服务率>10%，充电行为正常），可以升级到Phase-2：

#### 升级内容
1. **差分奖励**: 实现credit assignment
   - 每个agent获得独立的奖励信号
   - 避免全局奖励共享的问题

2. **价值归一化**: 稳定训练
   - 归一化Q值和优势函数
   - 减少训练方差

3. **完整MAPPO**: 使用GAE和PPO clip
   - 更稳定的策略更新
   - 更好的样本效率

4. **邻区信息**: 扩展局部观测
   - 添加邻近区域的信息
   - 提升协作能力

#### 升级触发条件
- IPPO在50区指标达到或超过单智能体基线
- 充电黑客问题得到解决
- 训练稳定收敛

---

## 技术支持

### 相关文档
- `checkpoint/marl_phase1_implementation_summary.md`: 完整实现文档
- `checkpoint/marl_migration_phase1_plan(1).md`: 原始设计方案
- `MARL_USAGE_GUIDE.md`: 英文使用指南（项目根目录）

### 问题反馈
如有问题，请检查：
1. test_marl.py 是否通过
2. 训练日志中的错误信息
3. final_metrics.png 中的指标趋势

---

## 版本历史

### v1.0 (2026-02-19)
- ✓ 完成MARL Phase-1实现
- ✓ 50个区域级智能体
- ✓ IPPO算法 + 参数共享
- ✓ 集中式Critic
- ✓ 完整可视化支持
- ✓ 与单智能体兼容

### 待开发功能
- Phase-2: 差分奖励
- Phase-2: 价值归一化
- Phase-2: 完整MAPPO
- Phase-2: 邻区信息
