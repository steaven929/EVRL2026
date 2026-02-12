# Hyperparameter Comparison Table

| Commit | Message | train_eps | gamma | critic_lr | actor_lr | batch_size | Image |
|--------|---------|-----------|-------|-----------|----------|------------|-------|
| 8c13a6d | 即时充电奖励+eps=400+reward=-3700 | 400 | **0.85** | 5e-4 | 5e-5 | 64 | 8c13a6d_final_metrics.png |
| fe61c47 | 移除充电权重+gamma=0.85 | 700 | **0.85** | 5e-4 | 5e-5 | 64 | fe61c47_final_metrics.png |
| 914201b | 移除环境充电权重+修改充电阈值 | 700 | 0.5 | 5e-4 | 5e-5 | 64 | 914201b_final_metrics.png |
| 265e729 | 最优解版本 | **10** | 0.5 | 5e-4 | 5e-5 | 64 | 265e729_final_metrics.png |
| cbe40fc | 最优解版本 | **2000** | 0.5 | 5e-4 | 5e-5 | 64 | cbe40fc_final_metrics.png |
| 18fc161 | 最优解版本 | 700 | 0.5 | 5e-4 | 5e-5 | 64 | 18fc161_final_metrics.png |

## Analysis

### Variable Parameters
1. **train_eps**: 10, 400, 700, 2000
2. **gamma**: 0.5, 0.85

### Constant Parameters
- critic_lr: 5e-4
- actor_lr: 5e-5
- batch_size: 64
- memory_capacity: 8000
- target_update: 2
- tau: 5e-3
- critic_hidden_dim: 256
- actor_hidden_dim: 256
- seed: 2
