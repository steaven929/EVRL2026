# Final Metrics Collection Summary

## Overview
Collected 6 versions of final_metrics.png with corresponding hyperparameters from git history.

---

## Version 1: 8c13a6d
**Commit Message:** 即时充电奖励+eps=400+reward=-3700

### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| train_eps | 400 |
| gamma | **0.85** |
| critic_lr | 5e-4 |
| actor_lr | 5e-5 |
| batch_size | 64 |

**Image:** `8c13a6d_final_metrics.png`

---

## Version 2: fe61c47
**Commit Message:** 移除充电权重+gamma=0.85

### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| train_eps | 700 |
| gamma | **0.85** |
| critic_lr | 5e-4 |
| actor_lr | 5e-5 |
| batch_size | 64 |

**Image:** `fe61c47_final_metrics.png`

---

## Version 3: 914201b
**Commit Message:** 移除环境充电权重+修改充电阈值

### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| train_eps | 700 |
| gamma | 0.5 |
| critic_lr | 5e-4 |
| actor_lr | 5e-5 |
| batch_size | 64 |

**Image:** `914201b_final_metrics.png`

---

## Version 4: 265e729
**Commit Message:** 最优解版本

### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| train_eps | **10** |
| gamma | 0.5 |
| critic_lr | 5e-4 |
| actor_lr | 5e-5 |
| batch_size | 64 |

**Image:** `265e729_final_metrics.png`

---

## Version 5: cbe40fc
**Commit Message:** 最优解版本

### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| train_eps | **2000** |
| gamma | 0.5 |
| critic_lr | 5e-4 |
| actor_lr | 5e-5 |
| batch_size | 64 |

**Image:** `cbe40fc_final_metrics.png`

---

## Version 6: 18fc161
**Commit Message:** 最优解版本

### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| train_eps | 700 |
| gamma | 0.5 |
| critic_lr | 5e-4 |
| actor_lr | 5e-5 |
| batch_size | 64 |

**Image:** `18fc161_final_metrics.png`

---

## Key Differences Summary

### Training Episodes (train_eps)
- 8c13a6d: 400
- fe61c47: 700
- 914201b: 700
- 265e729: **10** (minimal)
- cbe40fc: **2000** (maximum)
- 18fc161: 700

### Discount Factor (gamma)
- 8c13a6d: **0.85** (high)
- fe61c47: **0.85** (high)
- 914201b: 0.5
- 265e729: 0.5
- cbe40fc: 0.5
- 18fc161: 0.5

### Notes
- All versions use same learning rates (critic: 5e-4, actor: 5e-5)
- All versions use batch_size=64, memory_capacity=8000
- Network architecture consistent: hidden_dim=256, embed_dim=8
- Main variations: **train_eps** and **gamma**
