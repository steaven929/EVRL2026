# 超参数调优日志

记录所有超参数实验的配置和结果

---


## baseline_600eps
**时间**: 2026-02-10 21:42:27
**描述**: 基线配置，train_eps=600
**状态**: ✓ 成功

### 超参数配置 (从 util.py 读取)
```
train_eps: 600
gamma: 0.5
critic_lr: 5e-4
actor_lr: 5e-5
batch_size: 64
memory_capacity: 8000
target_update: 2
tau: 5e-3
critic_hidden_dim: 256
actor_hidden_dim: 256
seed: 2
```

**结果路径**: `tuning_results/baseline_600eps/`
**训练图像**: `tuning_results/baseline_600eps/final_metrics.png`

---


## baseline_600eps
**时间**: 2026-02-10 21:43:59
**描述**: 基线配置，train_eps=600
**状态**: ✓ 成功

### 超参数配置 (从 util.py 读取)
```
train_eps: 600
gamma: 0.85
critic_lr: 5e-4
actor_lr: 5e-5
batch_size: 128
memory_capacity: 8000
target_update: 2
tau: 5e-3
critic_hidden_dim: 256
actor_hidden_dim: 256
seed: 2
```

**结果路径**: `tuning_results/baseline_600eps/`
**训练图像**: `tuning_results/baseline_600eps/final_metrics.png`

---


## high_gamma_large_batch
**时间**: 2026-02-10 21:55:57
**描述**: 高gamma+大批量，增强长期规划
**状态**: ✓ 成功

### 超参数配置 (从 util.py 读取)
```
train_eps: 600
gamma: 0.85
critic_lr: 5e-4
actor_lr: 5e-5
batch_size: 128
memory_capacity: 8000
target_update: 2
tau: 5e-3
critic_hidden_dim: 256
actor_hidden_dim: 256
seed: 2
```

**结果路径**: `tuning_results/high_gamma_large_batch/`
**训练图像**: `tuning_results/high_gamma_large_batch/final_metrics.png`

---


## high_gamma_large_batch
**时间**: 2026-02-10 21:57:04
**描述**: 高gamma+大批量，增强长期规划
**状态**: ✓ 成功

### 超参数配置 (从 util.py 读取)
```
train_eps: 600
gamma: 0.5
critic_lr: 1e-4
actor_lr: 1e-5
batch_size: 64
memory_capacity: 8000
target_update: 2
tau: 5e-3
critic_hidden_dim: 256
actor_hidden_dim: 256
seed: 2
```

**结果路径**: `tuning_results/high_gamma_large_batch/`
**训练图像**: `tuning_results/high_gamma_large_batch/final_metrics.png`

---


## low_lr_stable
**时间**: 2026-02-10 22:06:06
**描述**: 低学习率，追求稳定收敛
**状态**: ✓ 成功

### 超参数配置 (从 util.py 读取)
```
train_eps: 600
gamma: 0.5
critic_lr: 1e-4
actor_lr: 1e-5
batch_size: 64
memory_capacity: 8000
target_update: 2
tau: 5e-3
critic_hidden_dim: 256
actor_hidden_dim: 256
seed: 2
```

**结果路径**: `tuning_results/low_lr_stable/`
**训练图像**: `tuning_results/low_lr_stable/final_metrics.png`

---


## low_lr_stable
**时间**: 2026-02-10 22:07:06
**描述**: 低学习率，追求稳定收敛
**状态**: ✓ 成功

### 超参数配置 (从 util.py 读取)
```
train_eps: 600
gamma: 0.5
critic_lr: 1e-3
actor_lr: 1e-4
batch_size: 64
memory_capacity: 8000
target_update: 2
tau: 5e-3
critic_hidden_dim: 256
actor_hidden_dim: 256
seed: 2
```

**结果路径**: `tuning_results/low_lr_stable/`
**训练图像**: `tuning_results/low_lr_stable/final_metrics.png`

---


## high_lr_fast
**时间**: 2026-02-10 22:16:26
**描述**: 高学习率，加速学习
**状态**: ✓ 成功

### 超参数配置 (从 util.py 读取)
```
train_eps: 600
gamma: 0.5
critic_lr: 1e-3
actor_lr: 1e-4
batch_size: 64
memory_capacity: 8000
target_update: 2
tau: 5e-3
critic_hidden_dim: 256
actor_hidden_dim: 256
seed: 2
```

**结果路径**: `tuning_results/high_lr_fast/`
**训练图像**: `tuning_results/high_lr_fast/final_metrics.png`

---


## high_lr_fast
**时间**: 2026-02-10 22:17:28
**描述**: 高学习率，加速学习
**状态**: ✓ 成功

### 超参数配置 (从 util.py 读取)
```
train_eps: 600
gamma: 0.7
critic_lr: 5e-4
actor_lr: 5e-5
batch_size: 64
memory_capacity: 8000
target_update: 2
tau: 1e-2
critic_hidden_dim: 256
actor_hidden_dim: 256
seed: 2
```

**结果路径**: `tuning_results/high_lr_fast/`
**训练图像**: `tuning_results/high_lr_fast/final_metrics.png`

---


## fast_target_update
**时间**: 2026-02-10 22:26:52
**描述**: 快速目标网络更新+中等gamma
**状态**: ✓ 成功

### 超参数配置 (从 util.py 读取)
```
train_eps: 600
gamma: 0.7
critic_lr: 5e-4
actor_lr: 5e-5
batch_size: 64
memory_capacity: 8000
target_update: 2
tau: 1e-2
critic_hidden_dim: 256
actor_hidden_dim: 256
seed: 2
```

**结果路径**: `tuning_results/fast_target_update/`
**训练图像**: `tuning_results/fast_target_update/final_metrics.png`

---


## fast_target_update
**时间**: 2026-02-10 22:27:37
**描述**: 快速目标网络更新+中等gamma
**状态**: ✓ 成功

### 超参数配置 (从 util.py 读取)
```
train_eps: 600
gamma: 0.5
critic_lr: 5e-4
actor_lr: 5e-5
batch_size: 64
memory_capacity: 8000
target_update: 2
tau: 5e-3
critic_hidden_dim: 256
actor_hidden_dim: 256
seed: 2
```

**结果路径**: `tuning_results/fast_target_update/`
**训练图像**: `tuning_results/fast_target_update/final_metrics.png`

---

