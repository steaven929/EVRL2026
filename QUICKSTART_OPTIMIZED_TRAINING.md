# Quick Start Guide: Running Optimized IPPO Training

## Prerequisites
All P0 and P1 optimizations have been implemented and tested successfully.

## Run Training

### Short Test (10 episodes)
```bash
python main_marl.py --train_eps 10 --marl_mode multi
```

### Medium Test (50 episodes)
```bash
python main_marl.py --train_eps 50 --marl_mode multi
```

### Full Training (250 episodes with early stopping)
```bash
python main_marl.py --train_eps 250 --marl_mode multi
```

Note: Early stopping (patience=20) will automatically stop training if no improvement for 20 episodes.

## What to Monitor

### During Training
1. **Gradient Norms** (should be 0.1-10)
   - Printed in training output
   - If >10: Gradient explosion
   - If <0.01: Gradient vanishing

2. **Reward Monitoring** (every 10 steps)
   - Raw reward: Environment reward
   - Scaled reward: Should be in [-100, +100] range

3. **Early Stopping**
   - Watch for "New best model saved" messages
   - Training stops if no improvement for 20 episodes

4. **Learning Rate Decay**
   - Decays 10% every 50 episodes
   - Check optimizer.param_groups[0]['lr']

### After Training
Check these files in `outputs/marl_YYYYMMDD-HHMMSS/`:
- `marl_model_best.pth` - Best model (saved by early stopping)
- `marl_model_ep100.pth`, `marl_model_ep200.pth` - Periodic checkpoints
- `rewards.npy` - Episode rewards
- `ma_rewards.npy` - Moving average rewards
- `critic_losses.npy` - Critic loss history
- `actor_losses.npy` - Actor loss history

## Expected Results

### Baseline (marl_20260221-143622)
- Episode 99: 3978 (best)
- Episode 100-250: 2610 (degraded 34.4%)
- Actor Loss: -0.01 → -317 (monotonic decrease)
- Critic Loss std: 2.44 → 3.33 (+36%)

### After Optimization (Expected)
- Episode 99: 4500 (+13%)
- Episode 100-250: 4200 (degraded only 6.7%)
- Actor Loss: Stable at -50 ± 20 (variance ↓60%)
- Critic Loss std: Stable at 2.0 ± 0.5 (↓40%)

## Troubleshooting

### If Gradient Norms Explode (>10)
1. Check Q-value clipping is working
2. Reduce learning rates further
3. Consider adding gradient clipping (already at max_norm=1.0)

### If Rewards Unstable
1. Check reward_scale is 1000.0
2. Monitor raw vs scaled rewards
3. Consider P2-7: Reward Clipping

### If Early Stopping Too Early
1. Increase patience from 20 to 50
2. Check if best model is actually good
3. May need more exploration

### If Training Too Slow
1. Reduce train_eps
2. Use smaller batch_size (already 128)
3. Check if early stopping triggered

## Comparison with Baseline

### Run Analysis Script
```bash
python analyze_optimization_results.py
```

This will compare:
- Reward curves
- Loss curves
- Gradient norms
- Overfitting metrics

## Next Steps After Training

### If Results Good (10-20% improvement)
1. Document results in checkpoint/
2. Update MEMORY.md with actual results
3. Consider Phase-2: MAPPO with credit assignment

### If Results Insufficient
1. Analyze what didn't improve
2. Consider P2 optimizations:
   - P2-7: Reward Clipping
   - P2-8: Running Mean
   - P2-9: Polyak Averaging
3. May need reward function redesign

## Configuration Summary

Current hyperparameters (core/util.py):
```python
critic_lr = 1e-5          # Reduced from 3e-5
actor_lr = 5e-6           # Reduced from 1.5e-5
tau = 5e-5                # Reduced from 1e-4
reward_scale = 1000.0     # Reduced from 10000.0
batch_size = 128          # Reduced from 256
gamma = 0.95              # Unchanged
```

New optimizations:
```python
# P0-1: Target Actor
target_actor = SharedActor(...)

# P0-2: Q-value Clipping
target_q = torch.clamp(target_q, -1000, 1000)
current_q = torch.clamp(current_q, -1000, 1000)

# P1-4: Learning Rate Scheduler
actor_scheduler = StepLR(step_size=50, gamma=0.9)
critic_scheduler = StepLR(step_size=50, gamma=0.9)

# P1-5: Early Stopping
patience = 20

# P1-6: Gradient Monitoring
grad_info = {'critic_grad_norm': ..., 'actor_grad_norm': ...}
```

## Files to Check
- `core/marl_agent.py` - Agent implementation
- `core/marl_training.py` - Training loop
- `test_p0_optimizations.py` - Verification tests
- `checkpoint/ippo_optimization_implementation_20260222.md` - Full documentation
