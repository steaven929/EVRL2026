# Training Analysis Report (2026-02-18)

## 1. Current Training Status

### Training Configuration
- **Episodes**: 100 (not 1000 as intended)
- **Algorithm**: SAC with dual-head architecture
- **State Space**: 251 dimensions (SIP dataset, 50 zones)
- **Batch Size**: 1024
- **Learning Rates**: critic_lr=1.5e-4, actor_lr=4e-5
- **Reward Scale**: 8000.0

### Performance Metrics

#### Training Performance (100 episodes)
- **Mean Reward (last 50)**: 1,095,490
- **Best Reward**: 1,658,259
- **Service Rate (last 10)**: 0.16% - 0.51% ⚠️ CRITICAL ISSUE
- **Service Rate (mean last 50)**: 0.87%
- **Service Rate (best)**: 6.58%

#### Baseline Comparison (Test)
- **Random Policy**: 2,016,112 reward, 98.13 SVC
- **Greedy Policy**: 1,567,199 reward, 60.23 SVC
- **RL-DDPG**: 2,676,912 reward, 77.45 SVC
- **Global Optimal**: 2,273,319 reward

#### Loss Metrics (last 10 episodes)
- **Actor Loss**: 0.2753
- **Critic1 Loss**: 3.9056
- **Critic2 Loss**: 3.5830
- **Q Values**: 1.17 ⚠️ VERY LOW

## 2. Critical Issues Identified

### Issue 1: Extremely Low Service Rate During Training
**Symptom**: Service rate 0.16-0.51% during training vs 77.45% during test
**Severity**: CRITICAL
**Possible Causes**:
1. Training uses exploration noise that disrupts policy
2. Different environment configuration between train/test
3. Vehicle distribution feature causing state space explosion
4. Actions not being executed correctly during training

### Issue 2: Very Low Q Values
**Symptom**: Q values around 1.17
**Severity**: HIGH
**Possible Causes**:
1. Reward scale (8000.0) too high, causing gradient issues
2. Critic not learning value function properly
3. State space (251 dims) too large for current network capacity

### Issue 3: High Variance in Test Performance
**Symptom**: RL test SVC std=66.09 (mean=77.45)
**Severity**: MEDIUM
**Interpretation**: Policy is unstable, sometimes works well, sometimes fails

### Issue 4: Training Stopped at 100 Episodes
**Symptom**: Only 100 episodes trained instead of intended longer training
**Severity**: MEDIUM
**Impact**: Insufficient training time for convergence

## 3. Root Cause Analysis

### Primary Hypothesis: State Space Dimensionality Issue
The vehicle distribution feature added 50 dimensions (201→251), increasing state space by 25%. This may cause:
- Embedding layer struggles with high-dimensional discrete states
- Network capacity (512 hidden, 32 embed) insufficient
- Slower learning requiring more episodes

### Secondary Hypothesis: Reward Scale Mismatch
Reward scale of 8000.0 may be too high for SAC:
- Causes Q values to be compressed (observed: 1.17)
- Makes critic learning unstable
- Actor receives weak gradients

### Tertiary Hypothesis: Exploration-Exploitation Imbalance
SAC alpha=0.25 may cause too much exploration:
- During training, random actions disrupt service
- During test (deterministic), policy works better
- Explains train SVC (0.5%) vs test SVC (77.45%) gap

## 4. Optimization Recommendations

### Priority 1: Fix State Space Issues (IMMEDIATE)

#### Option A: Remove Vehicle Distribution Feature (Recommended)
**Rationale**: The 25% increase in state dimensions may be causing learning instability
**Action**:
1. Revert state space from 251 to 201 dimensions
2. Remove vehicle_counts_per_zone from state construction
3. Network can learn vehicle distribution implicitly from OD demand patterns

**Expected Impact**:
- Faster learning convergence
- More stable Q values
- Better service rate during training

#### Option B: Increase Network Capacity
**Rationale**: Keep 251-dim state but increase network size
**Action**:
1. critic_hidden_dim: 512 → 1024
2. actor_hidden_dim: 512 → 1024
3. critic_embed_dim: 32 → 64
4. actor_embed_dim: 32 → 64

**Expected Impact**:
- Better representation of high-dimensional state
- Slower training (more parameters)
- Higher memory usage

### Priority 2: Adjust Reward Scale (HIGH)

#### Recommendation: Reduce Reward Scale
**Current**: 8000.0
**Proposed**: 2000.0 - 4000.0
**Rationale**:
- Current Q values (1.17) suggest over-compression
- Lower scale allows critic to learn finer value distinctions
- SAC works better with moderate reward scales

**Action**:
```python
# core/util.py line 48
parser.add_argument('--reward_scale',default=3000.0,type=float)
```

### Priority 3: Reduce Exploration During Training (MEDIUM)

#### Recommendation: Lower SAC Alpha
**Current**: 0.25
**Proposed**: 0.10 - 0.15
**Rationale**:
- High alpha causes excessive exploration
- Explains low service rate during training
- Test performance (deterministic) is much better

**Action**:
```python
# core/util.py line 43
parser.add_argument('--sac_alpha',default=0.12,type=float)
```

### Priority 4: Extend Training Duration (MEDIUM)

#### Recommendation: Train for 500-1000 Episodes
**Current**: 100 episodes
**Proposed**: 500-1000 episodes
**Rationale**:
- 251-dim state space needs more samples
- Current training shows no convergence yet
- Baseline comparison shows RL can reach 117% of optimal (suspicious)

**Action**:
```python
# When running training
python main.py --train_eps 1000
```

### Priority 5: Investigate Train/Test Discrepancy (HIGH)

#### Recommendation: Add Logging to Understand Gap
**Issue**: Train SVC (0.5%) vs Test SVC (77.45%) - 150x difference
**Action**:
1. Log action statistics during training vs testing
2. Check if exploration noise is being applied correctly
3. Verify environment reset between train/test

## 5. Recommended Experiment Plan

### Experiment A: Simplified State Space (RECOMMENDED FIRST)
**Changes**:
1. Remove vehicle distribution (251→201 dims)
2. Reduce reward_scale to 3000.0
3. Reduce sac_alpha to 0.12
4. Train for 500 episodes

**Expected Outcome**:
- Service rate during training: 5-15%
- Q values: 5-20
- Convergence by episode 300-400

### Experiment B: Increased Capacity (If A fails)
**Changes**:
1. Keep 251-dim state
2. Double network size (512→1024 hidden, 32→64 embed)
3. Reduce reward_scale to 2000.0
4. Train for 1000 episodes

**Expected Outcome**:
- Slower but more stable learning
- Service rate: 10-20%
- Higher memory usage

### Experiment C: Hybrid Approach
**Changes**:
1. Keep vehicle distribution but use continuous encoding (not discrete)
2. Moderate network increase (512→768 hidden)
3. Reward scale: 3000.0
4. SAC alpha: 0.12
5. Train for 800 episodes

## 6. Immediate Action Items

1. ✅ **Analyze current results** (DONE)
2. ⚠️ **Choose experiment path** (A, B, or C)
3. ⚠️ **Modify hyperparameters** in core/util.py
4. ⚠️ **Optionally modify state space** in specific/dclily_elec.py
5. ⚠️ **Run training** with proper episode count
6. ⚠️ **Monitor metrics** during training (SVC, Q values, losses)
7. ⚠️ **Compare with baseline** after training completes

## 7. Success Criteria

### Minimum Acceptable Performance
- Service rate during training: >5%
- Service rate during test: >60%
- RL reward: >80% of optimal (>1,818,655)
- Q values: >5.0
- Training convergence: <500 episodes

### Target Performance
- Service rate during training: >10%
- Service rate during test: >75%
- RL reward: >90% of optimal (>2,045,987)
- Q values: 10-30
- Stable policy (test std <20)

## 8. Key Insights

1. **State space dimensionality matters**: 25% increase (201→251) may have broken learning
2. **Reward scale is critical**: Current 8000.0 compresses Q values too much
3. **Exploration hurts service rate**: SAC alpha=0.25 causes poor training performance
4. **Test performance is misleading**: 77.45% SVC with 66.09 std means unstable policy
5. **Need more training**: 100 episodes insufficient for 251-dim state space

## 9. Next Steps

**Recommended**: Start with Experiment A (simplified state space)
- Fastest path to working solution
- Reduces complexity
- Can always add vehicle distribution back later if needed

**Alternative**: If vehicle distribution is critical, try Experiment B (increased capacity)
- Keeps all information
- Requires more compute and time
- Higher risk of overfitting
