# Training Analysis Report - 500 Episodes (2026-02-18)
# After Scale-Adaptive Reward Implementation

## 1. Critical Problem: Negative Learning

### Performance Degradation

| Metric | First 10 Eps | Last 10 Eps | Change | Status |
|--------|--------------|-------------|--------|--------|
| **Reward** | -42,316 | -44,781 | -5.8% | 🔴 WORSE |
| **Service Rate** | 2.69% | 0.55% | -79.6% | 🔴 COLLAPSE |
| **Best SVC** | 7.38% | - | - | - |

### Key Observations

1. **Policy is degrading, not improving**
   - Rewards becoming more negative
   - Service rate collapsed from 2.69% to 0.55%
   - No convergence after 500 episodes

2. **Charging behavior is rational**
   - Only 3% of vehicles sent to charge (3/92 in last 100 actions)
   - Only charging vehicles with SOC < 2000 (e.g., SOC=1998)
   - This confirms relative reward fix is working

3. **Q values improving but policy failing**
   - Q values: -15.76 → -4.73 (improving)
   - But actual performance degrading
   - Suggests value estimation is disconnected from policy

## 2. Root Cause Analysis

### Problem 1: Reward Scale Mismatch 🔴

**Current reward**: -44,781 (very negative)
**Reward scale**: 2500.0

**Issue**: Rewards are extremely negative, causing:
- Gradient instability
- Q value compression
- Actor learning difficulty

**Evidence**:
```
Episode reward: -44,781
After scaling (/2500): -17.9
Q values: -4.73
```

The reward scale of 2500 is still too high for this problem.

### Problem 2: Service Revenue Too Weak 🔴

**Current reward function**:
```python
reward = (service_revenue * 2.0 - soc_penalty) +
         charging_bonus + charging_maintenance_bonus +
         soc_maintain_bonus + service_rate_threshold_bonus -
         fixed_penalty - service_rate_penalty
```

**Problem**: Service revenue is multiplied by 2.0, but:
- Service revenue is already divided by some factor in Gurobi
- SOC penalties are very large (thousands)
- Service revenue gets overwhelmed

**Evidence**: Service rate collapsed to 0.55%, suggesting agent learned to avoid service.

### Problem 3: SOC Penalties Too Aggressive 🔴

**Current SOC penalty logic** (lines 1548-1587):
```python
if epoch < 300:
    penalty_scale = 1.5
    vehicle_penalty = 750
elif epoch < 600:
    penalty_scale = 1.5 - 0.3 * progress
    vehicle_penalty = 750 - 150 * progress
```

**Problem**:
- Penalties are extremely high (750-600 per low-SOC vehicle)
- With 200 vehicles, total penalty can be 150,000+
- This dominates the reward function
- Agent learns to avoid any action that might lower SOC

### Problem 4: Exploration Collapse 🟡

**SAC alpha**: 0.10 (low exploration)

**Issue**: With such negative rewards and high penalties:
- Agent quickly converges to "do nothing" policy
- Low exploration prevents discovering better strategies
- Service rate collapses because serving customers risks SOC penalties

## 3. Recommended Fixes

### Fix 1: Normalize Rewards (CRITICAL)

**Problem**: Rewards are in range [-50000, 0], causing instability

**Solution**: Normalize rewards to [-1, 1] range

```python
# =========2026.2.18修改，奖励归一化==========
# 在dclily_elec.py的step函数中，返回reward前：

# 估计的奖励范围（基于历史数据）
REWARD_MIN = -50000  # 最差情况
REWARD_MAX = 0       # 最好情况（无成本）

# 归一化到[-1, 1]
reward_normalized = 2 * (reward_value - REWARD_MIN) / (REWARD_MAX - REWARD_MIN) - 1

return state, reward_normalized, self.terminal, info
# ======修改结束
```

**Expected impact**:
- Stable gradients
- Better Q value learning
- Reward scale parameter becomes less critical

### Fix 2: Rebalance Reward Components (HIGH PRIORITY)

**Current imbalance**:
- Service revenue: ~1000-5000
- SOC penalties: ~10000-50000 (10-50x larger!)
- Charging rewards: ~500-1500

**Solution**: Reduce SOC penalty scale

```python
# =========2026.2.18修改，降低SOC惩罚强度==========
# 在dclily_elec.py lines 1548-1587

if epoch < 300:
    penalty_scale = 0.3  # 1.5 → 0.3 (80% reduction)
    vehicle_penalty = 150  # 750 → 150 (80% reduction)
elif epoch < 600:
    progress = (epoch - 300) / 300
    penalty_scale = 0.3 - 0.05 * progress  # 0.3 → 0.25
    vehicle_penalty = 150 - 30 * progress  # 150 → 120
else:
    penalty_scale = 0.25  # 1.2 → 0.25
    vehicle_penalty = 120  # 600 → 120
# ======修改结束
```

**Expected impact**:
- SOC penalties: 10000-50000 → 2000-10000 (80% reduction)
- Service revenue becomes competitive
- Agent willing to serve customers

### Fix 3: Increase Service Revenue Weight (HIGH PRIORITY)

**Current**: `service_revenue * 2.0`

**Solution**: Increase to 5.0-10.0

```python
# =========2026.2.18修改，大幅提升服务收益权重==========
# 在dclily_elec.py line 1639

reward_value = (service_revenue * 10.0 - soc_penalty) + ...
# ======修改结束
```

**Expected impact**:
- Service revenue: 1000-5000 → 10000-50000
- Competitive with SOC penalties
- Agent incentivized to serve customers

### Fix 4: Reduce Reward Scale (MEDIUM)

**Current**: 2500.0
**Recommended**: 1000.0

```python
# core/util.py line 46
parser.add_argument('--reward_scale',default=1000.0,type=float)
```

**Rationale**: With reward normalization, lower scale is sufficient

### Fix 5: Increase Exploration (MEDIUM)

**Current**: sac_alpha = 0.10
**Recommended**: sac_alpha = 0.20

```python
# core/util.py line 42
parser.add_argument('--sac_alpha',default=0.20,type=float)
```

**Rationale**: Need more exploration to escape "do nothing" local optimum

## 4. Implementation Priority

### Phase 1: Immediate (Critical Fixes)

1. **Fix 2**: Reduce SOC penalty scale (80% reduction)
2. **Fix 3**: Increase service revenue weight (2.0 → 10.0)

**Expected result**: Service rate 0.55% → 10-20%

### Phase 2: After Phase 1 Validation (100 episodes)

3. **Fix 1**: Normalize rewards to [-1, 1]
4. **Fix 4**: Reduce reward_scale to 1000.0
5. **Fix 5**: Increase sac_alpha to 0.20

**Expected result**: Service rate 10-20% → 30-50%

### Phase 3: Full Training (500 episodes)

**Expected result**:
- Service rate: 40-60%
- Reward: Approach optimal (within 80-90%)
- Stable, rational policy

## 5. Diagnostic Checklist

Before implementing fixes, verify:

✅ **Relative reward working**: Only 3% charging (correct)
✅ **Charging rational**: Only charge SOC < 2000 (correct)
❌ **Service rate**: 0.55% (too low)
❌ **Learning trend**: Negative (getting worse)
❌ **Reward balance**: SOC penalties dominate (imbalanced)

## 6. Expected Outcomes

### After Phase 1 (100 episodes)

| Metric | Current | Expected | Improvement |
|--------|---------|----------|-------------|
| Service Rate | 0.55% | 10-20% | 18-36x |
| Reward | -44,781 | -30,000 | 33% better |
| SOC Penalty | ~40,000 | ~8,000 | 80% reduction |

### After Phase 3 (500 episodes)

| Metric | Current | Expected | Improvement |
|--------|---------|----------|-------------|
| Service Rate | 0.55% | 40-60% | 73-109x |
| Reward | -44,781 | -15,000 | 66% better |
| vs Optimal | - | 80-90% | - |

## 7. Key Insights

1. **Scale-adaptive reward worked**: Charging behavior is now rational
2. **But reward imbalance remains**: SOC penalties too strong
3. **Agent learned wrong lesson**: "Avoid all actions" instead of "serve customers"
4. **Need aggressive rebalancing**: 80% reduction in penalties + 5x increase in service revenue
5. **Exploration needed**: Current policy stuck in local optimum

## 8. Next Steps

**Recommended**: Implement Phase 1 fixes immediately

1. Reduce SOC penalty scale: 1.5 → 0.3
2. Increase service revenue weight: 2.0 → 10.0
3. Run 100 episodes to validate
4. If successful, proceed to Phase 2

**Alternative**: If Phase 1 insufficient, also implement:
- Reward normalization (Fix 1)
- Increased exploration (Fix 5)
