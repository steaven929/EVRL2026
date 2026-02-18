# Training Analysis Report - 800 Episodes (2026-02-18)

## 1. Training Results Summary

### Configuration
- **Episodes**: 800 (completed)
- **State Space**: 251 dimensions
- **Network**: hidden_dim=768, embed_dim=48
- **Learning Rates**: critic_lr=1e-4, actor_lr=3e-5
- **Reward Scale**: 2500.0
- **SAC Alpha**: 0.10
- **Batch Size**: 1024

### Performance Metrics

| Metric | First 10 Eps | Last 10 Eps | Best | Trend |
|--------|--------------|-------------|------|-------|
| **Reward** | 1,307,992 | 1,156,881 | 1,736,148 | ⬇️ -11.5% |
| **Service Rate** | 1.65% | 0.63% | 6.85% | ⬇️ -62% |
| **Q Values** | 1.07 | 44.81 | - | ⬆️ +4100% |
| **Actor Loss** | - | -39.46 | - | - |
| **Critic Loss** | - | 8.59 | - | - |

### Trend Analysis (50-episode moving average)

| Episode | Reward | Service Rate |
|---------|--------|--------------|
| 50 | 1,246,175 | 1.65% |
| 400 | 1,155,210 | 1.43% |
| 750 | 1,199,471 | 1.26% |

## 2. Critical Problems Identified

### 🔴 Problem 1: Pathological Charging Behavior

**Observation from vehicle_action_log.csv**:
- **ALL vehicles have SOC=2000 (100% battery)**
- **ALL vehicles are being sent to charge (ChargingDecided=True)**
- This pattern persists throughout training (episode 0 to episode 24+)

**Example from log**:
```
24,100,48,2000.00,0.582,136.000,0.694,True,In_Zone_48_ChargingDecided
24,101,48,2000.00,0.582,136.000,0.694,True,In_Zone_48_ChargingDecided
24,102,48,2000.00,0.582,136.000,0.694,True,In_Zone_48_ChargingDecided
```

**Impact**:
- Vehicles at full battery are sent to chargers
- No vehicles available for customer service
- Service rate collapses to 0.63%
- This is completely irrational behavior

**Root Cause**:
The RL agent has learned a **degenerate policy**:
1. Send all idle vehicles to charge, regardless of SOC
2. This maximizes charging rewards
3. But completely ignores service revenue
4. The reward function is misaligned

### 🔴 Problem 2: Negative Learning Trend

**Observation**:
- Rewards DECREASE over training: 1,307,992 → 1,156,881 (-11.5%)
- Service rate DECREASES: 1.65% → 0.63% (-62%)
- The policy is getting WORSE, not better

**Interpretation**:
- The agent is not learning the correct objective
- It's optimizing for a local optimum (charging rewards)
- Ignoring the global optimum (service + charging balance)

### 🔴 Problem 3: Reward Function Misalignment

**Current Reward Structure** (from dclily_elec.py):
```python
reward = (service_revenue - soc_penalty) / 10 + charging_bonus +
         charging_maintenance_bonus + soc_maintain_bonus +
         distribution_balance_bonus - fixed_penalty
```

**Problem**:
- Charging bonuses are too attractive
- Service revenue is divided by 10 (weakened)
- Agent learns: "charge everything" > "serve customers"

**Evidence**:
- Line 879-884: charge_budget calculation
- Line 893-900: Forced charging when avg_soc < 1000
- But vehicles already at SOC=2000 are still being charged!

### 🟡 Problem 4: Forced Charging Logic Bug

**Code Analysis** (lines 893-900):
```python
if avg_soc < 1000:
    min_charge_vehicles = max(3, int(total_idle * 0.3))
    charge_budget = max(charge_budget, min_charge_vehicles)
```

**Issue**:
- This forces charging when avg_soc < 1000 (50%)
- But it doesn't check if vehicles NEED charging
- Vehicles at SOC=2000 are still selected for charging
- No SOC threshold check before charging decision

### 🟢 Success: Q Value Recovery

**Observation**:
- Q values increased from 1.07 to 44.81 (+4100%)
- This confirms reward_scale reduction (8000→2500) worked
- Critic is now learning proper value estimates

## 3. Root Cause Analysis

### Primary Issue: Missing SOC Threshold in Charging Logic

**Location**: `check_idle_vehicles()` function (lines 912-944)

**Current Logic**:
1. Sort all idle vehicles by SOC (line 913)
2. Select lowest SOC vehicles up to charge_budget (line 916-918)
3. Send them to chargers

**Missing Check**:
- No verification that vehicle SOC < threshold before charging
- Vehicles at SOC=2000 (full) are still sent to charge
- This wastes fleet capacity

**Expected Logic**:
```python
# Should be:
if _soc < BATTERY_CAPACITY * 0.8:  # Only charge if < 80%
    # Then proceed with charging
```

### Secondary Issue: Reward Function Imbalance

**Charging Rewards Too High**:
- Charging bonus: +1500 per charge
- Service revenue: divided by 10
- Result: Agent prefers charging over service

**Solution**:
- Reduce charging bonus: 1500 → 500
- Or increase service revenue weight: /10 → /5
- Or add penalty for charging full vehicles

## 4. Recommended Fixes

### Fix 1: Add SOC Threshold Check (CRITICAL)

**File**: `specific/dclily_elec.py`
**Location**: Line 916 (in `check_idle_vehicles`)

**Current**:
```python
for vehicle_id, current_zone, _soc in idle_vehicles_all:
    if charged_count >= charge_budget:
        break
    # ... proceed with charging
```

**Fix**:
```python
# =========2026.2.18修改，修复：只充电低电量车辆==========
CHARGE_THRESHOLD = BATTERY_CAPACITY * 0.75  # 只在SOC < 75%时充电

for vehicle_id, current_zone, _soc in idle_vehicles_all:
    if charged_count >= charge_budget:
        break

    # 关键修复：跳过高电量车辆
    if _soc >= CHARGE_THRESHOLD:
        continue

    # ... proceed with charging
# ======修改结束
```

**Expected Impact**:
- Vehicles at SOC > 75% won't be sent to charge
- More vehicles available for service
- Service rate: 0.63% → 15-30%

### Fix 2: Rebalance Reward Function (HIGH PRIORITY)

**File**: `specific/dclily_elec.py`
**Location**: Reward calculation (around line 1225)

**Option A: Reduce Charging Bonus**
```python
# Current
charging_bonus = 1500 * num_charged

# Fix
charging_bonus = 300 * num_charged  # 1500 → 300 (80% reduction)
```

**Option B: Increase Service Revenue Weight**
```python
# Current
reward = (service_revenue - soc_penalty) / 10 + ...

# Fix
reward = (service_revenue - soc_penalty) / 3 + ...  # /10 → /3
```

**Option C: Add Penalty for Unnecessary Charging**
```python
# New penalty
unnecessary_charge_penalty = 0
for vehicle_id in charged_vehicles:
    if self.car[vehicle_id][1] > BATTERY_CAPACITY * 0.8:
        unnecessary_charge_penalty += 500  # Penalty per full vehicle charged

reward = ... - unnecessary_charge_penalty
```

### Fix 3: Remove Forced Charging for High SOC (MEDIUM)

**File**: `specific/dclily_elec.py`
**Location**: Lines 893-900

**Current**:
```python
if avg_soc < 1000:
    min_charge_vehicles = max(3, int(total_idle * 0.3))
    charge_budget = max(charge_budget, min_charge_vehicles)
```

**Fix**:
```python
# =========2026.2.18修改，修复：只在真正需要时强制充电==========
if avg_soc < 800:  # 提高阈值：1000 → 800 (40%)
    # 只强制充电真正低电量的车辆
    low_soc_vehicles = [v for v in idle_vehicles_all if v[2] < 1200]
    if low_soc_vehicles:
        min_charge_vehicles = max(2, int(len(low_soc_vehicles) * 0.5))
        charge_budget = max(charge_budget, min_charge_vehicles)
# ======修改结束
```

## 5. Experiment Plan

### Experiment 1: SOC Threshold Only (Quick Fix)
**Changes**:
- Add SOC < 75% check before charging (Fix 1)
- Keep all other parameters same

**Expected**:
- Service rate: 0.63% → 10-20%
- Reward: 1,156,881 → 1,500,000+
- Training: 200 episodes to verify

### Experiment 2: Full Fix (Recommended)
**Changes**:
- Add SOC threshold check (Fix 1)
- Reduce charging bonus: 1500 → 500 (Fix 2A)
- Adjust forced charging logic (Fix 3)

**Expected**:
- Service rate: 0.63% → 20-40%
- Reward: 1,156,881 → 1,800,000+
- Training: 500 episodes

### Experiment 3: Reward Rebalance (Alternative)
**Changes**:
- Add SOC threshold check (Fix 1)
- Increase service weight: /10 → /5 (Fix 2B)
- Add unnecessary charge penalty (Fix 2C)

**Expected**:
- Service rate: 0.63% → 25-45%
- Reward: 1,156,881 → 2,000,000+
- Training: 500 episodes

## 6. Implementation Priority

### Immediate (Do Now):
1. ✅ **Fix 1**: Add SOC threshold check
   - This is a bug fix, not a hyperparameter tuning
   - Will immediately improve service rate
   - Low risk, high impact

### High Priority (Next):
2. ⚠️ **Fix 2A**: Reduce charging bonus
   - Rebalance reward function
   - Align agent incentives with true objective

### Medium Priority (After validation):
3. ⚠️ **Fix 3**: Adjust forced charging logic
   - Only if Fix 1 + Fix 2 are insufficient

## 7. Success Criteria

### Minimum Acceptable:
- Service rate during training: >10%
- Service rate during test: >50%
- Reward: >1,500,000
- No vehicles at full battery sent to charge

### Target Performance:
- Service rate during training: >20%
- Service rate during test: >70%
- Reward: >1,800,000 (80% of optimal 2,273,319)
- Rational charging behavior (only charge when SOC < 75%)

## 8. Key Insights

1. **Hyperparameter tuning alone is insufficient** - The core issue is a logic bug
2. **Reward function must align with true objective** - Charging rewards too high
3. **Always validate learned behavior** - Metrics can be misleading
4. **Q value recovery was successful** - Network capacity increase worked
5. **Need SOC-aware charging logic** - Current logic is SOC-blind

## 9. Next Steps

**Recommended**: Implement Fix 1 immediately
- This is a critical bug fix
- Will unblock learning
- Can be done in <10 minutes

**Then**: Run 200-episode test to validate
- Monitor service rate improvement
- Check vehicle action log for rational behavior
- If successful, proceed with Fix 2

**Finally**: Full 500-episode training with all fixes
- Should reach 80%+ of optimal performance
- Stable, rational policy
