# MARL Reward Structure Analysis: Global vs Individual Rewards

## Current Implementation (Phase-1: Global Reward Sharing)

### Architecture
```python
# In marl_agent.py line 152
reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)  # (batch, 1)
```

**Key Characteristics:**
- **Single scalar reward** for all 50 agents
- All agents receive the same reward value
- Reward shape: (batch_size, 1) - one value per transition

### How It Works

**Training Loop (marl_training.py):**
```python
# Step environment with joint action (50 agents × 3 actions)
next_global_state, reward, done, info = env.step(actions)

# Store single reward for all agents
agent.memory.push(
    obs_n=obs_n,           # (50, 7) - each agent's observation
    act_n=actions,         # (50, 3) - each agent's action
    global_state=global_state,  # (251,) - global state
    reward=scaled_reward,  # scalar - SAME for all agents
    ...
)
```

**Update Process:**
```python
# Critic learns: Q(global_state, joint_action) = global_reward
target_q = reward_batch + gamma * Q_target(next_state, next_action)
# reward_batch shape: (batch, 1) - single value
```

### Pros of Global Reward

1. **Simple Implementation**
   - Easy to understand and debug
   - Minimal code changes from single-agent

2. **Stable Training**
   - All agents learn from same signal
   - No variance from individual reward estimation

3. **Team Coordination**
   - Agents naturally learn to cooperate
   - Optimize for global objective

4. **Matches Problem Structure**
   - Fleet management is inherently cooperative
   - Total profit/service rate is what matters

### Cons of Global Reward

1. **No Credit Assignment**
   - Agent in Zone 1 can't tell if its action was good/bad
   - All agents get same reward even if only Zone 5 performed well
   - Slows learning: hard to distinguish good from bad actions

2. **Scalability Issues**
   - With 50 agents, signal is very noisy
   - Each agent's contribution is 1/50 of total
   - Harder to learn as number of agents increases

3. **Free-Rider Problem**
   - Agents might learn to "coast" on others' performance
   - No individual accountability

4. **Exploration Inefficiency**
   - Can't tell which agent's exploration was useful
   - Wastes exploration effort

---

## Alternative: Individual Agent Rewards (Phase-2)

### Architecture Change

**Reward Structure:**
```python
# Instead of: reward (scalar)
# Use: reward_n (50,) - one reward per agent
```

### How It Would Work

**Reward Decomposition:**
```python
# For each agent i (zone i):
reward_i = (
    service_revenue_zone_i          # Orders served in zone i
    - soc_penalty_zone_i            # SOC penalty for vehicles in zone i
    + charging_bonus_zone_i         # Charging in zone i
    - service_rate_penalty_zone_i   # Service rate in zone i
    + coordination_bonus            # Shared component (optional)
)
```

**Example Calculation:**
```python
# Zone 0: Served 5 orders, 2 vehicles charging, avg SOC 1500
reward_0 = (5*100 - 0) / 10 + 2*18000 + 600 - 0 = 50 + 36000 + 600 = 36,650

# Zone 1: Served 0 orders, 0 vehicles charging, avg SOC 800
reward_1 = (0 - 5000) / 10 + 0 + 0 - 4000 = -500 + 0 + 0 - 4000 = -4,500

# Zone 2: Served 10 orders, 5 vehicles charging, avg SOC 1800
reward_2 = (10*100 - 0) / 10 + 5*18000 + 1200 - 0 = 100 + 90000 + 1200 = 91,300
```

### Implementation Changes Required

#### 1. Environment (dclily_elec.py)
```python
def step(self, action):
    # Current: return state, reward (scalar), done, info
    # New: return state, reward_n (50,), done, info

    # Calculate per-zone rewards
    reward_n = np.zeros(self.n_agents)
    for zone_id in range(self.n_agents):
        reward_n[zone_id] = self._calculate_zone_reward(zone_id)

    return state, reward_n, done, info
```

#### 2. Buffer (buffer.py)
```python
class MARLReplayBuffer:
    def push(self, obs_n, act_n, global_state, reward_n, ...):
        # reward_n shape: (n_agents,) instead of scalar
        transition = (obs_n, act_n, global_state, reward_n, ...)
```

#### 3. Agent (marl_agent.py)
```python
def update(self):
    # reward_batch shape: (batch, n_agents) instead of (batch, 1)
    reward_batch = torch.FloatTensor(np.array(reward_batch)).to(self.device)

    # Option A: Sum rewards for global Q-value
    total_reward = reward_batch.sum(dim=1, keepdim=True)  # (batch, 1)

    # Option B: Use individual rewards with value decomposition (QMIX/VDN)
    # More complex, requires architecture changes
```

### Pros of Individual Rewards

1. **Better Credit Assignment**
   - Each agent knows its contribution
   - Faster learning: clear signal for good/bad actions

2. **Improved Exploration**
   - Can identify which agent's exploration was useful
   - More efficient exploration

3. **Scalability**
   - Learning speed doesn't degrade with more agents
   - Each agent has clear signal

4. **Debugging**
   - Can identify which zones are performing poorly
   - Easier to diagnose problems

### Cons of Individual Rewards

1. **More Complex**
   - Harder to implement correctly
   - More code to maintain

2. **Reward Design Challenge**
   - How to split global reward fairly?
   - What about cross-zone effects? (vehicle moves from Zone A to Zone B)

3. **Potential Conflicts**
   - Agents might optimize local reward at expense of global
   - Need careful reward design to align incentives

4. **Training Instability**
   - More variance in individual rewards
   - Might need more sophisticated algorithms (QMIX, COMA, etc.)

---

## Recommendation for Your Problem

### Current Status (Phase-1)
- Using global reward sharing
- 50 agents, all get same reward
- Simple but slow learning

### Should You Switch to Individual Rewards?

**Short Answer: Not Yet**

**Reasoning:**

1. **Current Problem is Reward Function, Not Architecture**
   - Reward is negative because shaping mode was disabled
   - After enabling shaping, global reward should work

2. **Global Reward is Appropriate for Fleet Management**
   - Fleet profit is inherently global
   - Vehicles move between zones (hard to attribute)
   - Coordination is more important than individual performance

3. **Phase-1 Should Work First**
   - Get global reward working properly first
   - If still not learning after 200-300 episodes, then consider Phase-2

4. **Implementation Complexity**
   - Individual rewards require significant code changes
   - Risk of introducing bugs
   - Need to carefully design reward decomposition

### When to Consider Individual Rewards (Phase-2)

**Indicators that you need Phase-2:**

1. **Learning is too slow** (>500 episodes, still no progress)
2. **Some zones learn, others don't** (uneven performance)
3. **Agents interfere with each other** (negative coordination)
4. **Exploration is inefficient** (random behavior persists)

**If you see these after fixing current issues, then implement:**
- Value Decomposition (VDN): Q_total = Σ Q_i
- QMIX: Q_total = mixing_network(Q_1, ..., Q_50)
- Or simple reward decomposition with coordination bonus

---

## Immediate Action Plan

### Step 1: Test with Global Reward + Shaping Mode (Current)
- Train 100-200 episodes
- Check if reward becomes positive
- Check if agents learn charging

### Step 2: If Learning is Successful
- Continue with global reward
- No need for individual rewards

### Step 3: If Learning is Still Slow
- Analyze which zones are performing poorly
- Consider implementing individual rewards
- Start with simple decomposition: reward_i = local_reward + 0.1 * global_reward

---

## Summary

**Current Implementation:**
- Global reward sharing (Phase-1)
- All 50 agents get same reward
- Simple, stable, appropriate for cooperative task

**Individual Rewards (Phase-2):**
- Each agent gets zone-specific reward
- Better credit assignment, faster learning
- More complex, requires careful design

**Recommendation:**
- **Keep global reward for now**
- Fix current issue (enable shaping mode)
- Only switch to individual rewards if learning is still too slow after 200-300 episodes

**Key Insight:**
Your current problem is reward function balance (negative rewards), not reward structure (global vs individual). Fix the balance first, then evaluate if you need better credit assignment.
