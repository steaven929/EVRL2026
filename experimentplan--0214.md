# RL Order Action Optimization Experiment Plan (0214)

## Goal
Validate whether current RL order actions align with OD demand and dispatch logic, and identify the smallest safe change set that improves late-stage service performance.

## Current Findings
- RL action values are highly concentrated (about 0.38-0.42), indicating weak policy discrimination across timesteps and zones.
- Historical comparison shows late timesteps with non-zero OD demand but zero RL service in multiple steps.
- Dispatch pipeline logic is structurally correct (RL dispatch solve, then OD-demand service solve), so the likely bottleneck is upstream action quality + decode sensitivity.

## Hypothesis
1. Action collapse + conservative decode causes under-dispatch in demand-active timesteps.
2. Improving decode sensitivity and no-order gating can recover service without major architecture changes.
3. Demand-aware exploration can further improve action diversity and temporal adaptation.

## Experiment Design

### Fixed Setup
- Dataset: `data/toy_network/od_demand_test.csv`
- Seeds: `2, 7, 13`
- Stage-1 train: `300 eps`
- Stage-2 verification (best config only): `700 eps`
- Test episodes per config: `20`

### Metrics (must report)
- `SvcLate`: mean serviced orders for timesteps `7..11`
- `ZeroSvcLateSteps`: count of timesteps in `7..11` with zero service
- `ActionStdAll`, `ActionStdDep`, `ActionStdDest`, `ActionStdCharge`
- `DispatchEfficiency = served_orders / dispatched_orders`
- `RewardMean`, `RewardStd`, `AvgSOC`

### Experiment Matrix
1. **Baseline**: current implementation
2. **A1 (Decode-Sqrt)**: decode `limit_now` with non-linear scaling (sqrt-style)
3. **A2 (Decode+MinDispatch)**: A1 + minimum dispatch guard when demand exists
4. **B1 (NoOrderGate)**: A2 + no-order timestep dispatch suppression
5. **B2 (NoOrderLookahead)**: A2 + lookahead preposition for `t+1/t+2` demand
6. **C1 (DemandAwareExploration)**: B2 + demand-aware action noise schedule

## Success Criteria (Go/No-Go)
- `ZeroSvcLateSteps` reduced by at least 50% vs Baseline
- `SvcLate` improved by at least 30% vs Baseline
- `RewardMean` does not drop by more than 5%

If all three pass, proceed to 700-episode verification; otherwise keep only the highest-impact single intervention.

## Execution Sequence
1. Run Baseline and collect all metrics.
2. Run A1 and A2 (decode-only track) and compare against Baseline.
3. Run B1 and B2 (gating/lookahead track).
4. Run C1 (exploration enhancement) only if B2 improves service stability.
5. Select winner by success criteria.
6. Re-run winner at 700 eps for confirmation.

## Risks and Controls
- Risk: improved late service but unstable SOC.
  - Control: enforce SOC trend checks and report `AvgSOC` by timestep.
- Risk: reward rises due to charging bias while service quality drops.
  - Control: require service-side criteria (`SvcLate`, `ZeroSvcLateSteps`) as hard gates.
- Risk: seed sensitivity.
  - Control: require results to hold across all 3 seeds.

## Deliverables
- `experiment_results.csv` (one row per seed x config)
- `experiment_summary.md` (ranking, failure analysis, recommended config)
- Optional plots: action std trend, late-step service trend, SOC trend

## Minimal Command Template
Use the same command format for each config/seed with only config flags changed.

```bash
python main.py --train_eps 300 --test_eps 20 --seed <SEED>
```

For winner confirmation:

```bash
python main.py --train_eps 700 --test_eps 20 --seed <SEED>
```
