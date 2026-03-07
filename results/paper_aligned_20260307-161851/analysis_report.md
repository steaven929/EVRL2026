# Paper Figure Analysis (Aligned Episodes)

- Output folder: `F:\Claude code\EVRL260207\results\paper_aligned_20260307-161851`
- Episode alignment rule: all compared algorithms are truncated to a unified episode count.

## Figure Notes
- Fig1: 150-episode aligned MARL vs SAC learning curves (mean ± std over seeds).
- Fig2: 150-episode aligned AUC and Final-10 comparison with error bars.
- Fig3: 10-episode aligned MARL/SAC/TD3 early-training comparison.
- Fig4: Success rate by algorithm from run_meta return codes in compare_baseline2.
- Fig5: MARL mechanism metrics showing coordination and service quality behavior.

## Key Interpretation
- In 150-episode aligned setting, AUC: MARL=1507.65, SAC=1475.06.
- In 150-episode aligned setting, Final-10: MARL=1802.54, SAC=1558.18.
- In 10-episode aligned setting: MARL final10=1386.49, SAC final10=1344.58, TD3 final10=1341.27.
- DDPG run success rate is low (0/12), so DDPG should be reported with caveats.

## Writing Guidance
- Use aligned-episode curves/tables for fairness whenever x-axis is training episodes.
- Report seed count and successful-run count together.
- Separate MARL-specific mechanism metrics from direct baseline reward comparison.