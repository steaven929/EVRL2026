"""
诊断充电决策逻辑 - 追踪数据流
"""
import numpy as np
import sys

# 模拟充电决策函数
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def analyze_charging_decision():
    """分析不同SOC和pref组合下的充电概率"""

    # 当前权重配置
    w_current = {"bias": 0.0, "soc": -1.0, "pref": 5.0, "dist": -0.5}
    # 原始权重配置
    w_original = {"bias": 0.0, "soc": -3.0, "pref": 2.0, "dist": -0.5}

    print("=" * 80)
    print("充电决策诊断分析")
    print("=" * 80)

    # 测试不同SOC和pref组合
    soc_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    pref_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    dist_term = 0.1  # 假设距离较近

    print("\n【当前权重配置】soc: -1.0, pref: 5.0, 硬阈值: 0.8")
    print("-" * 80)
    print(f"{'SOC':<6}", end="")
    for pref in pref_values:
        print(f"pref={pref:.1f}  ", end="")
    print()
    print("-" * 80)

    for soc in soc_values:
        print(f"{soc:<6.1f}", end="")
        for pref in pref_values:
            if soc > 0.8:
                prob_str = "BLOCKED"
            else:
                score = (w_current["bias"] +
                        w_current["soc"] * soc +
                        w_current["pref"] * pref +
                        w_current["dist"] * dist_term)
                prob = sigmoid(score)
                prob_str = f"{prob:.3f}"
                if prob >= 0.5:
                    prob_str = f"*{prob_str}*"
            print(f"{prob_str:>10}", end="")
        print()

    print("\n【原始权重配置】soc: -3.0, pref: 2.0, 硬阈值: 0.6")
    print("-" * 80)
    print(f"{'SOC':<6}", end="")
    for pref in pref_values:
        print(f"pref={pref:.1f}  ", end="")
    print()
    print("-" * 80)

    for soc in soc_values:
        print(f"{soc:<6.1f}", end="")
        for pref in pref_values:
            if soc > 0.6:
                prob_str = "BLOCKED"
            else:
                score = (w_original["bias"] +
                        w_original["soc"] * soc +
                        w_original["pref"] * pref +
                        w_original["dist"] * dist_term)
                prob = sigmoid(score)
                prob_str = f"{prob:.3f}"
                if prob >= 0.5:
                    prob_str = f"*{prob_str}*"
            print(f"{prob_str:>10}", end="")
        print()

    # 统计分析
    print("\n" + "=" * 80)
    print("关键发现")
    print("=" * 80)

    # 当前配置下能充电的情况
    charge_count_current = 0
    total_count = 0
    for soc in soc_values:
        for pref in pref_values:
            total_count += 1
            if soc <= 0.8:
                score = (w_current["bias"] +
                        w_current["soc"] * soc +
                        w_current["pref"] * pref +
                        w_current["dist"] * dist_term)
                prob = sigmoid(score)
                if prob >= 0.5:
                    charge_count_current += 1

    # 原始配置下能充电的情况
    charge_count_original = 0
    for soc in soc_values:
        for pref in pref_values:
            if soc <= 0.6:
                score = (w_original["bias"] +
                        w_original["soc"] * soc +
                        w_original["pref"] * pref +
                        w_original["dist"] * dist_term)
                prob = sigmoid(score)
                if prob >= 0.5:
                    charge_count_original += 1

    print(f"\n当前配置: {charge_count_current}/{total_count} 组合会充电 ({charge_count_current/total_count*100:.1f}%)")
    print(f"原始配置: {charge_count_original}/{total_count} 组合会充电 ({charge_count_original/total_count*100:.1f}%)")

    # 分析RL输出的pref值范围
    print("\n" + "=" * 80)
    print("RL输出分析")
    print("=" * 80)
    print("\n从action log看，RL输出的pref值范围:")
    print("  - 时间步0: [0.016, 0.293, 0.251, 0.958, 0.999, 0.570, 0.036, 0.000, 0.214, 0.795, 0.985, 0.998]")
    print("  - 时间步1: [0.741, 0.817, 0.570, 0.928, 0.882, 0.334, 0.581, 0.071, 0.658, 0.806, 0.727, 0.995]")
    print("\n观察: pref值分布广泛，从0.0到1.0都有")
    print("问题: 即使pref=1.0，如果SOC>0.5，充电概率仍然很低")

    # 计算临界点
    print("\n" + "=" * 80)
    print("临界点分析")
    print("=" * 80)

    print("\n当前配置下，要达到p>=0.5的充电概率:")
    for soc in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        # score = bias + soc*(-1.0) + pref*5.0 + dist*(-0.5)
        # 要使sigmoid(score) >= 0.5，需要score >= 0
        # 0 = 0 + soc*(-1.0) + pref*5.0 + 0.1*(-0.5)
        # pref*5.0 = soc + 0.05
        # pref = (soc + 0.05) / 5.0
        required_pref = (soc + 0.05) / 5.0
        print(f"  SOC={soc:.1f} 时，需要 pref >= {required_pref:.3f}")

    print("\n结论:")
    print("  1. 当SOC=0.5时，需要pref>=0.11才会充电")
    print("  2. 当SOC=0.7时，需要pref>=0.15才会充电")
    print("  3. 当SOC>0.8时，无论pref多高都不会充电（硬阈值）")
    print("  4. RL输出的pref值虽然有高有低，但SOC的负权重仍然压制充电意愿")

if __name__ == "__main__":
    analyze_charging_decision()
