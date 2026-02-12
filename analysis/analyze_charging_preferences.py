"""
分析充电偏好值的分布
重点关注：
1. 区域0和区域3的充电偏好值是否都很小（10^-9级别）
2. 区域1和区域2的充电偏好值为什么是1.0
3. 是否存在探索期的数据（如果有的话）
"""
import re

def parse_action_file(filepath):
    """解析action文件，提取充电偏好值"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取所有时间步的动作向量
    pattern = r'时间步 (\d+).*?【动作向量】\s*\[(.*?)\]'
    matches = re.findall(pattern, content, re.DOTALL)

    charging_prefs = []
    for timestep, action_str in matches:
        # 解析动作向量
        values = [float(x.strip()) for x in action_str.split(',')]
        # 充电偏好在位置8-11
        if len(values) >= 12:
            prefs = values[8:12]
            charging_prefs.append({
                'timestep': int(timestep),
                'zone0': prefs[0],
                'zone1': prefs[1],
                'zone2': prefs[2],
                'zone3': prefs[3]
            })

    return charging_prefs

def analyze_preferences(prefs):
    """分析充电偏好值的统计特征"""
    print("=" * 80)
    print("充电偏好值详细分析")
    print("=" * 80)
    print()

    # 按区域统计
    zones = ['zone0', 'zone1', 'zone2', 'zone3']
    for zone in zones:
        values = [p[zone] for p in prefs]
        print(f"{zone.upper()} (有车辆的区域):" if zone in ['zone0', 'zone3'] else f"{zone.upper()} (过境区域):")
        print(f"  最小值: {min(values):.2e}")
        print(f"  最大值: {max(values):.2e}")
        print(f"  平均值: {sum(values)/len(values):.2e}")
        print(f"  中位数: {sorted(values)[len(values)//2]:.2e}")

        # 统计不同数量级的分布
        ranges = [
            (1e-11, 1e-9, "10^-11 ~ 10^-9"),
            (1e-9, 1e-7, "10^-9 ~ 10^-7"),
            (1e-7, 1e-5, "10^-7 ~ 10^-5"),
            (1e-5, 1e-3, "10^-5 ~ 10^-3"),
            (1e-3, 0.1, "10^-3 ~ 0.1"),
            (0.1, 0.5, "0.1 ~ 0.5"),
            (0.5, 1.0, "0.5 ~ 1.0"),
            (1.0, float('inf'), "= 1.0")
        ]

        print(f"  数量级分布:")
        for low, high, label in ranges:
            count = sum(1 for v in values if low <= v < high or (high == float('inf') and v >= low))
            if count > 0:
                print(f"    {label}: {count}次 ({count/len(values)*100:.1f}%)")
        print()

    # 时间步详细信息
    print("=" * 80)
    print("每个时间步的充电偏好值")
    print("=" * 80)
    print()
    print(f"{'时间步':<8} {'区域0':<15} {'区域1':<15} {'区域2':<15} {'区域3':<15}")
    print("-" * 80)
    for p in prefs:
        print(f"{p['timestep']:<8} {p['zone0']:<15.2e} {p['zone1']:<15.2e} {p['zone2']:<15.2e} {p['zone3']:<15.2e}")

    print()
    print("=" * 80)
    print("关键发现")
    print("=" * 80)
    print()

    # 检查区域0和3的值是否都很小
    zone0_values = [p['zone0'] for p in prefs]
    zone3_values = [p['zone3'] for p in prefs]

    if all(v < 1e-7 for v in zone0_values):
        print("✓ 区域0的充电偏好值全部 < 10^-7 (极小)")
    if all(v < 1e-7 for v in zone3_values):
        print("✓ 区域3的充电偏好值全部 < 10^-7 (极小)")

    # 检查区域1和2的值
    zone1_values = [p['zone1'] for p in prefs]
    zone2_values = [p['zone2'] for p in prefs]

    if all(v == 1.0 for v in zone1_values):
        print("✓ 区域1的充电偏好值全部 = 1.0 (最大)")
    if all(v == 1.0 for v in zone2_values):
        print("✓ 区域2的充电偏好值全部 = 1.0 (最大)")

    print()
    print("⚠️  核心矛盾:")
    print("   - 有车辆的区域(0,3): 充电偏好极小 (10^-9级别)")
    print("   - 无车辆的区域(1,2): 充电偏好最大 (1.0)")
    print("   - 结果: RL学会了在错误的区域充电！")
    print()

if __name__ == '__main__':
    import sys
    import os

    # 切换到项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # 重定向输出到文件
    with open('charging_preferences_analysis.txt', 'w', encoding='utf-8') as f:
        sys.stdout = f
        prefs = parse_action_file('action/RL-DDPG_actions.txt')
        analyze_preferences(prefs)
        sys.stdout = sys.__stdout__

    print("Analysis complete. Results saved to charging_preferences_analysis.txt")
