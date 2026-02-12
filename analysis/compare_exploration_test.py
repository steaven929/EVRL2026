"""
分析探索期充电偏好值
对比测试期和探索期的充电偏好值分布
"""
import re

def parse_exploration_file(filepath):
    """解析探索期action文件，提取充电偏好值"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取所有时间步的充电偏好值
    pattern = r'时间步 (\d+).*?【充电偏好值】\s*(.*?)(?=\n\n|$)'
    matches = re.findall(pattern, content, re.DOTALL)

    charging_prefs = []
    for timestep, prefs_str in matches:
        # 解析充电偏好值
        zone_pattern = r'区域(\d+):\s*([\d.]+)'
        zone_matches = re.findall(zone_pattern, prefs_str)

        prefs_dict = {'timestep': int(timestep)}
        for zone_id, pref_value in zone_matches:
            prefs_dict[f'zone{zone_id}'] = float(pref_value)

        charging_prefs.append(prefs_dict)

    return charging_prefs

def analyze_and_compare(exploration_prefs, test_prefs):
    """对比分析探索期和测试期的充电偏好值"""
    print("="*80)
    print("探索期 vs 测试期充电偏好值对比分析")
    print("="*80)
    print()

    zones = ['zone0', 'zone1', 'zone2', 'zone3']
    zone_names = ['区域0 (有车辆)', '区域1 (过境区)', '区域2 (过境区)', '区域3 (有车辆)']

    for zone, zone_name in zip(zones, zone_names):
        print(f"{zone_name}:")
        print("-"*80)

        # 探索期统计
        exp_values = [p[zone] for p in exploration_prefs if zone in p]
        if exp_values:
            print(f"  探索期:")
            print(f"    最小值: {min(exp_values):.6f}")
            print(f"    最大值: {max(exp_values):.6f}")
            print(f"    平均值: {sum(exp_values)/len(exp_values):.6f}")
            print(f"    中位数: {sorted(exp_values)[len(exp_values)//2]:.6f}")

        # 测试期统计
        test_values = [p[zone] for p in test_prefs if zone in p]
        if test_values:
            print(f"  测试期:")
            print(f"    最小值: {min(test_values):.6f}")
            print(f"    最大值: {max(test_values):.6f}")
            print(f"    平均值: {sum(test_values)/len(test_values):.6f}")
            print(f"    中位数: {sorted(test_values)[len(test_values)//2]:.6f}")

        # 对比
        if exp_values and test_values:
            exp_avg = sum(exp_values)/len(exp_values)
            test_avg = sum(test_values)/len(test_values)
            ratio = exp_avg / test_avg if test_avg > 0 else float('inf')
            print(f"  对比:")
            print(f"    探索期平均值 / 测试期平均值 = {ratio:.2f}x")

        print()

    # 详细时间步对比
    print("="*80)
    print("每个时间步的充电偏好值对比")
    print("="*80)
    print()
    print(f"{'时间步':<8} {'期别':<8} {'区域0':<12} {'区域1':<12} {'区域2':<12} {'区域3':<12}")
    print("-"*80)

    for i in range(min(len(exploration_prefs), len(test_prefs))):
        exp = exploration_prefs[i]
        test = test_prefs[i]

        print(f"{exp['timestep']:<8} {'探索期':<8} {exp.get('zone0', 0):<12.6f} {exp.get('zone1', 0):<12.6f} {exp.get('zone2', 0):<12.6f} {exp.get('zone3', 0):<12.6f}")
        print(f"{test['timestep']:<8} {'测试期':<8} {test.get('zone0', 0):<12.6f} {test.get('zone1', 0):<12.6f} {test.get('zone2', 0):<12.6f} {test.get('zone3', 0):<12.6f}")
        print()

    # 关键发现
    print("="*80)
    print("关键发现")
    print("="*80)
    print()

    # 检查区域0和3的值是否在探索期更大
    exp_zone0 = [p['zone0'] for p in exploration_prefs if 'zone0' in p]
    test_zone0 = [p['zone0'] for p in test_prefs if 'zone0' in p]
    exp_zone3 = [p['zone3'] for p in exploration_prefs if 'zone3' in p]
    test_zone3 = [p['zone3'] for p in test_prefs if 'zone3' in p]

    if exp_zone0 and test_zone0:
        exp_avg_0 = sum(exp_zone0)/len(exp_zone0)
        test_avg_0 = sum(test_zone0)/len(test_zone0)
        print(f"✓ 区域0（有车辆）:")
        print(f"  探索期平均: {exp_avg_0:.6f}")
        print(f"  测试期平均: {test_avg_0:.6f}")
        print(f"  探索期是测试期的 {exp_avg_0/test_avg_0:.2f} 倍")
        print()

    if exp_zone3 and test_zone3:
        exp_avg_3 = sum(exp_zone3)/len(exp_zone3)
        test_avg_3 = sum(test_zone3)/len(test_zone3)
        print(f"✓ 区域3（有车辆）:")
        print(f"  探索期平均: {exp_avg_3:.6f}")
        print(f"  测试期平均: {test_avg_3:.6f}")
        print(f"  探索期是测试期的 {exp_avg_3/test_avg_3:.2f} 倍")
        print()

    print("⚠️  核心洞察:")
    print("   - 探索期的充电偏好值明显更大（由于±0.3的噪声增强）")
    print("   - 但是测试期的充电偏好值仍然很小（10^-6级别）")
    print("   - 这说明：即使探索期有更多充电，RL也没有学会在有车辆的区域充电")
    print()

if __name__ == '__main__':
    import sys
    import os

    # 切换到项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # 解析探索期数据
    exploration_prefs = parse_exploration_file('action/RL-DDPG_actions_exploration.txt')

    # 解析测试期数据（使用之前的脚本）
    sys.path.insert(0, os.path.join(project_root, 'analysis'))
    from analyze_charging_preferences import parse_action_file
    test_prefs = parse_action_file('action/RL-DDPG_actions.txt')

    # 重定向输出到文件
    with open('exploration_vs_test_comparison.txt', 'w', encoding='utf-8') as f:
        sys.stdout = f
        analyze_and_compare(exploration_prefs, test_prefs)
        sys.stdout = sys.__stdout__

    print("Comparison complete. Results saved to exploration_vs_test_comparison.txt")
