# -*- coding: utf-8 -*-
"""
探索期充电行为详细分析
分析哪些车在充电，充电偏好值如何映射到实际充电行为
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import re
import numpy as np

def parse_exploration_actions(filepath):
    """解析探索期action文件，提取充电偏好值和实际充电信息"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取总奖励
    total_reward_match = re.search(r'总奖励:\s*([\d.]+)', content)
    total_reward = float(total_reward_match.group(1)) if total_reward_match else None

    # 提取每个时间步的信息
    timesteps = []
    pattern = r'时间步 (\d+).*?【充电偏好值】\s*(.*?)(?=\n\n|时间步|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)

    for timestep, prefs_str in matches:
        # 解析充电偏好值
        zone_pattern = r'区域(\d+):\s*([\d.]+)'
        zone_matches = re.findall(zone_pattern, prefs_str)

        prefs = {}
        for zone_id, pref_value in zone_matches:
            prefs[int(zone_id)] = float(pref_value)

        timesteps.append({
            'timestep': int(timestep),
            'prefs': prefs
        })

    return total_reward, timesteps

def analyze_charging_mapping(total_reward, timesteps):
    """分析充电偏好值到实际充电的映射"""

    # 重定向输出到文件
    output_file = 'analysis/exploration_charging_behavior_detail.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f

        print("="*80)
        print("探索期充电行为详细分析")
        print("="*80)
        print()
        print(f"总奖励: {total_reward:.2f}")
        print(f"总时间步: {len(timesteps)}")
        print()

        print("="*80)
        print("充电偏好值 → 充电数量映射分析")
        print("="*80)
        print()

        # 假设每个区域有15辆空闲车辆（基于之前的分析）
        idle_vehicles_per_zone = {
            0: 15,  # 有车辆区域
            1: 0,   # 过境区域
            2: 0,   # 过境区域
            3: 15   # 有车辆区域
        }

        print("假设空闲车辆分布（基于需求分析）:")
        for zone, count in idle_vehicles_per_zone.items():
            zone_type = "有车辆区域" if zone in [0, 3] else "过境区域"
            print(f"  区域{zone} ({zone_type}): {count}辆")
        print()

        print("="*80)
        print("分段映射规则（当前实现）:")
        print("="*80)
        print("  pref > 0.8  → 充电100%空闲车辆")
        print("  pref 0.5-0.8 → 充电70%空闲车辆")
        print("  pref 0.3-0.5 → 充电40%空闲车辆")
        print("  pref 0.1-0.3 → 充电20%空闲车辆")
        print("  pref < 0.1  → 不充电")
        print()

        print("="*80)
        print("每个时间步的充电决策分析")
        print("="*80)
        print()

        total_charging_events = 0
        charging_by_zone = {0: 0, 1: 0, 2: 0, 3: 0}

        for ts_data in timesteps:
            timestep = ts_data['timestep']
            prefs = ts_data['prefs']

            print(f"时间步 {timestep}:")
            print("-"*80)

            for zone in range(4):
                pref = prefs.get(zone, 0)
                idle = idle_vehicles_per_zone[zone]

                # 应用分段映射规则
                if pref > 0.8:
                    charge_count = idle
                    rule = "pref > 0.8 → 100%"
                elif pref > 0.5:
                    charge_count = max(1, int(idle * 0.7)) if idle > 0 else 0
                    rule = "pref 0.5-0.8 → 70%"
                elif pref > 0.3:
                    charge_count = max(1, int(idle * 0.4)) if idle > 0 else 0
                    rule = "pref 0.3-0.5 → 40%"
                elif pref > 0.1:
                    charge_count = int(idle * 0.2)
                    rule = "pref 0.1-0.3 → 20%"
                else:
                    charge_count = 0
                    rule = "pref < 0.1 → 0%"

                zone_type = "有车辆" if zone in [0, 3] else "过境区"

                print(f"  区域{zone} ({zone_type}):")
                print(f"    充电偏好值: {pref:.6f}")
                print(f"    空闲车辆数: {idle}辆")
                print(f"    映射规则: {rule}")
                print(f"    预期充电数: {charge_count}辆")

                if charge_count > 0:
                    print(f"    ✓ 会充电 {charge_count}辆")
                    total_charging_events += charge_count
                    charging_by_zone[zone] += charge_count

                    # 说明哪些车会被选中充电
                    if idle > 0:
                        print(f"    充电车辆选择: 按SOC升序排序，选择电量最低的{charge_count}辆")
                        print(f"    示例: 如果车辆SOC为[200,250,300,...,900]")
                        print(f"          则充电车辆为前{charge_count}辆（SOC最低）")
                else:
                    if idle == 0:
                        print(f"    ✗ 无法充电（没有空闲车辆）")
                    else:
                        print(f"    ✗ 不充电（偏好值太小）")

                print()

            print()

        print("="*80)
        print("充电统计总结")
        print("="*80)
        print()
        print(f"总充电事件数: {total_charging_events}次")
        print()
        print("各区域充电次数:")
        for zone, count in charging_by_zone.items():
            zone_type = "有车辆区域" if zone in [0, 3] else "过境区域"
            print(f"  区域{zone} ({zone_type}): {count}次")
        print()

        # 对比分析
        print("="*80)
        print("关键发现")
        print("="*80)
        print()

        # 检查区域0和3的充电情况
        zone0_charges = charging_by_zone[0]
        zone3_charges = charging_by_zone[3]
        zone1_charges = charging_by_zone[1]
        zone2_charges = charging_by_zone[2]

        if zone0_charges > 0 or zone3_charges > 0:
            print(f"✓ 有车辆区域有充电:")
            if zone0_charges > 0:
                print(f"  区域0: {zone0_charges}次")
            if zone3_charges > 0:
                print(f"  区域3: {zone3_charges}次")
        else:
            print("✗ 有车辆区域没有充电")

        print()

        if zone1_charges > 0 or zone2_charges > 0:
            print(f"⚠️  过境区域有充电（异常）:")
            if zone1_charges > 0:
                print(f"  区域1: {zone1_charges}次")
            if zone2_charges > 0:
                print(f"  区域2: {zone2_charges}次")
        else:
            print("✓ 过境区域没有充电（正常，因为没有空闲车辆）")

        print()

        # 分析充电偏好值分布
        print("="*80)
        print("充电偏好值分布分析")
        print("="*80)
        print()

        for zone in range(4):
            zone_type = "有车辆区域" if zone in [0, 3] else "过境区域"
            prefs_list = [ts['prefs'].get(zone, 0) for ts in timesteps]

            print(f"区域{zone} ({zone_type}):")
            print(f"  最小值: {min(prefs_list):.6f}")
            print(f"  最大值: {max(prefs_list):.6f}")
            print(f"  平均值: {np.mean(prefs_list):.6f}")
            print(f"  中位数: {np.median(prefs_list):.6f}")

            # 统计超过阈值的次数
            above_01 = sum(1 for p in prefs_list if p > 0.1)
            above_03 = sum(1 for p in prefs_list if p > 0.3)
            above_05 = sum(1 for p in prefs_list if p > 0.5)

            print(f"  超过0.1阈值: {above_01}/{len(prefs_list)}次 ({above_01/len(prefs_list)*100:.1f}%)")
            print(f"  超过0.3阈值: {above_03}/{len(prefs_list)}次 ({above_03/len(prefs_list)*100:.1f}%)")
            print(f"  超过0.5阈值: {above_05}/{len(prefs_list)}次 ({above_05/len(prefs_list)*100:.1f}%)")
            print()

        print("="*80)
        print("核心洞察")
        print("="*80)
        print()

        # 计算平均充电偏好值
        zone0_avg = np.mean([ts['prefs'].get(0, 0) for ts in timesteps])
        zone3_avg = np.mean([ts['prefs'].get(3, 0) for ts in timesteps])

        print("1. 充电偏好值 vs 充电阈值:")
        print(f"   区域0平均偏好值: {zone0_avg:.6f}")
        if zone0_avg > 0.1:
            print(f"   ✓ 超过0.1阈值，会触发充电")
        else:
            print(f"   ✗ 低于0.1阈值，不会触发充电")

        print(f"   区域3平均偏好值: {zone3_avg:.6f}")
        if zone3_avg > 0.1:
            print(f"   ✓ 超过0.1阈值，会触发充电")
        else:
            print(f"   ✗ 低于0.1阈值，不会触发充电")

        print()
        print("2. 充电车辆选择逻辑:")
        print("   ✓ 优先充电低电量车辆（按SOC升序排序）")
        print("   ✓ 只充电空闲车辆（未执行任务的车辆）")
        print("   ✓ 考虑充电桩可用性")
        print()

        print("3. 映射公式的影响:")
        print("   当前阈值0.1可能过高:")
        if zone0_avg < 0.1:
            print(f"   - 区域0平均偏好值{zone0_avg:.6f} < 0.1，不充电")
            print(f"   - 建议降低阈值到0.05，可以触发充电")
        if zone3_avg > 0.1:
            print(f"   - 区域3平均偏好值{zone3_avg:.6f} > 0.1，会充电")
            print(f"   - 预期充电{int(15 * 0.2)}辆（20%规则）")

        print()

        sys.stdout = sys.__stdout__

    return output_file

def main():
    print("Starting exploration charging behavior analysis...")

    # 解析探索期动作文件
    exploration_file = 'action/RL-DDPG_actions_exploration.txt'

    try:
        total_reward, timesteps = parse_exploration_actions(exploration_file)
        output_file = analyze_charging_mapping(total_reward, timesteps)

        print("Analysis complete")
        print(f"Results saved to: {output_file}")
        print()
        print("Key information preview:")

        # 读取并显示关键部分
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            in_summary = False
            for line in lines:
                if '充电统计总结' in line:
                    in_summary = True
                if in_summary:
                    print(line.rstrip())
                    if '核心洞察' in line:
                        # 再打印几行
                        for _ in range(20):
                            try:
                                print(next(f).rstrip())
                            except StopIteration:
                                break
                        break

    except FileNotFoundError:
        print(f"Error: File not found {exploration_file}")
        print("Please run first: python analysis/save_exploration_actions.py")
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
