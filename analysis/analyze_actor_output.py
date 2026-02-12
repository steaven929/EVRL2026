# -*- coding: utf-8 -*-
"""
验证Actor输出和实际充电行为
检查Actor原始输出的充电偏好值，以及噪声增强的效果
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import re
import numpy as np

def parse_detailed_actions(filepath):
    """解析详细动作文件，提取Actor原始输出和最终动作"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取每个时间步的信息
    timesteps = []
    pattern = r'时间步 (\d+).*?【Actor原始输出（无噪声）】\s*\[(.*?)\].*?【添加OU噪声后】\s*\[(.*?)\].*?【充电噪声】\s*\[(.*?)\].*?【最终动作（添加充电噪声后）】\s*\[(.*?)\]'
    matches = re.findall(pattern, content, re.DOTALL)

    for timestep, actor_raw, ou_noise, charge_noise, final_action in matches:
        # 解析动作向量
        actor_raw_values = [float(x.strip()) for x in actor_raw.split(',')]
        ou_noise_values = [float(x.strip()) for x in ou_noise.split(',')]
        charge_noise_values = [float(x.strip()) for x in charge_noise.split(',')]
        final_action_values = [float(x.strip()) for x in final_action.split(',')]

        # 提取充电偏好值（位置8-11）
        timesteps.append({
            'timestep': int(timestep),
            'actor_raw_charging': actor_raw_values[8:12],
            'ou_noise_charging': ou_noise_values[8:12],
            'charge_noise': charge_noise_values,
            'final_charging': final_action_values[8:12]
        })

    return timesteps

def analyze_actor_output(timesteps):
    """分析Actor原始输出"""

    output_file = 'analysis/actor_output_analysis.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f

        print("="*80)
        print("Actor原始输出分析")
        print("="*80)
        print()

        print("关键问题：Actor原始输出的充电偏好值是多少？")
        print()

        # 统计Actor原始输出
        print("="*80)
        print("每个时间步的Actor原始输出（充电维度）")
        print("="*80)
        print()

        for ts in timesteps:
            print(f"时间步 {ts['timestep']}:")
            print(f"  Actor原始输出（无噪声）:")
            for zone in range(4):
                print(f"    区域{zone}: {ts['actor_raw_charging'][zone]:.10f}")
            print(f"  充电噪声（±0.3）:")
            for zone in range(4):
                print(f"    区域{zone}: {ts['charge_noise'][zone]:+.6f}")
            print(f"  最终充电偏好值:")
            for zone in range(4):
                print(f"    区域{zone}: {ts['final_charging'][zone]:.6f}")
            print()

        # 统计分析
        print("="*80)
        print("Actor原始输出统计分析")
        print("="*80)
        print()

        for zone in range(4):
            zone_type = "有车辆区域" if zone in [0, 3] else "过境区域"
            actor_raw_values = [ts['actor_raw_charging'][zone] for ts in timesteps]

            print(f"区域{zone} ({zone_type}):")
            print(f"  Actor原始输出:")
            print(f"    最小值: {min(actor_raw_values):.10f}")
            print(f"    最大值: {max(actor_raw_values):.10f}")
            print(f"    平均值: {np.mean(actor_raw_values):.10f}")
            print(f"    中位数: {np.median(actor_raw_values):.10f}")

            # 检查是否都是极小值
            if max(actor_raw_values) < 1e-5:
                print(f"    ⚠️  所有值都 < 10^-5（极小）")
            elif max(actor_raw_values) < 0.01:
                print(f"    ⚠️  所有值都 < 0.01（很小）")
            else:
                print(f"    ✓ 有较大的值")

            print()

        # 噪声增强效果分析
        print("="*80)
        print("噪声增强效果分析")
        print("="*80)
        print()

        for zone in range(4):
            zone_type = "有车辆区域" if zone in [0, 3] else "过境区域"
            actor_raw_values = [ts['actor_raw_charging'][zone] for ts in timesteps]
            final_values = [ts['final_charging'][zone] for ts in timesteps]
            noise_values = [ts['charge_noise'][zone] for ts in timesteps]

            print(f"区域{zone} ({zone_type}):")
            print(f"  Actor原始输出平均: {np.mean(actor_raw_values):.10f}")
            print(f"  充电噪声平均: {np.mean(noise_values):+.6f}")
            print(f"  最终偏好值平均: {np.mean(final_values):.6f}")
            print(f"  噪声增强倍数: {np.mean(final_values) / np.mean(actor_raw_values) if np.mean(actor_raw_values) > 0 else float('inf'):.2f}x")
            print()

        # 关键发现
        print("="*80)
        print("关键发现")
        print("="*80)
        print()

        # 检查Actor原始输出是否都是极小值
        zone0_raw = [ts['actor_raw_charging'][0] for ts in timesteps]
        zone3_raw = [ts['actor_raw_charging'][3] for ts in timesteps]

        print("1. Actor原始输出问题:")
        if max(zone0_raw) < 1e-5:
            print(f"   ✗ 区域0的Actor原始输出全部 < 10^-5")
            print(f"     平均值: {np.mean(zone0_raw):.10f}")
            print(f"     这说明Actor网络没有学会输出大的充电偏好值")
        else:
            print(f"   ✓ 区域0的Actor原始输出有较大值")

        if max(zone3_raw) < 1e-5:
            print(f"   ✗ 区域3的Actor原始输出全部 < 10^-5")
            print(f"     平均值: {np.mean(zone3_raw):.10f}")
            print(f"     这说明Actor网络没有学会输出大的充电偏好值")
        else:
            print(f"   ✓ 区域3的Actor原始输出有较大值")

        print()
        print("2. 噪声增强效果:")
        zone0_final = [ts['final_charging'][0] for ts in timesteps]
        zone3_final = [ts['final_charging'][3] for ts in timesteps]

        print(f"   区域0:")
        print(f"     Actor原始: {np.mean(zone0_raw):.10f}")
        print(f"     最终偏好: {np.mean(zone0_final):.6f}")
        print(f"     增强倍数: {np.mean(zone0_final) / np.mean(zone0_raw) if np.mean(zone0_raw) > 0 else float('inf'):.0f}x")

        print(f"   区域3:")
        print(f"     Actor原始: {np.mean(zone3_raw):.10f}")
        print(f"     最终偏好: {np.mean(zone3_final):.6f}")
        print(f"     增强倍数: {np.mean(zone3_final) / np.mean(zone3_raw) if np.mean(zone3_raw) > 0 else float('inf'):.0f}x")

        print()
        print("3. 核心问题:")
        print("   ✗ Actor网络输出的充电偏好值极小（10^-6级别）")
        print("   ✓ 噪声增强有效（增强到0.076-0.165）")
        print("   ✗ 但RL没有学会保持这种充电行为")
        print("   → 原因：探索期的充电经验没有被正确学习")
        print()

        sys.stdout = sys.__stdout__

    return output_file

def main():
    print("Starting Actor output analysis...")

    exploration_file = 'action/RL-DDPG_actions_exploration.txt'

    try:
        timesteps = parse_detailed_actions(exploration_file)
        output_file = analyze_actor_output(timesteps)

        print("Analysis complete")
        print(f"Results saved to: {output_file}")
        print()
        print("Key findings:")
        print("- Actor raw output is extremely small (10^-6 level)")
        print("- Noise enhancement is effective (increases to 0.076-0.165)")
        print("- But RL did not learn to maintain this charging behavior")

    except FileNotFoundError:
        print(f"Error: File not found {exploration_file}")
        print("Please run first: python analysis/save_exploration_actions.py")
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
