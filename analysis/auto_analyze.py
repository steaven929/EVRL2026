# -*- coding: utf-8 -*-
"""
训练后自动分析脚本
在训练完成后自动运行所有分析工具，生成对比报告
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import subprocess
from datetime import datetime

def run_command(cmd, description):
    """运行命令并捕获输出"""
    print(f"\n{'='*60}")
    print(f"正在执行: {description}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✓ {description} 完成")
        return True
    else:
        print(f"✗ {description} 失败")
        print(f"错误信息: {result.stderr}")
        return False

def main():
    print("="*60)
    print("训练后自动分析")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 检查是否存在训练好的模型
    if not os.path.exists('trained_model.pth'):
        print("错误: 未找到 trained_model.pth")
        print("请先运行训练: python main.py --train_eps 50")
        return

    # 检查是否存在测试期动作文件
    if not os.path.exists('action/RL-DDPG_actions.txt'):
        print("错误: 未找到 action/RL-DDPG_actions.txt")
        print("请确保训练脚本已经运行测试阶段")
        return

    # 步骤1：分析测试期充电偏好值
    success1 = run_command(
        'python analysis/analyze_charging_preferences.py',
        '分析测试期充电偏好值'
    )

    # 步骤2：保存探索期动作数据
    success2 = run_command(
        'python analysis/save_exploration_actions.py',
        '保存探索期动作数据'
    )

    # 步骤3：对比探索期和测试期
    success3 = run_command(
        'python analysis/compare_exploration_test.py',
        '对比探索期和测试期'
    )

    # 生成总结报告
    print("\n" + "="*60)
    print("分析完成总结")
    print("="*60)

    if success1 and success2 and success3:
        print("✓ 所有分析步骤成功完成")
        print()
        print("生成的文件:")
        print("  - analysis/charging_preferences_analysis.txt")
        print("  - action/RL-DDPG_actions_exploration.txt")
        print("  - analysis/exploration_vs_test_comparison.txt")
        print()
        print("查看结果:")
        print("  cat analysis/charging_preferences_analysis.txt")
        print("  cat analysis/exploration_vs_test_comparison.txt")
    else:
        print("✗ 部分分析步骤失败，请检查错误信息")

    # 读取关键指标并显示
    print("\n" + "="*60)
    print("关键指标快速预览")
    print("="*60)

    try:
        with open('analysis/exploration_vs_test_comparison.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 查找关键发现部分
            in_key_findings = False
            for line in lines:
                if '关键发现' in line:
                    in_key_findings = True
                if in_key_findings:
                    print(line.rstrip())
                    if '核心洞察' in line:
                        # 再打印几行
                        for _ in range(5):
                            try:
                                print(next(f).rstrip())
                            except StopIteration:
                                break
                        break
    except Exception as e:
        print(f"无法读取对比结果: {e}")

if __name__ == '__main__':
    main()
