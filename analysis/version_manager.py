# -*- coding: utf-8 -*-
"""
分析结果版本管理
保存每次训练的分析结果，方便对比不同版本
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import shutil
from datetime import datetime
import json

def save_analysis_version(version_name=None):
    """
    保存当前分析结果到版本文件夹

    参数:
        version_name: 版本名称（如 "D2_50epochs", "E3_100epochs"）
                     如果为None，使用时间戳
    """
    if version_name is None:
        version_name = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 创建版本文件夹
    version_dir = f'analysis/versions/{version_name}'
    os.makedirs(version_dir, exist_ok=True)

    print(f"保存分析结果到: {version_dir}")

    # 需要保存的文件列表
    files_to_save = [
        ('analysis/charging_preferences_analysis.txt', '测试期充电偏好值分析'),
        ('analysis/exploration_vs_test_comparison.txt', '探索期vs测试期对比'),
        ('action/RL-DDPG_actions.txt', '测试期详细动作'),
        ('action/RL-DDPG_actions_exploration.txt', '探索期详细动作'),
    ]

    saved_files = []
    for src_file, description in files_to_save:
        if os.path.exists(src_file):
            dst_file = os.path.join(version_dir, os.path.basename(src_file))
            shutil.copy2(src_file, dst_file)
            saved_files.append(src_file)
            print(f"  ✓ {description}")
        else:
            print(f"  ✗ {description} (文件不存在: {src_file})")

    # 保存元数据
    metadata = {
        'version_name': version_name,
        'timestamp': datetime.now().isoformat(),
        'saved_files': saved_files,
        'model_file': 'trained_model.pth' if os.path.exists('trained_model.pth') else None
    }

    # 尝试读取训练参数
    try:
        import numpy as np
        if os.path.exists('rewards.npy'):
            rewards = np.load('rewards.npy')
            metadata['final_reward'] = float(rewards[-1]) if len(rewards) > 0 else None
            metadata['avg_reward'] = float(np.mean(rewards[-10:])) if len(rewards) >= 10 else None
    except Exception as e:
        print(f"  警告: 无法读取奖励数据: {e}")

    with open(os.path.join(version_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 版本 '{version_name}' 保存完成")
    print(f"  位置: {version_dir}")

    return version_dir

def list_versions():
    """列出所有保存的版本"""
    versions_dir = 'analysis/versions'
    if not os.path.exists(versions_dir):
        print("没有保存的版本")
        return []

    versions = []
    for version_name in os.listdir(versions_dir):
        version_path = os.path.join(versions_dir, version_name)
        if os.path.isdir(version_path):
            metadata_file = os.path.join(version_path, 'metadata.json')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                versions.append((version_name, metadata))
            else:
                versions.append((version_name, {}))

    return versions

def compare_versions(version1, version2):
    """对比两个版本的分析结果"""
    print(f"\n{'='*60}")
    print(f"对比版本: {version1} vs {version2}")
    print(f"{'='*60}\n")

    v1_dir = f'analysis/versions/{version1}'
    v2_dir = f'analysis/versions/{version2}'

    # 读取元数据
    try:
        with open(os.path.join(v1_dir, 'metadata.json'), 'r', encoding='utf-8') as f:
            meta1 = json.load(f)
        with open(os.path.join(v2_dir, 'metadata.json'), 'r', encoding='utf-8') as f:
            meta2 = json.load(f)

        print("训练奖励对比:")
        if 'final_reward' in meta1 and 'final_reward' in meta2:
            print(f"  {version1}: {meta1['final_reward']:.2f}")
            print(f"  {version2}: {meta2['final_reward']:.2f}")
            diff = meta2['final_reward'] - meta1['final_reward']
            print(f"  差异: {diff:+.2f}")
        print()
    except Exception as e:
        print(f"无法读取元数据: {e}\n")

    # 对比充电偏好值
    print("充电偏好值对比:")
    try:
        # 简单对比：读取文件并显示关键行
        file1 = os.path.join(v1_dir, 'exploration_vs_test_comparison.txt')
        file2 = os.path.join(v2_dir, 'exploration_vs_test_comparison.txt')

        if os.path.exists(file1) and os.path.exists(file2):
            with open(file1, 'r', encoding='utf-8') as f:
                lines1 = f.readlines()
            with open(file2, 'r', encoding='utf-8') as f:
                lines2 = f.readlines()

            # 查找关键指标
            for i, line in enumerate(lines1):
                if '区域0（有车辆）' in line and '探索期平均' in lines1[i+1]:
                    print(f"  {version1}:")
                    print(f"    {lines1[i+1].strip()}")
                    print(f"    {lines1[i+2].strip()}")
                    break

            for i, line in enumerate(lines2):
                if '区域0（有车辆）' in line and '探索期平均' in lines2[i+1]:
                    print(f"  {version2}:")
                    print(f"    {lines2[i+1].strip()}")
                    print(f"    {lines2[i+2].strip()}")
                    break
        else:
            print("  对比文件不存在")
    except Exception as e:
        print(f"  对比失败: {e}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='分析结果版本管理')
    parser.add_argument('action', choices=['save', 'list', 'compare'],
                       help='操作: save(保存), list(列出), compare(对比)')
    parser.add_argument('--name', type=str, help='版本名称（用于save）')
    parser.add_argument('--v1', type=str, help='版本1（用于compare）')
    parser.add_argument('--v2', type=str, help='版本2（用于compare）')

    args = parser.parse_args()

    if args.action == 'save':
        save_analysis_version(args.name)

    elif args.action == 'list':
        versions = list_versions()
        if not versions:
            print("没有保存的版本")
        else:
            print(f"\n{'='*60}")
            print("已保存的版本")
            print(f"{'='*60}\n")
            for version_name, metadata in versions:
                print(f"版本: {version_name}")
                if 'timestamp' in metadata:
                    print(f"  时间: {metadata['timestamp']}")
                if 'final_reward' in metadata:
                    print(f"  最终奖励: {metadata['final_reward']:.2f}")
                print()

    elif args.action == 'compare':
        if not args.v1 or not args.v2:
            print("错误: 需要指定两个版本 --v1 和 --v2")
        else:
            compare_versions(args.v1, args.v2)

if __name__ == '__main__':
    main()
