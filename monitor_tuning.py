"""
监控超参数调优进度
"""
import os
import time
from pathlib import Path

def check_progress():
    """检查调优进度"""
    tuning_dir = Path("tuning_results")

    if not tuning_dir.exists():
        print("调优尚未开始...")
        return

    # 统计已完成的实验
    completed = []
    for exp_dir in tuning_dir.iterdir():
        if exp_dir.is_dir():
            config_path = exp_dir / "config.json"
            image_path = exp_dir / "final_metrics.png"

            if config_path.exists() and image_path.exists():
                completed.append(exp_dir.name)

    print(f"\n{'='*60}")
    print(f"超参数调优进度监控")
    print(f"{'='*60}")
    print(f"已完成实验: {len(completed)}/5")
    print(f"\n完成列表:")
    for exp in completed:
        print(f"  - {exp}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    while True:
        check_progress()
        time.sleep(30)  # 每30秒检查一次
