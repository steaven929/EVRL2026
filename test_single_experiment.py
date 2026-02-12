"""
测试单个超参数实验
用于快速验证调优系统是否正常工作
"""
import os
import shutil
import subprocess
import json
from datetime import datetime
import re

# 定义测试实验（使用较少的训练轮数以加快测试）
TEST_EXPERIMENT = {
    "name": "test_experiment",
    "description": "测试实验，train_eps=10（快速验证）",
    "params": {
        "train_eps": 10,
        "gamma": 0.5,
        "critic_lr": "5e-4",
        "actor_lr": "5e-5",
        "batch_size": 64,
        "tau": "5e-3"
    }
}

def backup_util():
    """备份原始util.py"""
    shutil.copy("core/util.py", "core/util.py.backup")
    print("[Backup] util.py backed up to util.py.backup")

def restore_util():
    """恢复原始util.py"""
    if os.path.exists("core/util.py.backup"):
        shutil.copy("core/util.py.backup", "core/util.py")
        print("[Restore] util.py restored from backup")

def modify_util(params):
    """修改util.py中的超参数"""
    with open("core/util.py", "r", encoding="utf-8") as f:
        content = f.read()

    # 替换各个参数
    for param_name, param_value in params.items():
        if param_name == "train_eps":
            content = re.sub(
                r"(parser\.add_argument\('--train_eps',default=)\d+",
                f"\\g<1>{param_value}",
                content
            )
        elif param_name == "gamma":
            content = re.sub(
                r"(parser\.add_argument\('--gamma',default=)[\d.]+",
                f"\\g<1>{param_value}",
                content
            )
        elif param_name == "critic_lr":
            content = re.sub(
                r"(parser\.add_argument\('--critic_lr',default=)[\de\-]+",
                f"\\g<1>{param_value}",
                content
            )
        elif param_name == "actor_lr":
            content = re.sub(
                r"(parser\.add_argument\('--actor_lr',default=)[\de\-]+",
                f"\\g<1>{param_value}",
                content
            )
        elif param_name == "batch_size":
            content = re.sub(
                r"(parser\.add_argument\('--batch_size',default=)\d+",
                f"\\g<1>{param_value}",
                content
            )
        elif param_name == "tau":
            content = re.sub(
                r"(parser\.add_argument\('--tau',default=)[\de\-]+",
                f"\\g<1>{param_value}",
                content
            )

    with open("core/util.py", "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[Modified] util.py updated with test parameters")

def run_training():
    """运行训练"""
    print("[Training] Starting training...")
    result = subprocess.run(
        ["python", "main.py"],
        capture_output=True,
        text=True
    )

    # 打印输出（用于调试）
    if result.stdout:
        print("\n[Training Output]:")
        print(result.stdout[-1000:])  # 只显示最后1000字符

    if result.returncode != 0:
        print("\n[Training Error]:")
        print(result.stderr[-1000:])

    print("[Training] Training completed")
    return result.returncode == 0

def collect_results(exp_name, exp_desc, params):
    """收集训练结果：final_metrics.png + util.py超参数"""
    # 创建结果目录
    result_dir = f"tuning_results/{exp_name}"
    os.makedirs(result_dir, exist_ok=True)

    # 1. 复制 final_metrics.png
    if os.path.exists("final_metrics.png"):
        shutil.copy("final_metrics.png", f"{result_dir}/final_metrics.png")
        print(f"[Collected] final_metrics.png -> {result_dir}/")
    else:
        print(f"[Warning] final_metrics.png not found")

    # 2. 读取当前 util.py 中的所有超参数
    with open("core/util.py", "r", encoding="utf-8") as f:
        util_content = f.read()

    # 提取所有超参数
    hyperparams = {}
    patterns = {
        'train_eps': r"'--train_eps',default=(\d+)",
        'gamma': r"'--gamma',default=([\d.]+)",
        'critic_lr': r"'--critic_lr',default=([\de\-]+)",
        'actor_lr': r"'--actor_lr',default=([\de\-]+)",
        'batch_size': r"'--batch_size',default=(\d+)",
        'memory_capacity': r"'--memory_capacity',default=(\d+)",
        'target_update': r"'--target_update',default=(\d+)",
        'tau': r"'--tau',default=([\de\-]+)",
        'critic_hidden_dim': r"'--critic_hidden_dim',default=(\d+)",
        'actor_hidden_dim': r"'--actor_hidden_dim',default=(\d+)",
        'seed': r"'--seed',default=(\d+)"
    }

    for param, pattern in patterns.items():
        match = re.search(pattern, util_content)
        if match:
            hyperparams[param] = match.group(1)

    # 3. 保存配置
    config = {
        "experiment_name": exp_name,
        "description": exp_desc,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparameters": hyperparams
    }

    with open(f"{result_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"[Collected] Hyperparameters saved to {result_dir}/config.json")

    # 打印收集到的超参数
    print("\n[Collected Hyperparameters]:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")

    return result_dir, hyperparams

def main():
    print("="*60)
    print("测试单个超参数实验")
    print("="*60)
    print(f"实验名称: {TEST_EXPERIMENT['name']}")
    print(f"描述: {TEST_EXPERIMENT['description']}")
    print("="*60)

    # 备份原始util.py
    backup_util()

    try:
        # 修改util.py
        modify_util(TEST_EXPERIMENT['params'])

        # 运行训练
        success = run_training()

        # 收集结果
        result_dir, hyperparams = collect_results(
            TEST_EXPERIMENT['name'],
            TEST_EXPERIMENT['description'],
            TEST_EXPERIMENT['params']
        )

        print(f"\n{'='*60}")
        if success:
            print("[SUCCESS] Test completed successfully!")
            print(f"Results saved in: {result_dir}/")
            print(f"  - final_metrics.png")
            print(f"  - config.json")
        else:
            print("[FAILED] Training failed, please check error messages")
        print('='*60)

    finally:
        # 恢复原始util.py
        restore_util()

if __name__ == "__main__":
    main()
