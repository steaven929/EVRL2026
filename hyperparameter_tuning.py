"""
自动化超参数调优脚本
根据预设的超参数组合，自动修改util.py、运行训练、收集结果
"""
import os
import shutil
import subprocess
import json
from datetime import datetime
import re

# 定义超参数组合
EXPERIMENTS = [
    {
        "name": "baseline_600eps",
        "description": "基线配置，train_eps=600",
        "params": {
            "train_eps": 600,
            "gamma": 0.5,
            "critic_lr": "5e-4",
            "actor_lr": "5e-5",
            "batch_size": 64,
            "tau": "5e-3"
        }
    },
    {
        "name": "high_gamma_large_batch",
        "description": "高gamma+大批量，增强长期规划",
        "params": {
            "train_eps": 600,
            "gamma": 0.85,
            "critic_lr": "5e-4",
            "actor_lr": "5e-5",
            "batch_size": 128,
            "tau": "5e-3"
        }
    },
    {
        "name": "low_lr_stable",
        "description": "低学习率，追求稳定收敛",
        "params": {
            "train_eps": 600,
            "gamma": 0.5,
            "critic_lr": "1e-4",
            "actor_lr": "1e-5",
            "batch_size": 64,
            "tau": "5e-3"
        }
    },
    {
        "name": "high_lr_fast",
        "description": "高学习率，加速学习",
        "params": {
            "train_eps": 600,
            "gamma": 0.5,
            "critic_lr": "1e-3",
            "actor_lr": "1e-4",
            "batch_size": 64,
            "tau": "5e-3"
        }
    },
    {
        "name": "fast_target_update",
        "description": "快速目标网络更新+中等gamma",
        "params": {
            "train_eps": 600,
            "gamma": 0.7,
            "critic_lr": "5e-4",
            "actor_lr": "5e-5",
            "batch_size": 64,
            "tau": "1e-2"
        }
    }
]

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

    print(f"[Modified] util.py updated with new parameters")

def run_training():
    """运行训练"""
    print("[Training] Starting training...")
    result = subprocess.run(
        ["python", "main.py"],
        capture_output=True,
        text=True
    )
    print("[Training] Training completed")
    return result.returncode == 0

# =========2026.2.10修改，简化结果收集==========
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
    return result_dir, hyperparams
# ======修改结束

# =========2026.2.10修改，简化日志记录==========
def log_to_tuning_log(exp_name, exp_desc, params, result_dir, hyperparams, success):
    """记录到tuning_log.md"""
    log_entry = f"""
## {exp_name}
**时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**描述**: {exp_desc}
**状态**: {'✓ 成功' if success else '✗ 失败'}

### 超参数配置 (从 util.py 读取)
```
train_eps: {hyperparams.get('train_eps', 'N/A')}
gamma: {hyperparams.get('gamma', 'N/A')}
critic_lr: {hyperparams.get('critic_lr', 'N/A')}
actor_lr: {hyperparams.get('actor_lr', 'N/A')}
batch_size: {hyperparams.get('batch_size', 'N/A')}
memory_capacity: {hyperparams.get('memory_capacity', 'N/A')}
target_update: {hyperparams.get('target_update', 'N/A')}
tau: {hyperparams.get('tau', 'N/A')}
critic_hidden_dim: {hyperparams.get('critic_hidden_dim', 'N/A')}
actor_hidden_dim: {hyperparams.get('actor_hidden_dim', 'N/A')}
seed: {hyperparams.get('seed', 'N/A')}
```

**结果路径**: `{result_dir}/`
**训练图像**: `{result_dir}/final_metrics.png`

---

"""

    # 如果文件不存在，创建并添加标题
    if not os.path.exists("tuning_log.md"):
        with open("tuning_log.md", "w", encoding="utf-8") as f:
            f.write("# 超参数调优日志\n\n")
            f.write("记录所有超参数实验的配置和结果\n\n")
            f.write("---\n\n")

    # 追加日志
    with open("tuning_log.md", "a", encoding="utf-8") as f:
        f.write(log_entry)

    print(f"[Logged] Experiment logged to tuning_log.md")
# ======修改结束

def main():
    print("="*60)
    print("超参数自动调优系统")
    print("="*60)
    print(f"计划运行 {len(EXPERIMENTS)} 组实验\n")

    # 创建结果目录
    os.makedirs("tuning_results", exist_ok=True)

    # 备份原始util.py
    backup_util()

    try:
        for i, exp in enumerate(EXPERIMENTS, 1):
            print(f"\n{'='*60}")
            print(f"实验 {i}/{len(EXPERIMENTS)}: {exp['name']}")
            print(f"描述: {exp['description']}")
            print('='*60)

            # 修改util.py
            modify_util(exp['params'])

            # 运行训练
            success = run_training()

            # 收集结果
            result_dir, hyperparams = collect_results(
                exp['name'],
                exp['description'],
                exp['params']
            )

            # 记录到日志
            log_to_tuning_log(
                exp['name'],
                exp['description'],
                exp['params'],
                result_dir,
                hyperparams,
                success
            )

            print(f"\n[完成] 实验 {i}/{len(EXPERIMENTS)} 完成")

    finally:
        # 恢复原始util.py
        restore_util()

    print(f"\n{'='*60}")
    print("所有实验完成！")
    print(f"结果保存在: tuning_results/")
    print(f"日志文件: tuning_log.md")
    print('='*60)

if __name__ == "__main__":
    main()

