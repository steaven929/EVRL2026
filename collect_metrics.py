# =========2026.2.10修改，改为实时指标提取工具==========
"""
指标提取工具
从训练结果中提取关键性能指标
"""
import numpy as np
import json
import os
from pathlib import Path

def extract_metrics_from_npy(result_dir):
    """从.npy文件中提取指标"""
    metrics = {}

    # 提取Reward指标
    rewards_path = os.path.join(result_dir, "rewards.npy")
    if os.path.exists(rewards_path):
        rewards = np.load(rewards_path)
        metrics["rewards"] = {
            "final_mean": float(np.mean(rewards[-10:])),
            "final_std": float(np.std(rewards[-10:])),
            "best": float(np.max(rewards)),
            "worst": float(np.min(rewards)),
            "initial_mean": float(np.mean(rewards[:10])),
            "improvement": float(np.mean(rewards[-10:]) - np.mean(rewards[:10])),
            "improvement_pct": float((np.mean(rewards[-10:]) - np.mean(rewards[:10])) / abs(np.mean(rewards[:10])) * 100)
        }

    # 提取Critic Loss指标
    loss_path = os.path.join(result_dir, "critic1_losses.npy")
    if os.path.exists(loss_path):
        losses = np.load(loss_path)
        metrics["critic_loss"] = {
            "final_mean": float(np.mean(losses[-10:])),
            "final_std": float(np.std(losses[-10:])),
            "initial_mean": float(np.mean(losses[:10]))
        }

    # 提取Q值指标
    q_path = os.path.join(result_dir, "q_values.npy")
    if os.path.exists(q_path):
        q_vals = np.load(q_path)
        metrics["q_values"] = {
            "final_mean": float(np.mean(q_vals[-10:])),
            "final_std": float(np.std(q_vals[-10:])),
            "initial_mean": float(np.mean(q_vals[:10]))
        }

    return metrics

def compare_experiments(tuning_results_dir="tuning_results"):
    """对比所有实验结果"""
    experiments = []

    # 遍历所有实验目录
    for exp_dir in Path(tuning_results_dir).iterdir():
        if exp_dir.is_dir():
            config_path = exp_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                # 提取指标
                metrics = extract_metrics_from_npy(str(exp_dir))

                experiments.append({
                    "name": config["experiment_name"],
                    "description": config["description"],
                    "hyperparameters": config["hyperparameters"],
                    "metrics": metrics
                })

    # 按最终Reward排序
    experiments.sort(
        key=lambda x: x["metrics"].get("rewards", {}).get("final_mean", float('-inf')),
        reverse=True
    )

    return experiments

def print_comparison_table(experiments):
    """打印对比表格"""
    print("\n" + "="*100)
    print("超参数调优结果对比")
    print("="*100)
    print(f"{'实验名称':<25} {'最终Reward':<15} {'改进幅度':<15} {'Critic LR':<12} {'Batch Size':<12}")
    print("-"*100)

    for exp in experiments:
        name = exp["name"][:24]
        reward = exp["metrics"].get("rewards", {}).get("final_mean", 0)
        improvement = exp["metrics"].get("rewards", {}).get("improvement_pct", 0)
        critic_lr = exp["hyperparameters"].get("critic_lr", "N/A")
        batch_size = exp["hyperparameters"].get("batch_size", "N/A")

        print(f"{name:<25} {reward:<15.2f} {improvement:<15.2f}% {str(critic_lr):<12} {str(batch_size):<12}")

    print("="*100)

if __name__ == "__main__":
    # 对比所有实验
    experiments = compare_experiments()

    # 打印对比表格
    print_comparison_table(experiments)

    # 保存对比结果
    with open("experiments_comparison.json", "w", encoding="utf-8") as f:
        json.dump(experiments, f, indent=2, ensure_ascii=False)

    print("\n✓ 对比结果已保存到 experiments_comparison.json")
# ======修改结束
