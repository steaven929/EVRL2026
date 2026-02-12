"""
超参数调优结果可视化
生成对比图表，展示不同超参数配置的性能差异
"""
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False

def load_experiments(tuning_results_dir="tuning_results"):
    """加载所有实验结果"""
    experiments = []

    for exp_dir in Path(tuning_results_dir).iterdir():
        if exp_dir.is_dir():
            config_path = exp_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                experiments.append(config)

    # 按最终Reward排序
    experiments.sort(
        key=lambda x: x.get("metrics", {}).get("final_reward_mean", float('-inf')),
        reverse=True
    )

    return experiments

def plot_reward_comparison(experiments):
    """绘制Reward对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    names = [exp["experiment_name"][:20] for exp in experiments]
    rewards = [exp.get("metrics", {}).get("final_reward_mean", 0) for exp in experiments]
    improvements = [exp.get("metrics", {}).get("improvement_pct", 0) for exp in experiments]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, rewards, width, label='最终Reward', color='steelblue')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, improvements, width, label='改进幅度(%)', color='coral')

    ax.set_xlabel('实验配置', fontsize=12)
    ax.set_ylabel('最终Reward', fontsize=12)
    ax2.set_ylabel('改进幅度 (%)', fontsize=12)
    ax.set_title('超参数调优 - Reward对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)

    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('tuning_reward_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存: tuning_reward_comparison.png")
    plt.close()

def plot_hyperparameter_heatmap(experiments):
    """绘制超参数热力图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    params_to_plot = ['gamma', 'critic_lr', 'actor_lr', 'batch_size', 'tau', 'train_eps']
    param_labels = ['Gamma', 'Critic LR', 'Actor LR', 'Batch Size', 'Tau', 'Train Episodes']

    for idx, (param, label) in enumerate(zip(params_to_plot, param_labels)):
        ax = axes[idx]

        # 提取数据
        param_values = []
        rewards = []
        for exp in experiments:
            val = exp.get("hyperparameters", {}).get(param, None)
            if val is not None:
                # 转换科学计数法
                if isinstance(val, str) and 'e' in val:
                    val = float(val)
                param_values.append(float(val))
                rewards.append(exp.get("metrics", {}).get("final_reward_mean", 0))

        if param_values:
            # 散点图
            scatter = ax.scatter(param_values, rewards, c=rewards, cmap='viridis',
                               s=100, alpha=0.6, edgecolors='black')
            ax.set_xlabel(label, fontsize=11)
            ax.set_ylabel('最终Reward', fontsize=11)
            ax.set_title(f'{label} vs Reward', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)

            # 如果是学习率，使用对数刻度
            if 'lr' in param or param == 'tau':
                ax.set_xscale('log')

            plt.colorbar(scatter, ax=ax, label='Reward')

    plt.tight_layout()
    plt.savefig('tuning_hyperparameter_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存: tuning_hyperparameter_heatmap.png")
    plt.close()

def plot_training_curves(tuning_results_dir="tuning_results"):
    """绘制所有实验的训练曲线对比"""
    fig, ax = plt.subplots(figsize=(14, 7))

    for exp_dir in Path(tuning_results_dir).iterdir():
        if exp_dir.is_dir():
            rewards_path = exp_dir / "rewards.npy"
            config_path = exp_dir / "config.json"

            if rewards_path.exists() and config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                rewards = np.load(str(rewards_path))
                exp_name = config["experiment_name"][:20]

                # 平滑曲线
                window = min(10, len(rewards) // 10)
                if window > 1:
                    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                else:
                    smoothed = rewards

                ax.plot(smoothed, label=exp_name, linewidth=2, alpha=0.8)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('训练曲线对比 (所有实验)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('tuning_training_curves.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存: tuning_training_curves.png")
    plt.close()

def generate_summary_table(experiments):
    """生成汇总表格图"""
    fig, ax = plt.subplots(figsize=(16, len(experiments) * 0.6 + 2))
    ax.axis('tight')
    ax.axis('off')

    # 准备表格数据
    headers = ['实验名称', '最终Reward', '改进幅度', 'Gamma', 'Critic LR', 'Batch Size', 'Tau']
    table_data = []

    for exp in experiments:
        row = [
            exp["experiment_name"][:20],
            f"{exp.get('metrics', {}).get('final_reward_mean', 0):.2f}",
            f"{exp.get('metrics', {}).get('improvement_pct', 0):.2f}%",
            str(exp.get('hyperparameters', {}).get('gamma', 'N/A')),
            str(exp.get('hyperparameters', {}).get('critic_lr', 'N/A')),
            str(exp.get('hyperparameters', {}).get('batch_size', 'N/A')),
            str(exp.get('hyperparameters', {}).get('tau', 'N/A'))
        ]
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.2, 0.12, 0.12, 0.1, 0.12, 0.12, 0.12])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 设置表头样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 设置行颜色
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.title('超参数调优汇总表', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('tuning_summary_table.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存: tuning_summary_table.png")
    plt.close()

if __name__ == "__main__":
    print("="*60)
    print("生成超参数调优对比图表")
    print("="*60)

    # 加载实验结果
    experiments = load_experiments()

    if not experiments:
        print("❌ 未找到实验结果，请先运行 hyperparameter_tuning.py")
        exit(1)

    print(f"\n找到 {len(experiments)} 组实验结果\n")

    # 生成各类图表
    plot_reward_comparison(experiments)
    plot_hyperparameter_heatmap(experiments)
    plot_training_curves()
    generate_summary_table(experiments)

    print("\n" + "="*60)
    print("所有图表生成完成！")
    print("="*60)
    print("生成的文件:")
    print("  - tuning_reward_comparison.png")
    print("  - tuning_hyperparameter_heatmap.png")
    print("  - tuning_training_curves.png")
    print("  - tuning_summary_table.png")
    print("="*60)
