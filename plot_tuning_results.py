"""
超参数调优结果可视化（简化版）
展示 final_metrics.png 和对应的超参数配置
"""
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_experiments(tuning_results_dir="tuning_results"):
    """加载所有实验结果"""
    experiments = []

    for exp_dir in Path(tuning_results_dir).iterdir():
        if exp_dir.is_dir():
            config_path = exp_dir / "config.json"
            image_path = exp_dir / "final_metrics.png"

            if config_path.exists() and image_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                config['image_path'] = str(image_path)
                experiments.append(config)

    return experiments

def create_comparison_grid(experiments):
    """创建对比网格图"""
    n_exp = len(experiments)
    if n_exp == 0:
        print("❌ 未找到实验结果")
        return

    # 计算网格布局
    n_cols = min(2, n_exp)
    n_rows = (n_exp + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(16, 8 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

    for idx, exp in enumerate(experiments):
        row = idx // n_cols
        col = idx % n_cols

        # 创建子图
        ax = fig.add_subplot(gs[row, col])

        # 显示图像
        img = mpimg.imread(exp['image_path'])
        ax.imshow(img)
        ax.axis('off')

        # 添加标题和超参数信息
        title = f"{exp['experiment_name']}\n"
        title += f"时间: {exp['timestamp']}\n\n"

        hyperparams = exp['hyperparameters']
        title += f"train_eps={hyperparams.get('train_eps', 'N/A')}, "
        title += f"gamma={hyperparams.get('gamma', 'N/A')}\n"
        title += f"critic_lr={hyperparams.get('critic_lr', 'N/A')}, "
        title += f"actor_lr={hyperparams.get('actor_lr', 'N/A')}\n"
        title += f"batch_size={hyperparams.get('batch_size', 'N/A')}, "
        title += f"tau={hyperparams.get('tau', 'N/A')}"

        ax.set_title(title, fontsize=10, pad=10, loc='left')

    plt.suptitle('超参数调优结果对比', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('tuning_comparison_grid.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存: tuning_comparison_grid.png")
    plt.close()

def create_hyperparameter_table(experiments):
    """创建超参数对比表格"""
    fig, ax = plt.subplots(figsize=(18, len(experiments) * 0.8 + 2))
    ax.axis('tight')
    ax.axis('off')

    # 准备表格数据
    headers = ['实验名称', '时间', 'train_eps', 'gamma', 'critic_lr',
               'actor_lr', 'batch_size', 'tau', 'memory_cap', 'target_update']
    table_data = []

    for exp in experiments:
        hp = exp['hyperparameters']
        row = [
            exp['experiment_name'][:25],
            exp['timestamp'].split()[1],  # 只显示时间
            hp.get('train_eps', 'N/A'),
            hp.get('gamma', 'N/A'),
            hp.get('critic_lr', 'N/A'),
            hp.get('actor_lr', 'N/A'),
            hp.get('batch_size', 'N/A'),
            hp.get('tau', 'N/A'),
            hp.get('memory_capacity', 'N/A'),
            hp.get('target_update', 'N/A')
        ]
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.08, 0.08, 0.08, 0.1, 0.1, 0.1, 0.08, 0.1, 0.1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
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

    plt.title('超参数配置对比表', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('tuning_hyperparameter_table.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存: tuning_hyperparameter_table.png")
    plt.close()

def print_summary(experiments):
    """打印汇总信息"""
    print("\n" + "="*100)
    print("超参数调优结果汇总")
    print("="*100)
    print(f"{'实验名称':<30} {'时间':<20} {'Gamma':<10} {'Critic LR':<12} {'Batch Size':<12}")
    print("-"*100)

    for exp in experiments:
        name = exp['experiment_name'][:29]
        time = exp['timestamp']
        hp = exp['hyperparameters']
        gamma = hp.get('gamma', 'N/A')
        critic_lr = hp.get('critic_lr', 'N/A')
        batch_size = hp.get('batch_size', 'N/A')

        print(f"{name:<30} {time:<20} {str(gamma):<10} {str(critic_lr):<12} {str(batch_size):<12}")

    print("="*100)

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

    # 打印汇总
    print_summary(experiments)

    # 生成图表
    create_comparison_grid(experiments)
    create_hyperparameter_table(experiments)

    print("\n" + "="*60)
    print("所有图表生成完成！")
    print("="*60)
    print("生成的文件:")
    print("  - tuning_comparison_grid.png (final_metrics对比网格)")
    print("  - tuning_hyperparameter_table.png (超参数对比表)")
    print("="*60)
