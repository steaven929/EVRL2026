"""
Flexible Algorithm Comparison Visualization Script
Usage:
    python plot_flexible_comparison.py --algos SAC TD3 MARL
    python plot_flexible_comparison.py --algos DDPG SAC --output ddpg_vs_sac.png
    python plot_flexible_comparison.py --paths output/sac_seed7 output/td3_seed7
"""

import argparse
import numpy as np
import pandas as pd
import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10


def load_algorithm_data(algo_path):
    """Load all data for a single algorithm"""
    algo_path = Path(algo_path)
    if not algo_path.exists():
        return None

    data = {}

    # Load metadata (try multiple formats)
    metadata_file = algo_path / 'training_metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data['metadata'] = json.load(f)
    else:
        run_meta_file = algo_path / 'run_meta.json'
        if run_meta_file.exists():
            with open(run_meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                data['metadata'] = {
                    'algo_name': meta.get('algo', algo_path.name),
                    'seed': meta.get('seed', 0),
                    'elapsed_sec': meta.get('elapsed_sec', 0)
                }
        else:
            data['metadata'] = {'algo_name': algo_path.name}

    # Load required data
    rewards_file = algo_path / 'rewards.npy'
    if not rewards_file.exists():
        return None
    data['rewards'] = np.load(rewards_file)

    # Load optional data
    optional_files = {
        'critic_losses': ['critic_losses.npy', 'critic1_losses.npy'],
        'critic2_losses': ['critic2_losses.npy'],
        'actor_losses': ['actor_losses.npy'],
        'q_values': ['q_values.npy'],
        'target_q_values': ['target_q_values.npy'],
        'raw_rewards': ['raw_rewards.npy'],
        'eval_rewards': ['eval_rewards.npy'],
        'marl_test_rewards': ['marl_test_rewards.npy', 'test_rewards.npy'],
        'ma_rewards': ['ma_rewards.npy'],
        'svc_history': ['svc_history.npy'],
        'srd_history': ['srd_history.npy'],
        'rld_history': ['rld_history.npy'],
        'zone_coordination': ['zone_coordination_history.npy'],
        'response_readiness': ['response_readiness_history.npy'],
        'fleet_health': ['fleet_health_history.npy'],
        'charging_efficiency': ['charging_efficiency_history.npy'],
        'vehicle_utilization': ['vehicle_utilization_history.npy'],
        'charger_occupancy_rate': ['charger_occupancy_rate_history.npy'],
        'charger_queue_delay': ['charger_queue_delay_history.npy'],
        'energy_not_served': ['energy_not_served_history.npy'],
        'safety_soc_violation_count': ['safety_soc_violation_count_history.npy'],
        'charge_shift_success_rate': ['charge_shift_success_rate_history.npy'],
        'weighted_service_rate': ['weighted_service_rate_history.npy'],
        'effective_service_rate': ['effective_service_rate_history.npy'],
        'service_quality_index': ['service_quality_index_history.npy'],
        'dispatch_efficiency': ['dispatch_efficiency_history.npy'],
        'effective_dispatch_rate': ['effective_dispatch_rate_history.npy'],
        'dispatch_roi': ['dispatch_roi_history.npy'],
    }

    for key, filenames in optional_files.items():
        for filename in filenames:
            file_path = algo_path / filename
            if file_path.exists():
                data[key] = np.load(file_path)
                break

    return data


def compute_stats(rewards):
    return {
        'mean': np.mean(rewards),
        'std': np.std(rewards),
        'max': np.max(rewards),
        'final': rewards[-1],
        'final_10': np.mean(rewards[-10:]),
        'final_50': np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards),
    }


def compute_sample_efficiency(rewards, threshold_pct=0.9):
    if rewards is None or len(rewards) == 0:
        return np.nan
    rewards = np.asarray(rewards, dtype=float)
    target = np.max(rewards) * float(threshold_pct)
    hit = np.where(rewards >= target)[0]
    return float(hit[0]) if len(hit) > 0 else float(len(rewards))


def compute_auc(rewards):
    if rewards is None or len(rewards) == 0:
        return np.nan
    rewards = np.asarray(rewards, dtype=float)
    r_min = np.min(rewards)
    r_max = np.max(rewards)
    norm = (rewards - r_min) / (r_max - r_min + 1e-8)
    return float(np.trapz(norm) / len(norm))


def plot_comparison(algos_data, output_path='comparison.png'):
    """Generate comprehensive comparison plots"""
    n_algos = len(algos_data)

    # Check if we have loss data
    has_losses = any('critic_losses' in data or 'actor_losses' in data
                     for data in algos_data.values())

    # Adjust layout based on available data
    if has_losses:
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    else:
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Training rewards
    ax = fig.add_subplot(gs[0, :2])
    for algo_name, data in algos_data.items():
        rewards = data['rewards']
        episodes = np.arange(len(rewards))
        ax.plot(episodes, rewards, label=algo_name, linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Rewards', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Performance bars
    ax = fig.add_subplot(gs[0, 2])
    metrics = ['Mean', 'Max', 'Final']
    x = np.arange(len(metrics))
    width = 0.8 / n_algos
    for i, (algo_name, data) in enumerate(algos_data.items()):
        stats = compute_stats(data['rewards'])
        values = [stats['mean'], stats['max'], stats['final']]
        offset = (i - n_algos / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=algo_name, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Reward')
    ax.set_title('Performance Metrics', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Box plot
    ax = fig.add_subplot(gs[1, 0])
    data_to_plot = [data['rewards'] for data in algos_data.values()]
    labels = list(algos_data.keys())
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    set3_colors = sns.color_palette('Set3', len(labels))
    for patch, color in zip(bp['boxes'], set3_colors[:len(labels)]):
        patch.set_facecolor(color)
    ax.set_ylabel('Reward')
    ax.set_title('Distribution', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Summary table
    ax = fig.add_subplot(gs[1, 1:])
    ax.axis('off')
    table_data = [['Algorithm', 'Mean', 'Std', 'Max', 'Final', 'Final-10']]
    for algo_name, data in algos_data.items():
        stats = compute_stats(data['rewards'])
        table_data.append([
            algo_name,
            f"{stats['mean']:.1f}",
            f"{stats['std']:.1f}",
            f"{stats['max']:.1f}",
            f"{stats['final']:.1f}",
            f"{stats['final_10']:.1f}"
        ])
    table = ax.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 5. Loss curves (if available)
    if has_losses:
        # Critic losses
        ax = fig.add_subplot(gs[2, 0])
        for algo_name, data in algos_data.items():
            if 'critic_losses' in data:
                losses = data['critic_losses']
                episodes = np.arange(len(losses))
                ax.plot(episodes, losses, label=algo_name, linewidth=1.2, alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Critic Loss')
        ax.set_title('Critic Loss Curves', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Actor losses
        ax = fig.add_subplot(gs[2, 1])
        for algo_name, data in algos_data.items():
            if 'actor_losses' in data:
                losses = data['actor_losses']
                episodes = np.arange(len(losses))
                ax.plot(episodes, losses, label=algo_name, linewidth=1.2, alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Actor Loss')
        ax.set_title('Actor Loss Curves', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Q-values (if available)
        ax = fig.add_subplot(gs[2, 2])
        has_q = False
        for algo_name, data in algos_data.items():
            if 'q_values' in data:
                q_vals = data['q_values']
                episodes = np.arange(len(q_vals))
                ax.plot(episodes, q_vals, label=f'{algo_name} Q', linewidth=1.2, alpha=0.7)
                has_q = True
            if 'target_q_values' in data:
                tq_vals = data['target_q_values']
                episodes = np.arange(len(tq_vals))
                ax.plot(episodes, tq_vals, label=f'{algo_name} Target Q',
                        linewidth=1.2, alpha=0.5, linestyle='--')
                has_q = True
        if has_q:
            ax.set_xlabel('Episode')
            ax.set_ylabel('Q-Value')
            ax.set_title('Q-Value Evolution', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')

    algo_list = ', '.join(algos_data.keys())
    plt.suptitle(f'Comparison: {algo_list}', fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary_bars(algos_data, output_dir):
    """Generate 2x2 summary bars similar to baseline2.0."""
    algos = list(algos_data.keys())
    if not algos:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    specs = [
        ('Reward Last-50 Mean', 'reward_last50', 'higher'),
        ('Eval Mean (single-agent)', 'eval_mean', 'higher'),
        ('SVC Last-10 Mean', 'svc_last10', 'higher'),
        ('RLD Last-10 Mean', 'rld_last10', 'lower'),
    ]

    metric_values = {algo: {} for algo in algos}
    for algo, data in algos_data.items():
        rewards = np.asarray(data.get('rewards', []), dtype=float)
        eval_rewards = np.asarray(data.get('eval_rewards', []), dtype=float)
        svc = np.asarray(data.get('svc_history', []), dtype=float)
        rld = np.asarray(data.get('rld_history', []), dtype=float)
        metric_values[algo]['reward_last50'] = float(np.mean(rewards[-50:])) if rewards.size else np.nan
        metric_values[algo]['eval_mean'] = float(np.mean(eval_rewards)) if eval_rewards.size else np.nan
        metric_values[algo]['svc_last10'] = float(np.mean(svc[-10:])) if svc.size else np.nan
        metric_values[algo]['rld_last10'] = float(np.mean(rld[-10:])) if rld.size else np.nan

    x = np.arange(len(algos))
    for ax, (title, key, pref) in zip(axes, specs):
        vals = [metric_values[a].get(key, np.nan) for a in algos]
        bars = ax.bar(x, vals)
        valid = [v for v in vals if not np.isnan(v)]
        if valid:
            best_idx = int(np.nanargmax(vals) if pref == 'higher' else np.nanargmin(vals))
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(3)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=15, ha='right')
        ax.set_title(f'{title} ({pref} is better)')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'comparison_summary.png'
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_radar_chart(algos_data, output_dir):
    """Generate multi-dimensional normalized radar chart."""
    algos = list(algos_data.keys())
    if not algos:
        return

    dims = [
        ('reward_last50', 'Reward', 'higher'),
        ('svc_last10', 'Service Rate', 'higher'),
        ('srd_last10', 'Service Quality', 'higher'),
        ('rld_last10', 'Efficiency', 'lower'),
        ('reward_std', 'Stability', 'lower'),
        ('conv_ep', 'Convergence', 'lower'),
    ]

    matrix = []
    for _, data in algos_data.items():
        rewards = np.asarray(data.get('rewards', []), dtype=float)
        svc = np.asarray(data.get('svc_history', []), dtype=float)
        srd = np.asarray(data.get('srd_history', []), dtype=float)
        rld = np.asarray(data.get('rld_history', []), dtype=float)

        reward_last50 = float(np.mean(rewards[-50:])) if rewards.size else np.nan
        reward_std = float(np.std(rewards)) if rewards.size else np.nan
        svc_last10 = float(np.mean(svc[-10:])) if svc.size else np.nan
        srd_last10 = float(np.mean(srd[-10:])) if srd.size else np.nan
        rld_last10 = float(np.mean(rld[-10:])) if rld.size else np.nan
        conv_ep = compute_sample_efficiency(rewards, 0.9)

        matrix.append([reward_last50, svc_last10, srd_last10, rld_last10, reward_std, conv_ep])

    matrix = np.asarray(matrix, dtype=float)
    norm = np.zeros_like(matrix)
    for j, (_, _, pref) in enumerate(dims):
        col = matrix[:, j]
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            continue
        cmin = np.nanmin(col)
        cmax = np.nanmax(col)
        if abs(cmax - cmin) < 1e-8:
            ncol = np.full_like(col, 0.5, dtype=float)
        else:
            ncol = (col - cmin) / (cmax - cmin)
        if pref == 'lower':
            ncol = 1.0 - ncol
        norm[:, j] = ncol

    labels = [d[1] for d in dims]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'P', 'X']

    for i, algo in enumerate(algos):
        vals = norm[i].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, '-', linewidth=2, label=algo,
                color=colors[i % len(colors)], marker=markers[i % len(markers)], markersize=6)
        ax.fill(angles, vals, alpha=0.15, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.title('Multi-Dimensional Performance Radar Chart\n(Normalized, Higher is Better)',
              fontsize=13, fontweight='bold', pad=20)

    output_path = output_dir / 'comparison_radar.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_sample_efficiency(algos_data, output_dir):
    """Generate sample efficiency and AUC comparison."""
    algos = list(algos_data.keys())
    if not algos:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    rewards_map = {algo: np.asarray(data.get('rewards', []), dtype=float) for algo, data in algos_data.items()}
    eps90 = [compute_sample_efficiency(rewards_map[a], 0.9) for a in algos]
    eps95 = [compute_sample_efficiency(rewards_map[a], 0.95) for a in algos]

    x = np.arange(len(algos))
    width = 0.35
    bars1 = ax1.bar(x - width / 2, eps90, width, label='90% Performance', alpha=0.8)
    bars2 = ax1.bar(x + width / 2, eps95, width, label='95% Performance', alpha=0.8)
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Episodes Required')
    ax1.set_title('Sample Efficiency: Episodes to Reach Target Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algos, rotation=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax1.text(bar.get_x() + bar.get_width() / 2, h, f'{int(h)}', ha='center', va='bottom', fontsize=8)

    auc_vals = [compute_auc(rewards_map[a]) for a in algos]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = ax2.bar(algos, auc_vals, alpha=0.8, color=[colors[i % len(colors)] for i in range(len(algos))])
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('AUC (Normalized)')
    ax2.set_title('Learning Curve AUC (Higher is Better)')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        h = bar.get_height()
        if np.isfinite(h):
            ax2.text(bar.get_x() + bar.get_width() / 2, h, f'{h:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / 'comparison_sample_efficiency.png'
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_business_metrics(algos_data, output_dir):
    """Generate business metrics bars for SVC/RLD/SRD/Reward."""
    algos = list(algos_data.keys())
    if not algos:
        return

    metrics = [
        ('Service Rate', 'svc_history', 'higher'),
        ('Relocation Dist.', 'rld_history', 'lower'),
        ('Service Quality', 'srd_history', 'higher'),
        ('Reward (Last 50)', 'rewards', 'higher'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    x = np.arange(len(algos))

    for ax, (name, key, pref) in zip(axes, metrics):
        vals = []
        for algo in algos:
            arr = np.asarray(algos_data[algo].get(key, []), dtype=float)
            if arr.size == 0:
                vals.append(np.nan)
            else:
                if key == 'rewards':
                    vals.append(float(np.mean(arr[-50:])))
                else:
                    vals.append(float(np.mean(arr[-10:])))

        bars = ax.bar(x, vals, color=[colors[i % len(colors)] for i in range(len(algos))], alpha=0.8)
        valid = [v for v in vals if not np.isnan(v)]
        if valid:
            best_idx = int(np.nanargmax(vals) if pref == 'higher' else np.nanargmin(vals))
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(3)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=10)
        ax.set_title(f'{name} Comparison ({pref} is better)')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Business Metrics Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = output_dir / 'comparison_business_metrics.png'
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_algorithm_legend(algos_data, output_dir):
    """Generate standalone legend for algorithms."""
    algos = list(algos_data.keys())
    if not algos:
        return

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'P', 'X']

    handles = []
    for i, algo in enumerate(algos):
        handles.append(
            Line2D(
                [0], [0],
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                linewidth=2.4,
                markersize=8,
                label=algo
            )
        )

    fig_w = max(6.0, 1.9 * len(algos))
    fig, ax = plt.subplots(figsize=(fig_w, 1.8))
    ax.axis('off')
    ax.legend(
        handles=handles,
        loc='center',
        ncol=min(4, len(algos)),
        frameon=False,
        fontsize=12,
        handlelength=2.2,
        columnspacing=1.2
    )

    png_path = output_dir / 'comparison_algorithm_legend.png'
    svg_path = output_dir / 'comparison_algorithm_legend.svg'
    plt.tight_layout()
    plt.savefig(png_path, dpi=220, bbox_inches='tight', transparent=True)
    plt.savefig(svg_path, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


def find_latest_run(algo_name, base_dir='output'):
    """Find the latest training run for an algorithm"""
    base_path = Path(base_dir)

    # Search patterns for different algorithm types
    patterns = [
        f"single_{algo_name.lower()}_*ep_seed*",
        f"single_{algo_name.lower()}_seed*",
        f"marl_{algo_name.lower()}_*ep_seed*",
        f"marl_{algo_name.lower()}_seed*",
        f"{algo_name.lower()}_*ep_*",
        f"{algo_name.lower()}_*",
        f"*{algo_name.lower()}*",
    ]

    candidates = []
    for pattern in patterns:
        candidates.extend(base_path.glob(pattern))

    # Also search in compare_baseline2 directory
    compare_dir = base_path / 'compare_baseline2'
    if compare_dir.exists():
        for pattern in patterns:
            candidates.extend(compare_dir.glob(pattern))

    # Filter valid directories with rewards.npy
    valid_dirs = [d for d in candidates if d.is_dir() and (d / 'rewards.npy').exists()]

    if not valid_dirs:
        return None

    # Return the most recently modified directory
    return max(valid_dirs, key=lambda d: d.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description='Flexible Algorithm Comparison')
    parser.add_argument('--algos', nargs='+', help='Algorithm names (e.g., SAC TD3 MARL)')
    parser.add_argument('--paths', nargs='+', help='Algorithm data paths')
    parser.add_argument('--output', default='comparison.png', help='Output path')
    parser.add_argument('--base-dir', default='output', help='Base directory for auto-search')
    parser.add_argument('--latest', action='store_true', help='Use latest runs for each algorithm')
    parser.add_argument('--no-timestamp', action='store_true', help='Do not create timestamped folder')
    args = parser.parse_args()

    # ========== 默认配置：双击运行时使用这些设置 ==========
    # 如果没有传入任何参数（双击运行），使用默认配置
    if not args.paths and not args.algos:
        # 方式1：指定具体路径
        args.paths = [
            'output/marl_20260221-143622',
            'output/marl_20260305-192426',
            'output/compare_baseline2/single_sac_seed7'
        ]
        args.algos = ['MARL-300ep', 'MARL-100ep', 'SAC-700ep']

        # 方式2：或者使用最新结果（注释掉方式1，启用这个）
        # args.algos = ['SAC', 'MARL']
        # args.latest = True

        print("Using default configuration (no command-line args provided)")
        print(f"Comparing: {', '.join(args.algos)}")
    # ========== 默认配置结束 ==========

    # 创建带时间戳的输出文件夹
    if not args.no_timestamp:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path('output/comparison_results') / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # 如果用户指定了输出文件名，使用它；否则使用默认名
        if args.output == 'comparison.png':
            output_path = output_dir / 'comparison.png'
        else:
            output_path = output_dir / Path(args.output).name

        args.output = str(output_path)
        print(f"Output directory: {output_dir}")
    else:
        # 如果用户不想要时间戳文件夹，直接使用指定的路径
        output_path = Path(args.output)
        if output_path.parent != Path('.'):
            output_path.parent.mkdir(parents=True, exist_ok=True)

    algos_data = {}

    if args.paths:
        # Use explicit paths
        for i, path in enumerate(args.paths):
            data = load_algorithm_data(path)
            if data:
                name = args.algos[i] if args.algos and i < len(args.algos) else Path(path).name
                algos_data[name] = data
                print(f"Loaded {name}: {path}")
            else:
                print(f"Warning: Could not load data from {path}")
    elif args.algos:
        # Search for algorithm directories
        for algo_name in args.algos:
            if args.latest:
                # Find latest run
                path = find_latest_run(algo_name, args.base_dir)
                if path:
                    data = load_algorithm_data(path)
                    if data:
                        algos_data[algo_name] = data
                        print(f"Found {algo_name}: {path}")
                    else:
                        print(f"Warning: Could not load data for {algo_name}")
                else:
                    print(f"Warning: No runs found for {algo_name}")
            else:
                # Try predefined paths (支持新旧两种命名格式)
                possible_paths = [
                    f"{args.base_dir}/single_{algo_name.lower()}_*ep_seed7",
                    f"{args.base_dir}/single_{algo_name.lower()}_seed7",
                    f"{args.base_dir}/compare_baseline2/single_{algo_name.lower()}_*ep_seed7",
                    f"{args.base_dir}/compare_baseline2/single_{algo_name.lower()}_seed7",
                    f"{args.base_dir}/marl_{algo_name.lower()}_*ep_seed7",
                    f"{args.base_dir}/marl_{algo_name.lower()}_seed7",
                    f"{args.base_dir}/{algo_name.lower()}_*ep_seed7",
                    f"{args.base_dir}/{algo_name.lower()}_seed7",
                ]
                for path_pattern in possible_paths:
                    # 支持通配符匹配
                    matches = list(Path('.').glob(path_pattern))
                    if matches:
                        # 使用最新的匹配结果
                        path = max(matches, key=lambda p: p.stat().st_mtime)
                        data = load_algorithm_data(path)
                        if data:
                            algos_data[algo_name] = data
                            print(f"Found {algo_name}: {path}")
                            break
                else:
                    print(f"Warning: No data found for {algo_name}")
    else:
        print("Error: Must specify --algos or --paths")
        return

    if len(algos_data) < 2:
        print(f"Error: Need at least 2 algorithms, found {len(algos_data)}")
        return

    # Generate main comparison plot
    plot_comparison(algos_data, args.output)

    # Generate additional comparison plots
    if not args.no_timestamp:
        output_dir = Path(args.output).parent
        print("\nGenerating additional comparison plots...")

        try:
            plot_reward_curves(algos_data, output_dir)
        except Exception as e:
            print(f"Warning: Could not generate reward curves: {e}")

        try:
            plot_extended_comparison(algos_data, output_dir)
        except Exception as e:
            print(f"Warning: Could not generate extended comparison: {e}")

        try:
            plot_convergence_analysis(algos_data, output_dir)
        except Exception as e:
            print(f"Warning: Could not generate convergence analysis: {e}")

        try:
            plot_summary_bars(algos_data, output_dir)
        except Exception as e:
            print(f"Warning: Could not generate summary bars: {e}")

        try:
            plot_radar_chart(algos_data, output_dir)
        except Exception as e:
            print(f"Warning: Could not generate radar chart: {e}")

        try:
            plot_sample_efficiency(algos_data, output_dir)
        except Exception as e:
            print(f"Warning: Could not generate sample efficiency: {e}")

        try:
            plot_business_metrics(algos_data, output_dir)
        except Exception as e:
            print(f"Warning: Could not generate business metrics: {e}")

        try:
            plot_algorithm_legend(algos_data, output_dir)
        except Exception as e:
            print(f"Warning: Could not generate algorithm legend: {e}")

    # 保存对比元数据
    if not args.no_timestamp:
        import json
        import time
        output_dir = Path(args.output).parent
        metadata = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'output_file': Path(args.output).name,
            'algorithms': list(algos_data.keys()),
            'comparison_summary': {}
        }

        for algo_name, data in algos_data.items():
            stats = compute_stats(data['rewards'])
            metadata['comparison_summary'][algo_name] = {
                'mean_reward': float(stats['mean']),
                'max_reward': float(stats['max']),
                'final_reward': float(stats['final']),
                'final_10_avg': float(stats['final_10']),
                'episodes': len(data['rewards']),
                'source_path': data.get('metadata', {}).get('source_path', 'unknown')
            }

        metadata_file = output_dir / 'comparison_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"\nMetadata saved: {metadata_file}")

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for algo_name, data in algos_data.items():
        stats = compute_stats(data['rewards'])
        meta = data.get('metadata', {})
        print(f"\n{algo_name}:")
        print(f"  Mean Reward: {stats['mean']:.1f}")
        print(f"  Max Reward:  {stats['max']:.1f}")
        print(f"  Final Reward: {stats['final']:.1f}")
        print(f"  Final-10 Avg: {stats['final_10']:.1f}")
        if 'elapsed_sec' in meta:
            hours = meta['elapsed_sec'] / 3600
            print(f"  Training Time: {hours:.2f} hours")
    print("=" * 60)


def plot_reward_curves(algos_data, output_dir):
    """Generate reward curves comparison"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for algo_name, data in algos_data.items():
        rewards = data['rewards']
        episodes = np.arange(len(rewards))

        # Plot raw curve
        ax.plot(episodes, rewards, label=algo_name, linewidth=1.5, alpha=0.6)

        # Plot smoothed curve
        if len(rewards) > 10:
            window = min(50, len(rewards) // 10)
            smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
            ax.plot(np.arange(len(smoothed)) + window // 2, smoothed,
                    linewidth=2.5, alpha=0.9)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Reward Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    output_path = output_dir / 'comparison_reward_curves.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_extended_comparison(algos_data, output_dir):
    """Generate extended comparison with 9 metrics"""
    algos = list(algos_data.keys())

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    # Compute metrics for each algorithm
    metrics_data = {}
    for algo_name, data in algos_data.items():
        rewards = data['rewards']
        metrics_data[algo_name] = {
            'mean_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'final_50_mean': np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards),
            'reward_std': np.std(rewards),
            'reward_variance': np.var(rewards),
            'final_10_mean': np.mean(rewards[-10:]),
            'best_final_gap': np.max(rewards) - np.mean(rewards[-10:]),
        }

        # Compute convergence episode (when reward reaches 90% of max)
        max_r = np.max(rewards)
        threshold = 0.9 * max_r
        conv_ep = np.where(rewards >= threshold)[0]
        metrics_data[algo_name]['convergence_ep'] = conv_ep[0] if len(conv_ep) > 0 else len(rewards)

    # Define metrics to plot
    specs = [
        ('final_50_mean', 'Final 50 Episodes Mean', 'higher'),
        ('max_reward', 'Best Reward (Peak)', 'higher'),
        ('mean_reward', 'Overall Mean Reward', 'higher'),
        ('final_10_mean', 'Final 10 Episodes Mean', 'higher'),
        ('reward_std', 'Reward Std Dev', 'lower'),
        ('reward_variance', 'Reward Variance', 'lower'),
        ('convergence_ep', 'Convergence Speed (Episodes)', 'lower'),
        ('best_final_gap', 'Best-Final Gap', 'lower'),
    ]

    x = np.arange(len(algos))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for idx, (ax, (metric_key, title, pref)) in enumerate(zip(axes[:len(specs)], specs)):
        values = [metrics_data[algo][metric_key] for algo in algos]

        bars = ax.bar(x, values, color=[colors[i % len(colors)] for i in range(len(algos))])

        # Highlight best
        if pref == 'higher':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)

        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=15, ha='right')
        ax.set_title(f"{title} ({pref} is better)", fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, val in enumerate(values):
            ax.text(i, val, f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    # Hide unused subplots
    for idx in range(len(specs), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Extended Algorithm Comparison (8 Metrics)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = output_dir / 'comparison_extended.png'
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_convergence_analysis(algos_data, output_dir):
    """Generate convergence and stability analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # 1. Convergence curves (cumulative max)
    ax = axes[0]
    for algo_name, data in algos_data.items():
        rewards = data['rewards']
        cummax = np.maximum.accumulate(rewards)
        ax.plot(cummax, label=algo_name, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Max Reward')
    ax.set_title('Convergence Progress', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Rolling std (stability)
    ax = axes[1]
    window = 50
    for algo_name, data in algos_data.items():
        rewards = data['rewards']
        if len(rewards) >= window:
            rolling_std = pd.Series(rewards).rolling(window=window).std()
            ax.plot(rolling_std, label=algo_name, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel(f'Rolling Std (window={window})')
    ax.set_title('Training Stability', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Improvement rate
    ax = axes[2]
    for algo_name, data in algos_data.items():
        rewards = data['rewards']
        if len(rewards) > 1:
            improvement = np.diff(rewards)
            ax.plot(improvement, label=algo_name, linewidth=1, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode-to-Episode Improvement')
    ax.set_title('Learning Progress', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Distribution comparison
    ax = axes[3]
    data_to_plot = [data['rewards'] for data in algos_data.values()]
    labels = list(algos_data.keys())
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    set3_colors = sns.color_palette('Set3', len(labels))
    for patch, color in zip(bp['boxes'], set3_colors[:len(labels)]):
        patch.set_facecolor(color)
    ax.set_ylabel('Reward')
    ax.set_title('Reward Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

    plt.suptitle('Convergence and Stability Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = output_dir / 'comparison_convergence.png'
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    main()
