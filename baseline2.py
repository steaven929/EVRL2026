import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 导入充电分析模块
from charging_analysis import plot_charging_comprehensive, plot_charging_efficiency_radar
from charging_visualization import plot_charging_dashboard
from marl_charging_analysis import plot_marl_charging_strategy, plot_marl_charging_comparison


def _parse_list(raw: str):
    return [x.strip() for x in raw.split(",") if x.strip()]


def _load_npy(path: Path):
    if not path.exists():
        return None
    arr = np.load(path, allow_pickle=True)
    return np.asarray(arr, dtype=float).reshape(-1)


def _safe_mean(arr):
    if arr is None or arr.size == 0:
        return np.nan
    return float(np.mean(arr))


def _safe_last(arr):
    if arr is None or arr.size == 0:
        return np.nan
    return float(arr[-1])


def _safe_mean_last(arr, k):
    if arr is None or arr.size == 0:
        return np.nan
    kk = min(k, arr.size)
    return float(np.mean(arr[-kk:]))


def _safe_max(arr):
    if arr is None or arr.size == 0:
        return np.nan
    return float(np.max(arr))


def _safe_std(arr):
    """计算标准差"""
    if arr is None or arr.size == 0:
        return np.nan
    return float(np.std(arr))


def _compute_reward_variance(arr):
    """计算奖励方差系数 (std/mean)"""
    if arr is None or arr.size == 0:
        return np.nan
    mean_val = np.mean(arr)
    if abs(mean_val) < 1e-6:
        return np.nan
    return float(np.std(arr) / abs(mean_val))


def _compute_convergence_episode(arr, threshold=0.9):
    """计算收敛轮数（达到最优90%的轮数）"""
    if arr is None or arr.size == 0:
        return np.nan
    max_val = np.max(arr)
    target = max_val * threshold
    for i, val in enumerate(arr):
        if val >= target:
            return float(i)
    return float(len(arr))  # 未收敛，返回总轮数


def _compute_best_final_gap(arr):
    """计算峰值与最终值的差距（过拟合程度）"""
    if arr is None or arr.size == 0:
        return np.nan
    return float(np.max(arr) - arr[-1])


# =========2026.3.4新增，样本效率和统计检验==========
def _compute_sample_efficiency(arr, threshold_pct=0.9):
    """
    计算样本效率：达到目标性能所需的轮数

    Args:
        arr: 奖励数组
        threshold_pct: 目标性能百分比（默认90%）

    Returns:
        达到目标所需的轮数，如果未达到则返回总轮数
    """
    if arr is None or arr.size == 0:
        return np.nan
    max_val = np.max(arr)
    target = max_val * threshold_pct
    for i, val in enumerate(arr):
        if val >= target:
            return float(i)
    return float(len(arr))


def _compute_auc(arr):
    """计算学习曲线的AUC（Area Under Curve）"""
    if arr is None or arr.size == 0:
        return np.nan
    # 归一化到[0,1]
    arr_norm = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
    return float(np.trapz(arr_norm) / len(arr_norm))


def _statistical_test(group1, group2):
    """
    统计显著性检验（t检验）

    Args:
        group1, group2: 两组数据（例如不同算法的奖励）

    Returns:
        dict: 包含t统计量、p值、效应量
    """
    from scipy import stats

    if len(group1) < 2 or len(group2) < 2:
        return {"t_stat": np.nan, "p_value": np.nan, "effect_size": np.nan, "significant": False}

    # t检验
    t_stat, p_value = stats.ttest_ind(group1, group2)

    # Cohen's d (效应量)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((len(group1)-1)*std1**2 + (len(group2)-1)*std2**2) / (len(group1)+len(group2)-2))
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "effect_size": float(cohens_d),
        "significant": p_value < 0.05
    }
# ======修改结束


def _collect_root_artifacts(workdir: Path, since_ts: float):
    exts = {".npy", ".png", ".pth"}
    files = []
    for p in workdir.iterdir():
        if p.is_file() and p.suffix.lower() in exts and p.stat().st_mtime >= since_ts - 1:
            files.append(p)
    return files


def _find_latest_marl_output(workdir: Path, since_ts: float):
    out_root = workdir / "outputs"
    if not out_root.exists():
        return None
    cands = [
        p for p in out_root.iterdir()
        if p.is_dir() and p.name.startswith("marl_") and p.stat().st_mtime >= since_ts - 1
    ]
    if not cands:
        return None
    cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0]


def _summarize_run(run_dir: Path, algo: str, seed: int):
    rewards = _load_npy(run_dir / "rewards.npy")
    eval_rewards = _load_npy(run_dir / "eval_rewards.npy")
    marl_test_rewards = _load_npy(run_dir / "marl_test_rewards.npy")
    svc = _load_npy(run_dir / "svc_history.npy")
    rld = _load_npy(run_dir / "rld_history.npy")
    srd = _load_npy(run_dir / "srd_history.npy")
    critic1 = _load_npy(run_dir / "critic1_losses.npy")
    critic_losses = _load_npy(run_dir / "critic_losses.npy")

    return {
        "algo": algo,
        "seed": seed,
        "final_reward": _safe_last(rewards),
        "best_reward": _safe_max(rewards),
        "reward_last50_mean": _safe_mean_last(rewards, 50),
        "eval_last": _safe_last(eval_rewards),
        "eval_mean": _safe_mean(eval_rewards),
        "marl_test_mean": _safe_mean(marl_test_rewards),
        "svc_last10_mean": _safe_mean_last(svc, 10),
        "rld_last10_mean": _safe_mean_last(rld, 10),
        "srd_last10_mean": _safe_mean_last(srd, 10),
        "critic1_last100_mean": _safe_mean_last(critic1, 100),
        "critic_last100_mean": _safe_mean_last(critic_losses, 100),
        # =========2026.3.3修改，添加新指标==========
        "reward_variance": _compute_reward_variance(rewards),
        "convergence_episode": _compute_convergence_episode(rewards, threshold=0.9),
        "best_final_gap": _compute_best_final_gap(rewards),
        "reward_std": _safe_std(rewards),
        # ======修改结束
    }


def _aggregate(rows, key):
    vals = [r[key] for r in rows if np.isfinite(r[key])]
    if not vals:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals))


def _plot_summary_bars(algo_summary, output_root: Path, stamp: str):
    algos = [r["algo"] for r in algo_summary]
    if not algos:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    specs = [
        ("reward_last50_mean", "reward_last50_std", "Reward Last50 Mean", "higher"),
        ("eval_mean", "eval_mean_std", "Eval Mean (single-agent)", "higher"),
        ("svc_last10_mean", "svc_last10_std", "SVC Last10 Mean", "higher"),
        ("rld_last10_mean", "rld_last10_std", "RLD Last10 Mean", "lower"),
    ]

    x = np.arange(len(algos))
    for ax, (m_key, s_key, title, pref) in zip(axes, specs):
        means = [r.get(m_key, np.nan) for r in algo_summary]
        stds = [r.get(s_key, np.nan) for r in algo_summary]
        ax.bar(x, means, yerr=stds, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=15)
        ax.set_title(f"{title} ({pref} is better)")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = output_root / f"comparison_summary_{stamp}.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()


# =========2026.3.3修改，方案1：扩展对比图（3×3子图）==========
def _plot_extended_comparison(algo_summary, output_root: Path, stamp: str):
    """生成扩展对比图（3×3子图，9个指标）"""
    algos = [r["algo"] for r in algo_summary]
    if not algos:
        return

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    specs = [
        ("reward_last50_mean", "reward_last50_std", "Reward Last50 Mean", "higher"),
        ("best_reward_mean", "best_reward_std", "Best Reward (Peak Performance)", "higher"),
        ("svc_last10_mean", "svc_last10_std", "Service Rate (SVC)", "higher"),
        ("srd_last10_mean", "srd_last10_std", "Service Reward Ratio (SRD)", "higher"),
        ("rld_last10_mean", "rld_last10_std", "Relocation Distance (RLD)", "lower"),
        ("reward_variance_mean", "reward_variance_std", "Reward Variance (Stability)", "lower"),
        ("convergence_episode_mean", "convergence_episode_std", "Convergence Speed (Episodes)", "lower"),
        ("best_final_gap_mean", "best_final_gap_std", "Best-Final Gap (Overfitting)", "lower"),
        ("reward_std_mean", "reward_std_std", "Reward Std Dev", "lower"),
    ]

    x = np.arange(len(algos))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 4种颜色对应4个算法

    for idx, (ax, (m_key, s_key, title, pref)) in enumerate(zip(axes, specs)):
        means = [r.get(m_key, np.nan) for r in algo_summary]
        stds = [r.get(s_key, np.nan) for r in algo_summary]

        # 使用不同颜色
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=[colors[i % len(colors)] for i in range(len(algos))])

        # 标注最优值（处理全NaN情况）
        valid_means = [m for m in means if not np.isnan(m)]
        if valid_means:  # 只有当有有效值时才标注
            if pref == "higher":
                best_idx = np.nanargmax(means)
            else:
                best_idx = np.nanargmin(means)
            if not np.isnan(means[best_idx]):
                bars[best_idx].set_edgecolor('red')
                bars[best_idx].set_linewidth(3)

        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=15, ha='right')
        ax.set_title(f"{title} ({pref} is better)", fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, (mean, std) in enumerate(zip(means, stds)):
            if not np.isnan(mean):
                ax.text(i, mean + (std if not np.isnan(std) else 0), f'{mean:.2f}',
                       ha='center', va='bottom', fontsize=8)

    plt.suptitle('Extended Algorithm Comparison (9 Metrics)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    out = output_root / f"comparison_extended_{stamp}.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
# ======修改结束


# =========2026.3.3修改，方案2：多维性能雷达图==========
def _plot_radar_chart(algo_summary, output_root: Path, stamp: str):
    """生成多维性能雷达图"""
    algos = [r["algo"] for r in algo_summary]
    if not algos:
        return

    # 定义6个维度（归一化到0-1）
    dimensions = [
        ("reward_last50_mean", "Reward", "higher"),
        ("svc_last10_mean", "Service Rate", "higher"),
        ("srd_last10_mean", "Service Quality", "higher"),
        ("rld_last10_mean", "Efficiency", "lower"),  # RLD越低越好，需要反转
        ("reward_variance_mean", "Stability", "lower"),  # 方差越低越好，需要反转
        ("convergence_episode_mean", "Convergence", "lower"),  # 轮数越少越好，需要反转
    ]

    # 提取数据并归一化
    data_matrix = []
    for algo_row in algo_summary:
        values = []
        for key, label, pref in dimensions:
            val = algo_row.get(key, np.nan)
            values.append(val)
        data_matrix.append(values)

    data_matrix = np.array(data_matrix)

    # 归一化到0-1（处理NaN）
    normalized_data = np.zeros_like(data_matrix)
    for col_idx in range(data_matrix.shape[1]):
        col = data_matrix[:, col_idx]
        valid_mask = ~np.isnan(col)
        if valid_mask.sum() == 0:
            continue

        col_min = np.nanmin(col)
        col_max = np.nanmax(col)

        if abs(col_max - col_min) < 1e-6:
            normalized_data[:, col_idx] = 0.5
        else:
            # 归一化
            normalized_col = (col - col_min) / (col_max - col_min)

            # 如果是"lower is better"，反转
            _, _, pref = dimensions[col_idx]
            if pref == "lower":
                normalized_col = 1.0 - normalized_col

            normalized_data[:, col_idx] = normalized_col

    # 绘制雷达图
    labels = [label for _, label, _ in dimensions]
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    for idx, (algo, values) in enumerate(zip(algos, normalized_data)):
        values_plot = values.tolist()
        values_plot += values_plot[:1]  # 闭合

        ax.plot(angles, values_plot, 'o-', linewidth=2, label=algo,
               color=colors[idx % len(colors)], marker=markers[idx % len(markers)], markersize=8)
        ax.fill(angles, values_plot, alpha=0.15, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    plt.title('Multi-Dimensional Performance Radar Chart\\n(Normalized, Higher is Better)',
             fontsize=14, fontweight='bold', pad=20)

    out = output_root / f"comparison_radar_{stamp}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
# ======修改结束


# =========2026.3.3修改，方案2：收敛速度分析图==========
def _plot_convergence_analysis(per_run_rows, output_root: Path, stamp: str):
    """生成收敛速度分析图"""
    algo_to_curves = {}
    for row in per_run_rows:
        run_dir = Path(row.get("run_dir", ""))
        rewards = _load_npy(run_dir / "rewards.npy")
        if rewards is None or rewards.size == 0:
            continue
        algo_to_curves.setdefault(row["algo"], []).append(rewards)

    if not algo_to_curves:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # 子图1: 收敛速度对比（达到90%最优的轮数）
    algos = sorted(algo_to_curves.keys())
    convergence_episodes = []
    convergence_stds = []

    for algo in algos:
        curves = algo_to_curves[algo]
        conv_eps = []
        for curve in curves:
            max_val = np.max(curve)
            target = max_val * 0.9
            for i, val in enumerate(curve):
                if val >= target:
                    conv_eps.append(i)
                    break
            else:
                conv_eps.append(len(curve))
        convergence_episodes.append(np.mean(conv_eps))
        convergence_stds.append(np.std(conv_eps))

    x = np.arange(len(algos))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = axes[0].bar(x, convergence_episodes, yerr=convergence_stds, capsize=4,
                      color=[colors[i % len(colors)] for i in range(len(algos))])

    # 标注最快收敛
    best_idx = np.argmin(convergence_episodes)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(algos, rotation=15, ha='right')
    axes[0].set_ylabel('Episodes to 90% Optimal')
    axes[0].set_title('Convergence Speed (Lower is Better)', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    for i, (mean, std) in enumerate(zip(convergence_episodes, convergence_stds)):
        axes[0].text(i, mean + std, f'{mean:.1f}', ha='center', va='bottom', fontsize=9)

    # 子图2: 训练稳定性（奖励曲线的一阶差分标准差）
    smoothness_scores = []
    smoothness_stds = []

    for algo in algos:
        curves = algo_to_curves[algo]
        smooth_vals = []
        for curve in curves:
            if len(curve) > 1:
                diff = np.diff(curve)
                smooth_vals.append(np.std(diff))
        smoothness_scores.append(np.mean(smooth_vals) if smooth_vals else np.nan)
        smoothness_stds.append(np.std(smooth_vals) if smooth_vals else np.nan)

    bars = axes[1].bar(x, smoothness_scores, yerr=smoothness_stds, capsize=4,
                      color=[colors[i % len(colors)] for i in range(len(algos))])

    # 标注最平滑
    valid_scores = [s for s in smoothness_scores if not np.isnan(s)]
    if valid_scores:
        best_idx = np.nanargmin(smoothness_scores)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(algos, rotation=15, ha='right')
    axes[1].set_ylabel('Std of Reward Differences')
    axes[1].set_title('Training Smoothness (Lower is Better)', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # 子图3: 累积收敛曲线
    max_len = max(len(curves[0]) for curves in algo_to_curves.values() if curves)
    for algo in algos:
        curves = algo_to_curves[algo]
        convergence_counts = np.zeros(max_len)

        for curve in curves:
            max_val = np.max(curve)
            target = max_val * 0.9
            for i in range(len(curve)):
                if curve[i] >= target:
                    convergence_counts[i:] += 1
                    break

        convergence_rate = convergence_counts / len(curves)
        axes[2].plot(range(max_len), convergence_rate, linewidth=2.5, label=algo,
                    marker='o', markersize=4, markevery=max(1, max_len // 10))

    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Fraction of Runs Converged')
    axes[2].set_title('Cumulative Convergence Rate', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 1.05])

    # 子图4: 峰值性能对比
    best_rewards = []
    best_stds = []

    for algo in algos:
        curves = algo_to_curves[algo]
        best_vals = [np.max(curve) for curve in curves]
        best_rewards.append(np.mean(best_vals))
        best_stds.append(np.std(best_vals))

    bars = axes[3].bar(x, best_rewards, yerr=best_stds, capsize=4,
                      color=[colors[i % len(colors)] for i in range(len(algos))])

    # 标注最高峰值
    best_idx = np.argmax(best_rewards)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)

    axes[3].set_xticks(x)
    axes[3].set_xticklabels(algos, rotation=15, ha='right')
    axes[3].set_ylabel('Best Reward')
    axes[3].set_title('Peak Performance (Higher is Better)', fontweight='bold')
    axes[3].grid(True, alpha=0.3, axis='y')

    for i, (mean, std) in enumerate(zip(best_rewards, best_stds)):
        axes[3].text(i, mean + std, f'{mean:.0f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Convergence and Stability Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    out = output_root / f"comparison_convergence_{stamp}.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
# ======修改结束


# =========2026.3.3修改，方案3：LaTeX表格生成==========
def _generate_latex_table(algo_summary, output_root: Path, stamp: str):
    """生成LaTeX对比表格"""
    algos = [r["algo"] for r in algo_summary]
    if not algos:
        return

    # 定义要包含的指标
    metrics = [
        ("reward_last50", "Reward (Last 50)", "higher", "%.1f"),
        ("best_reward", "Best Reward", "higher", "%.1f"),
        ("svc_last10", "Service Rate (\\%)", "higher", "%.2f"),
        ("srd_last10", "Service Quality", "higher", "%.3f"),
        ("rld_last10", "Relocation Dist.", "lower", "%.2f"),
        ("reward_variance", "Reward Variance", "lower", "%.4f"),
        ("convergence_episode", "Convergence (Eps)", "lower", "%.1f"),
        ("best_final_gap", "Overfitting Gap", "lower", "%.1f"),
    ]

    # 开始生成LaTeX代码
    latex_lines = []
    latex_lines.append("% Algorithm Comparison Table")
    latex_lines.append("% Generated by baseline2.py")
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Comprehensive Algorithm Comparison}")
    latex_lines.append("\\label{tab:algo_comparison}")
    latex_lines.append("\\begin{tabular}{l" + "c" * len(algos) + "}")
    latex_lines.append("\\toprule")

    # 表头
    header = "Metric & " + " & ".join(algos) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")

    # 统计Win/Tie/Loss
    win_counts = {algo: 0 for algo in algos}
    tie_counts = {algo: 0 for algo in algos}
    loss_counts = {algo: 0 for algo in algos}

    # 每个指标一行
    for metric_key, metric_name, pref, fmt in metrics:
        values = []
        stds = []
        for algo_row in algo_summary:
            mean_val = algo_row.get(f"{metric_key}_mean", np.nan)
            std_val = algo_row.get(f"{metric_key}_std", np.nan)
            values.append(mean_val)
            stds.append(std_val)

        # 找出最优值
        valid_values = [v for v in values if not np.isnan(v)]
        if not valid_values:
            continue

        if pref == "higher":
            best_val = np.nanmax(values)
            best_indices = [i for i, v in enumerate(values) if abs(v - best_val) < 1e-6]
        else:
            best_val = np.nanmin(values)
            best_indices = [i for i, v in enumerate(values) if abs(v - best_val) < 1e-6]

        # 更新Win/Tie/Loss统计
        for i, algo in enumerate(algos):
            if i in best_indices:
                if len(best_indices) == 1:
                    win_counts[algo] += 1
                else:
                    tie_counts[algo] += 1
            else:
                loss_counts[algo] += 1

        # 生成表格行
        row_parts = [metric_name]
        for i, (val, std) in enumerate(zip(values, stds)):
            if np.isnan(val):
                cell = "N/A"
            else:
                cell = f"{fmt % val}"
                if not np.isnan(std):
                    cell += f" $\\pm$ {fmt % std}"

                # 加粗最优值
                if i in best_indices:
                    cell = f"\\textbf{{{cell}}}"

            row_parts.append(cell)

        row = " & ".join(row_parts) + " \\\\"
        latex_lines.append(row)

    latex_lines.append("\\midrule")

    # Win/Tie/Loss统计行
    win_row = "Wins & " + " & ".join([str(win_counts[algo]) for algo in algos]) + " \\\\"
    tie_row = "Ties & " + " & ".join([str(tie_counts[algo]) for algo in algos]) + " \\\\"
    loss_row = "Losses & " + " & ".join([str(loss_counts[algo]) for algo in algos]) + " \\\\"

    latex_lines.append(win_row)
    latex_lines.append(tie_row)
    latex_lines.append(loss_row)

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    # 保存到文件
    out = output_root / f"comparison_table_{stamp}.tex"
    with out.open("w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))

    print(f"Saved: {out}")

    # 同时生成Markdown版本（便于预览）
    md_lines = []
    md_lines.append("# Algorithm Comparison Table\n")
    md_lines.append("| Metric | " + " | ".join(algos) + " |")
    md_lines.append("|" + "---|" * (len(algos) + 1))

    for metric_key, metric_name, pref, fmt in metrics:
        values = []
        stds = []
        for algo_row in algo_summary:
            mean_val = algo_row.get(f"{metric_key}_mean", np.nan)
            std_val = algo_row.get(f"{metric_key}_std", np.nan)
            values.append(mean_val)
            stds.append(std_val)

        valid_values = [v for v in values if not np.isnan(v)]
        if not valid_values:
            continue

        if pref == "higher":
            best_val = np.nanmax(values)
            best_indices = [i for i, v in enumerate(values) if abs(v - best_val) < 1e-6]
        else:
            best_val = np.nanmin(values)
            best_indices = [i for i, v in enumerate(values) if abs(v - best_val) < 1e-6]

        row_parts = [metric_name]
        for i, (val, std) in enumerate(zip(values, stds)):
            if np.isnan(val):
                cell = "N/A"
            else:
                cell = f"{fmt % val}"
                if not np.isnan(std):
                    cell += f" ± {fmt % std}"

                if i in best_indices:
                    cell = f"**{cell}**"

            row_parts.append(cell)

        md_lines.append("| " + " | ".join(row_parts) + " |")

    md_lines.append("| **Wins** | " + " | ".join([f"**{win_counts[algo]}**" for algo in algos]) + " |")
    md_lines.append("| **Ties** | " + " | ".join([f"{tie_counts[algo]}" for algo in algos]) + " |")
    md_lines.append("| **Losses** | " + " | ".join([f"{loss_counts[algo]}" for algo in algos]) + " |")

    out_md = output_root / f"comparison_table_{stamp}.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Saved: {out_md}")
# ======修改结束


def _plot_reward_curves(per_run_rows, output_root: Path, stamp: str):
    algo_to_curves = {}
    for row in per_run_rows:
        run_dir = Path(row.get("run_dir", ""))
        rewards = _load_npy(run_dir / "rewards.npy")
        if rewards is None or rewards.size == 0:
            continue
        algo_to_curves.setdefault(row["algo"], []).append(rewards)

    if not algo_to_curves:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：原始曲线
    for algo, curves in sorted(algo_to_curves.items()):
        for i, curve in enumerate(curves):
            label = f"{algo}" if i == 0 else None
            ax1.plot(range(len(curve)), curve, linewidth=1.5, alpha=0.7, label=label)

    ax1.set_title("Reward Curves (Raw)", fontweight='bold', fontsize=12)
    ax1.set_xlabel("Episode", fontweight='bold')
    ax1.set_ylabel("Reward", fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 右图：平滑曲线（mean±std）
    for algo, curves in sorted(algo_to_curves.items()):
        min_len = min(len(c) for c in curves)
        if min_len <= 1:
            continue
        mat = np.vstack([c[:min_len] for c in curves])
        mean = np.mean(mat, axis=0)
        std = np.std(mat, axis=0)
        x = np.arange(min_len)
        ax2.plot(x, mean, linewidth=2.5, label=algo)
        ax2.fill_between(x, mean - std, mean + std, alpha=0.2)

    ax2.set_title("Reward Curves (Mean±Std)", fontweight='bold', fontsize=12)
    ax2.set_xlabel("Episode", fontweight='bold')
    ax2.set_ylabel("Reward", fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle("Reward Curves Comparison", fontsize=14, fontweight='bold', y=0.98)
    out = output_root / f"comparison_reward_curves_{stamp}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()


# =========2026.3.4新增，样本效率对比图==========
def _plot_sample_efficiency(per_run_rows, output_root: Path, stamp: str):
    """绘制样本效率对比图"""
    algo_to_curves = {}
    for row in per_run_rows:
        run_dir = Path(row.get("run_dir", ""))
        rewards = _load_npy(run_dir / "rewards.npy")
        if rewards is None or rewards.size == 0:
            continue
        algo_to_curves.setdefault(row["algo"], []).append(rewards)

    if not algo_to_curves:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：达到90%/95%性能所需轮数
    algos = sorted(algo_to_curves.keys())
    episodes_90 = []
    episodes_95 = []

    for algo in algos:
        curves = algo_to_curves[algo]
        eps_90_list = [_compute_sample_efficiency(c, 0.9) for c in curves]
        eps_95_list = [_compute_sample_efficiency(c, 0.95) for c in curves]
        episodes_90.append(np.mean(eps_90_list))
        episodes_95.append(np.mean(eps_95_list))

    x = np.arange(len(algos))
    width = 0.35

    bars1 = ax1.bar(x - width/2, episodes_90, width, label='90% Performance', alpha=0.8)
    bars2 = ax1.bar(x + width/2, episodes_95, width, label='95% Performance', alpha=0.8)

    ax1.set_xlabel('Algorithm', fontweight='bold')
    ax1.set_ylabel('Episodes Required', fontweight='bold')
    ax1.set_title('Sample Efficiency: Episodes to Reach Target Performance', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algos)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 标注数值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9)

    # 右图：学习曲线AUC对比
    auc_values = []
    for algo in algos:
        curves = algo_to_curves[algo]
        auc_list = [_compute_auc(c) for c in curves]
        auc_values.append(np.mean(auc_list))

    bars = ax2.bar(algos, auc_values, alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(algos)])
    ax2.set_xlabel('Algorithm', fontweight='bold')
    ax2.set_ylabel('AUC (Normalized)', fontweight='bold')
    ax2.set_title('Learning Curve AUC (Higher is Better)', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 标注数值
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    out = output_root / f"comparison_sample_efficiency_{stamp}.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def _plot_business_metrics(per_run_rows, output_root: Path, stamp: str):
    """绘制业务指标对比图"""
    algo_stats = {}
    for row in per_run_rows:
        algo = row["algo"]
        if algo not in algo_stats:
            algo_stats[algo] = {'svc': [], 'rld': [], 'srd': [], 'reward': []}

        # 收集业务指标
        if 'svc_last10_mean' in row and not np.isnan(row['svc_last10_mean']):
            algo_stats[algo]['svc'].append(row['svc_last10_mean'])
        if 'rld_last10_mean' in row and not np.isnan(row['rld_last10_mean']):
            algo_stats[algo]['rld'].append(row['rld_last10_mean'])
        if 'srd_last10_mean' in row and not np.isnan(row['srd_last10_mean']):
            algo_stats[algo]['srd'].append(row['srd_last10_mean'])
        if 'reward_last50_mean' in row and not np.isnan(row['reward_last50_mean']):
            algo_stats[algo]['reward'].append(row['reward_last50_mean'])

    if not algo_stats:
        return

    algos = sorted(algo_stats.keys())

    # 计算平均值
    metrics = ['Service Rate', 'Relocation Dist.', 'Service Quality', 'Reward (Last 50)']
    metric_keys = ['svc', 'rld', 'srd', 'reward']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric_name, metric_key) in enumerate(zip(metrics, metric_keys)):
        ax = axes[idx]
        values = []
        errors = []

        for algo in algos:
            data = algo_stats[algo][metric_key]
            if data:
                values.append(np.mean(data))
                errors.append(np.std(data))
            else:
                values.append(0)
                errors.append(0)

        x = np.arange(len(algos))
        bars = ax.bar(x, values, yerr=errors, capsize=5, alpha=0.8,
                     color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(algos)])

        ax.set_xlabel('Algorithm', fontweight='bold')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algos)
        ax.grid(True, alpha=0.3, axis='y')

        # 标注最优值
        if values:
            if metric_key == 'rld':  # 调度距离越小越好
                best_idx = np.argmin(values)
            else:  # 其他指标越大越好
                best_idx = np.argmax(values)
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(3)

        # 标注数值
        for i, (val, err) in enumerate(zip(values, errors)):
            if val > 0:
                ax.text(i, val + err, f'{val:.2f}',
                       ha='center', va='bottom', fontsize=9)

    plt.suptitle('Business Metrics Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    out = output_root / f"comparison_business_metrics_{stamp}.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def _generate_statistical_report(per_run_rows, output_root: Path, stamp: str):
    """生成统计显著性检验报告"""
    # 按算法分组收集最终奖励
    algo_rewards = {}
    for row in per_run_rows:
        algo = row["algo"]
        if 'reward_last50_mean' in row and not np.isnan(row['reward_last50_mean']):
            algo_rewards.setdefault(algo, []).append(row['reward_last50_mean'])

    if len(algo_rewards) < 2:
        print("Not enough algorithms for statistical testing")
        return

    algos = sorted(algo_rewards.keys())

    # 生成统计报告
    report_lines = []
    report_lines.append("# Statistical Significance Test Report\n")
    report_lines.append(f"Generated: {stamp}\n")
    report_lines.append("=" * 80 + "\n\n")

    report_lines.append("## Pairwise Comparisons (t-test)\n\n")
    report_lines.append("| Comparison | t-statistic | p-value | Effect Size (Cohen's d) | Significant? |\n")
    report_lines.append("|------------|-------------|---------|-------------------------|-------------|\n")

    # 两两比较
    for i in range(len(algos)):
        for j in range(i+1, len(algos)):
            algo1, algo2 = algos[i], algos[j]
            group1 = np.array(algo_rewards[algo1])
            group2 = np.array(algo_rewards[algo2])

            result = _statistical_test(group1, group2)

            sig_mark = "✓ Yes" if result['significant'] else "✗ No"
            report_lines.append(
                f"| {algo1} vs {algo2} | "
                f"{result['t_stat']:.4f} | "
                f"{result['p_value']:.4f} | "
                f"{result['effect_size']:.4f} | "
                f"{sig_mark} |\n"
            )

    report_lines.append("\n## Interpretation\n\n")
    report_lines.append("- **p-value < 0.05**: Statistically significant difference\n")
    report_lines.append("- **Effect Size (Cohen's d)**:\n")
    report_lines.append("  - Small: 0.2\n")
    report_lines.append("  - Medium: 0.5\n")
    report_lines.append("  - Large: 0.8+\n\n")

    report_lines.append("## Summary Statistics\n\n")
    report_lines.append("| Algorithm | Mean | Std | Min | Max | N |\n")
    report_lines.append("|-----------|------|-----|-----|-----|---|\n")

    for algo in algos:
        data = np.array(algo_rewards[algo])
        report_lines.append(
            f"| {algo} | "
            f"{np.mean(data):.2f} | "
            f"{np.std(data):.2f} | "
            f"{np.min(data):.2f} | "
            f"{np.max(data):.2f} | "
            f"{len(data)} |\n"
        )

    # 保存报告
    out = output_root / f"statistical_report_{stamp}.md"
    with open(out, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)

    print(f"Saved: {out}")
# ======修改结束


def _run_single_algo(workdir: Path, output_root: Path, algo: str, seed: int, extra_args):
    # 从 extra_args 中提取训练轮数
    train_eps = 700  # 默认值
    for i, arg in enumerate(extra_args):
        if arg == '--train_eps' and i + 1 < len(extra_args):
            try:
                train_eps = int(extra_args[i + 1])
            except ValueError:
                pass
            break

    run_dir = output_root / f"single_{algo.lower()}_{train_eps}ep_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "main.py", "--algo_name", algo, "--seed", str(seed)] + extra_args
    print(f"[RUN] {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=workdir)
    elapsed = time.time() - t0

    meta = {
        "runner": "main.py",
        "algo": algo,
        "seed": seed,
        "train_eps": train_eps,
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    for p in _collect_root_artifacts(workdir, t0):
        shutil.copy2(p, run_dir / p.name)

    if proc.returncode != 0:
        return None
    row = _summarize_run(run_dir, algo, seed)
    row["run_dir"] = str(run_dir)
    return row


def _run_marl(workdir: Path, output_root: Path, label: str, seed: int, extra_args):
    # 从 extra_args 中提取训练轮数
    train_eps = 700  # 默认值
    for i, arg in enumerate(extra_args):
        if arg == '--train_eps' and i + 1 < len(extra_args):
            try:
                train_eps = int(extra_args[i + 1])
            except ValueError:
                pass
            break

    run_dir = output_root / f"marl_{label.lower()}_{train_eps}ep_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "main_marl.py", "--seed", str(seed)] + extra_args
    print(f"[RUN] {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=workdir)
    elapsed = time.time() - t0

    meta = {
        "runner": "main_marl.py",
        "algo": label,
        "seed": seed,
        "train_eps": train_eps,
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    latest_marl = _find_latest_marl_output(workdir, t0)
    if latest_marl is not None:
        dst = run_dir / "marl_outputs"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(latest_marl, dst)
        for p in dst.iterdir():
            if p.is_file() and p.suffix.lower() in {".npy", ".png", ".pth"}:
                shutil.copy2(p, run_dir / p.name)

    for p in _collect_root_artifacts(workdir, t0):
        shutil.copy2(p, run_dir / p.name)

    if proc.returncode != 0:
        return None
    row = _summarize_run(run_dir, label, seed)
    row["run_dir"] = str(run_dir)
    return row


def main():
    parser = argparse.ArgumentParser(description="Unified baseline runner: single-agent + MARL")
    # =========2026.3.4修改，移除DDPG算法==========
    # parser.add_argument("--single_algos", default="DDPG,SAC,TD3", help="Comma-separated single-agent algos")
    parser.add_argument("--single_algos", default="SAC,TD3", help="Comma-separated single-agent algos (DDPG removed)")
    # ======修改结束
    parser.add_argument("--include_marl", default="true", help="true/false")
    parser.add_argument("--marl_label", default="MARL", help="Label used in summary for MARL")
    # =========2026.3.4修改，默认只运行seed 7==========
    # parser.add_argument("--seeds", default="2,7,11", help="Comma-separated seed list")
    parser.add_argument("--seeds", default="7", help="Comma-separated seed list (default: only seed 7)")
    # ======修改结束
    parser.add_argument("--output_root", default="outputs/compare_baseline2", help="Output directory")
    parser.add_argument("--single_extra_args", default="", help="Extra args for main.py, comma-separated")
    parser.add_argument("--marl_extra_args", default="", help="Extra args for main_marl.py, comma-separated")
    args = parser.parse_args()

    workdir = Path(__file__).resolve().parent

    # =========2026.3.4修改，每次运行创建带时间戳的新文件夹==========
    # 生成时间戳
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # 创建带时间戳的输出目录
    output_root = (workdir / args.output_root / f"run_{timestamp}").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_root}")
    # ======修改结束

    include_marl = args.include_marl.strip().lower() in {"1", "true", "yes", "y"}
    single_algos = _parse_list(args.single_algos)
    seeds = [int(x) for x in _parse_list(args.seeds)]
    single_extra_args = _parse_list(args.single_extra_args)
    marl_extra_args = _parse_list(args.marl_extra_args)

    per_run_rows = []

    for algo in single_algos:
        for seed in seeds:
            row = _run_single_algo(workdir, output_root, algo, seed, single_extra_args)
            if row is not None:
                per_run_rows.append(row)

    if include_marl:
        for seed in seeds:
            row = _run_marl(workdir, output_root, args.marl_label, seed, marl_extra_args)
            if row is not None:
                per_run_rows.append(row)

    if not per_run_rows:
        print("No successful runs. Nothing to summarize.")
        return

    stamp = time.strftime("%Y%m%d-%H%M%S")
    per_run_csv = output_root / f"per_run_summary_{stamp}.csv"
    with per_run_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_run_rows)

    metrics = [
        "final_reward",
        "best_reward",
        "reward_last50_mean",
        "eval_last",
        "eval_mean",
        "marl_test_mean",
        "svc_last10_mean",
        "rld_last10_mean",
        "srd_last10_mean",
        "critic1_last100_mean",
        "critic_last100_mean",
        # =========2026.3.3修改，添加新指标==========
        "reward_variance",
        "convergence_episode",
        "best_final_gap",
        "reward_std",
        # ======修改结束
    ]

    algo_summary = []
    for algo in sorted(set(r["algo"] for r in per_run_rows)):
        rows = [r for r in per_run_rows if r["algo"] == algo]
        row = {"algo": algo, "runs": len(rows)}
        for m in metrics:
            mean_v, std_v = _aggregate(rows, m)
            row[f"{m}_mean"] = mean_v
            row[f"{m}_std"] = std_v
        algo_summary.append(row)

    summary_csv = output_root / f"algo_summary_{stamp}.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(algo_summary[0].keys()))
        writer.writeheader()
        writer.writerows(algo_summary)

    _plot_summary_bars(algo_summary, output_root, stamp)
    _plot_reward_curves(per_run_rows, output_root, stamp)

    # =========2026.3.3修改，调用新的对比图函数==========
    _plot_extended_comparison(algo_summary, output_root, stamp)
    _plot_radar_chart(algo_summary, output_root, stamp)
    _plot_convergence_analysis(per_run_rows, output_root, stamp)
    _generate_latex_table(algo_summary, output_root, stamp)
    # ======修改结束

    # =========2026.3.4新增，调用新的评估指标函数==========
    print("\nGenerating advanced analysis...")
    _plot_sample_efficiency(per_run_rows, output_root, stamp)
    _plot_business_metrics(per_run_rows, output_root, stamp)
    _generate_statistical_report(per_run_rows, output_root, stamp)
    # ======修改结束

    # =========2026.3.4新增，充电调度专项分析==========
    print("\nGenerating charging scheduling analysis...")
    plot_charging_comprehensive(per_run_rows, output_root, stamp)
    plot_charging_efficiency_radar(per_run_rows, output_root, stamp)
    plot_charging_dashboard(per_run_rows, output_root, stamp)  # 新增：充电仪表盘

    # MARL充电策略单独分析
    marl_rows = [r for r in per_run_rows if 'MARL' in r.get('algo', '').upper()]
    if marl_rows:
        print("\nGenerating MARL-specific charging analysis...")
        plot_marl_charging_strategy(marl_rows, output_root, stamp)
        plot_marl_charging_comparison(marl_rows, output_root, stamp)
    # ======修改结束

    print(f"Saved: {per_run_csv}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
