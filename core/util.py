# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 19:29:03 2024

@author: TLab
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# =============================================================================
# Reviewer experiment control panel
# Only edit these three values when launching experiments from PyCharm.
# Command-line values still override the selected preset when needed.
# =============================================================================
ACTIVE_EXPERIMENT = "E2"
ACTIVE_VARIANT = "no_adaptive_exploration"
ACTIVE_SEED = 1

RECOMMENDED_SEEDS = (1, 11, 21, 31, 41)

# 实验编号中文说明（启动时会随参数一起打印）
REVIEW_EXPERIMENT_NAMES = {
    "E1": "Zhu 2025 方法对比",
    "E2": "模块消融实验",
    "E3": "探索策略对比",
    "E4": "多测试日、多随机种子实验（25天训练+5天测试）",
    "E5": "车队规模扩展（200/1000/2000辆）",
    "E6": "计算效率实验",
    "E7": "Charging 效果验证",
}

# 每个子配置的中文说明
REVIEW_VARIANT_NAMES = {
    ("E1", "hmarl_rc"): "本文 HMARL-RC 方法",
    ("E1", "zhu2025"): "Zhu 2025 方法（当前尚未实现，禁止伪标签运行）",
    ("E2", "full"): "完整模型",
    ("E2", "no_zone_embedding"): "去掉区域嵌入 Zone Embedding",
    ("E2", "no_global_context"): "去掉全局上下文 Global Context",
    ("E2", "no_parameter_sharing"): "去掉参数共享 Parameter Sharing",
    ("E2", "no_adaptive_exploration"): "去掉自适应探索，改用线性衰减",
    ("E3", "adaptive"): "服务率驱动的自适应探索",
    ("E3", "fixed"): "固定探索噪声",
    ("E3", "linear"): "线性衰减探索噪声",
    ("E4", "five_days"): "训练日1-25，测试日26-30，五个随机种子",
    ("E5", "fleet_200"): "200辆车规模",
    ("E5", "fleet_1000"): "1000辆车规模",
    ("E5", "fleet_2000"): "2000辆车规模",
    ("E6", "efficiency"): "Actor、环境、总决策及P0/P1求解效率",
    ("E7", "optimized"): "改进后的P1充电优化",
    ("E7", "original_rule"): "原始规则式充电",
    ("E7", "no_charging"): "完全关闭充电",
}

_FULL_REVIEW_CONFIG = {
    "train_day_indices": list(range(1, 26)),
    "test_day_indices": list(range(26, 31)),
    "fleet_size": 200,
    "use_zone_embedding": True,
    "use_global_context": True,
    "share_policy": True,
    "exploration_schedule": "adaptive",
    "charging_assignment_mode": "p1",
    "charger_ratio": 0.576,
    "reward_mode": "shaping",  # 与历史主方法 e9d67bc 保持一致，包含充电与SOC塑形奖励
    "exp_variant": "a1",
    "record_runtime_metrics": True,
    "method": "hmarl-rc",
    "evaluation_experiment": "E2-module-ablation",
}


def _preset(**overrides):
    value = dict(_FULL_REVIEW_CONFIG)
    value.update(overrides)
    return value


REVIEW_EXPERIMENT_PRESETS = {
    # E1：Zhu 2025 方法对比
    "E1": {
        "hmarl_rc": _preset(
            method="hmarl-rc", evaluation_experiment="E1-zhu-comparison"
        ),
        # Zhu 2025 is intentionally represented as unavailable below. A label
        # change must never be mistaken for an implementation of that method.
    },
    # E2：分别去掉 Zone Embedding、Global Context、Parameter Sharing、
    #     Adaptive Exploration 的模块消融实验
    "E2": {
        "full": _preset(method="hmarl-rc", evaluation_experiment="E2-module-ablation"),
        "no_zone_embedding": _preset(
            use_zone_embedding=False,
            method="hmarl-no-zone",
            evaluation_experiment="E2-module-ablation",
        ),
        "no_global_context": _preset(
            use_global_context=False,
            method="hmarl-no-global",
            evaluation_experiment="E2-module-ablation",
        ),
        "no_parameter_sharing": _preset(
            share_policy=False,
            method="hmarl-no-sharing",
            evaluation_experiment="E2-module-ablation",
        ),
        "no_adaptive_exploration": _preset(
            exploration_schedule="linear",
            method="hmarl-linear-exploration",
            evaluation_experiment="E2-module-ablation",
        ),
    },
    # E3：Adaptive、Fixed Noise、Linear Decay 探索策略对比
    "E3": {
        schedule: _preset(
            exploration_schedule=schedule,
            method=f"hmarl-exploration-{schedule}",
            evaluation_experiment="E3-exploration-comparison",
        )
        for schedule in ("adaptive", "fixed", "linear")
    },
    # E4：多测试日、多随机种子；固定训练日1-25、测试日26-30
    "E4": {
        "five_days": _preset(
            train_day_indices=list(range(1, 26)),
            test_day_indices=list(range(26, 31)),
            method="hmarl-rc-five-test-days",
            evaluation_experiment="E4-multiday-multiseed",
        ),
    },
    # E5：车队规模扩展；三个规模均使用25天训练、5天测试
    "E5": {
        # Per user decision: use days 1-25 for training and days 26-30 for testing.
        f"fleet_{size}": _preset(
            fleet_size=size,
            method=f"hmarl-rc-fleet-{size}",
            evaluation_experiment="E5-fleet-scaling-25train-5test",
        )
        for size in (200, 1000, 2000)
    },
    # E6：记录 Actor 推理、环境执行、总决策时间以及 P0/P1 求解质量
    "E6": {
        "efficiency": _preset(
            record_runtime_metrics=True,
            method="hmarl-rc-efficiency",
            evaluation_experiment="E6-computational-efficiency",
        ),
    },
    # E7：P1优化充电、原规则充电、完全无充电三组对照
    "E7": {
        "optimized": _preset(
            charging_assignment_mode="p1",
            method="hmarl-charge-p1",
            evaluation_experiment="E7-charging-effect",
        ),
        "original_rule": _preset(
            charging_assignment_mode="legacy",
            method="hmarl-charge-legacy",
            evaluation_experiment="E7-charging-effect",
        ),
        "no_charging": _preset(
            charging_assignment_mode="none",
            method="hmarl-no-charging",
            evaluation_experiment="E7-charging-effect",
        ),
    },
}


def resolve_experiment_preset(experiment_id, variant):
    """Return one reviewer preset, failing loudly for unavailable methods."""
    experiment_id = str(experiment_id).strip().upper()
    variant = str(variant).strip().lower()
    if experiment_id == "E1" and variant == "zhu2025":
        raise NotImplementedError(
            "Zhu 2025 is not implemented in this repository. Obtain or implement "
            "the real method before running E1; changing --method would only relabel HMARL."
        )
    variants = REVIEW_EXPERIMENT_PRESETS.get(experiment_id)
    if variants is None:
        raise ValueError(
            f"unknown reviewer experiment {experiment_id!r}; "
            f"choose one of {sorted(REVIEW_EXPERIMENT_PRESETS)}"
        )
    if variant not in variants:
        raise ValueError(
            f"unknown variant {variant!r} for {experiment_id}; "
            f"choose one of {sorted(variants)}"
        )
    return dict(variants[variant])


def str_to_bool(value):
    """Parse explicit CLI booleans without bool("false") surprises."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def get_args():
    """解析训练、测试及算法相关超参数。"""
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--experiment_id', default=ACTIVE_EXPERIMENT)
    selector.add_argument('--experiment_variant', default=ACTIVE_VARIANT)
    selected, _ = selector.parse_known_args()
    preset = resolve_experiment_preset(selected.experiment_id, selected.experiment_variant)

    parser = argparse.ArgumentParser(description="hyperparameters")

    parser.add_argument('--experiment_id', default=str(selected.experiment_id).upper(), type=str,
                        help="reviewer experiment group E1-E7")
    parser.add_argument('--experiment_variant', default=str(selected.experiment_variant).lower(), type=str,
                        help="variant name inside the selected reviewer experiment")

    parser.add_argument('--algo_name', default='MARL', type=str, help="name of algorithm")  # 主方法标识；main_marl固定使用多智能体实现
    parser.add_argument('--env_name', default='dclily', type=str, help="name of environment")  # 环境名称
    parser.add_argument('--data_folder', default='data\\SIP\\', type=str, help="name of environment")  # SIP数据路径

    # 与 results_v1/marl_20260222-230455- 主方法保持一致：正式训练 1200 轮。
    parser.add_argument('--train_eps', default=1000, type=int, help="episodes of training")  # 训练轮数
    # ====== 修改结束

    parser.add_argument('--test_eps', default=10, type=int, help="episodes of testing")  # 测试轮数
    parser.add_argument('--max_steps', default=26, type=int, help="steps per episode")  # 每轮最大步数，与 TIME_NUM 对齐，避免无效超长仿真

    # 与 results_v1 主方法 checkpoint 及对应 Git 提交 e9d67bc 对齐。
    parser.add_argument('--gamma', default=0.95, type=float, help="discounted factor")
    parser.add_argument('--critic_lr', default=1e-4, type=float, help="learning rate of critic")
    # ====== 修改结束

    parser.add_argument('--actor_lr', default=5e-5, type=float, help="learning rate of actor")
    parser.add_argument('--memory_capacity', default=64000, type=int, help="memory capacity")

    parser.add_argument('--batch_size', default=128, type=int)
    # ====== 修改结束

    parser.add_argument('--target_update', default=2, type=int)  # 目标网络更新频率

    parser.add_argument('--tau', default=5e-4, type=float)
    # ====== 修改结束

    parser.add_argument('--critic_hidden_dim', default=256, type=int)  # critic的dense层神经元，实验18恢复为256
    parser.add_argument('--actor_hidden_dim', default=256, type=int)  # actor的dense层神经元，实验18恢复为256
    parser.add_argument('--critic_embed_dim', default=8, type=int)  # critic的embedding层维度，实验18恢复为8
    parser.add_argument('--actor_embed_dim', default=8, type=int)  # actor的embedding层维度，实验18恢复为8

    parser.add_argument('--sac_alpha', default=0.2, type=float, help="SAC initial temperature")  # SAC温度参数
    parser.add_argument('--sac_alpha_lr', default=1e-4, type=float, help="SAC alpha learning rate")  # SAC alpha学习率
    parser.add_argument('--reward_scale', default=10000.0, type=float, help="reward scaling factor for SAC/DDPG updates")  # 2026.2.20：提高到10000以稳定Q值
    parser.add_argument('--eval_every', default=50, type=int, help="run deterministic eval every N episodes during training")  # 2026.3.4修改：从20改为50，减少评估次数
    parser.add_argument('--eval_eps', default=3, type=int, help="number of episodes per periodic eval")
    parser.add_argument('--exp_variant', default=preset['exp_variant'], type=str, help="legacy action-decoding variant used by env")
    parser.add_argument('--reward_mode', default=preset['reward_mode'], type=str, help="reward mode: base or shaping")  # 奖励模式：base为基础奖励，shaping为奖励塑形模式
    parser.add_argument('--use_dual_head', default=False, type=str_to_bool, help="Use dual-head architecture for SAC")  # 是否使用双头架构，实验19禁用双头
    parser.add_argument('--device', default='cpu', type=str, help="cpu or cuda")  # 运行设备，2026.3.3修改：启用GPU加速
    parser.add_argument('--seed', default=ACTIVE_SEED, type=int, help="random seed")  # 随机种子

    # ========= 2026.2.19 修改：MARL Phase-1，添加多智能体参数 =========
    parser.add_argument('--marl_mode', default='single', type=str, help="single or multi agent mode")  # 单智能体/多智能体模式
    parser.add_argument('--n_agents', default=50, type=int, help="number of agents (zones)")  # 智能体数量，对应区域数量
    parser.add_argument('--share_policy', default=preset['share_policy'], type=str_to_bool, help="whether to share policy across agents")
    parser.add_argument('--use_zone_embedding', default=preset['use_zone_embedding'], type=str_to_bool, help="include zone identity embedding in actor")
    parser.add_argument('--use_global_context', default=preset['use_global_context'], type=str_to_bool, help="include global context in actor")  # 是否在多智能体之间共享策略参数
    parser.add_argument('--policy_delay', default=6, type=int, help="TD3 delayed policy update frequency")  # TD3延迟策略更新频率
    parser.add_argument('--target_policy_noise', default=0.02, type=float, help="TD3 target policy smoothing noise std")  # TD3目标策略平滑噪声标准差
    parser.add_argument('--target_noise_clip', default=0.05, type=float, help="TD3 target policy noise clip")  # TD3目标策略噪声裁剪范围
    parser.add_argument('--expl_noise', default=0.02, type=float, help="TD3 exploration noise std")  # TD3探索噪声标准差
    parser.add_argument('--battery_log_keep_steps', default=2000, type=int, help="max timesteps kept for battery heatmap during training")  # 训练过程中电量热力图最多保留的时间步数
    parser.add_argument('--battery_log_stride', default=1, type=int, help="record one battery snapshot every N env steps")  # 每隔N个环境步记录一次车辆电量快照
    parser.add_argument('--marl_metric_sample_every', default=1, type=int, help="sample interval for MARL per-step metrics")
    parser.add_argument('--progress_every', default=50, type=int, help="print training progress every N episodes")  # MARL逐步指标采样间隔

    # ========= 2026.3.27 修改：进一步压缩 MARL 探索噪声，修复 train/eval gap 约22%的问题 =========
    parser.add_argument('--marl_noise_start', default=0.12, type=float, help="initial exploration noise scale for MARL")  # 初始探索噪声，50智能体叠加效应较强，因此需要降低
    parser.add_argument('--marl_noise_min', default=0.02, type=float, help="minimum exploration noise scale for MARL")  # 最小探索噪声，后期更接近确定性策略
    parser.add_argument('--marl_noise_decay_power', default=2.0, type=float, help="decay curvature for MARL exploration noise")  # 探索噪声衰减曲率，用于控制平滑衰减过程
    # ====== 修改结束

    parser.add_argument('--marl_noise_warmup_eps', default=30, type=int, help="warmup episodes before MARL noise decay")
    parser.add_argument('--exploration_schedule', default=preset['exploration_schedule'], choices=['adaptive', 'linear', 'exponential', 'fixed'], help="MARL exploration schedule")
    parser.add_argument('--marl_service_ema_alpha', default=0.2, type=float, help="EMA update rate for service-driven exploration")  # 探索噪声衰减前的预热轮数
    parser.add_argument('--marl_service_target', default=0.85, type=float, help="target svc used for MARL adaptive exploration")  # MARL自适应探索的服务率目标
    parser.add_argument('--marl_noise_service_boost', default=0.05, type=float, help="extra noise when svc is below target")  # 当服务率低于目标值时额外增加的探索噪声
    parser.add_argument('--marl_updates_per_step', default=2, type=int, help="number of gradient updates per env step for MARL")  # MARL每个环境步对应的梯度更新次数
    parser.add_argument('--marl_learning_starts', default=2048, type=int, help="minimum replay size before MARL updates")  # 开始MARL训练更新前所需的最小经验池样本量
    parser.add_argument('--marl_policy_delay', default=2, type=int, help="delayed actor update frequency for MARL twin-critic")  # MARL双Critic结构下的延迟Actor更新频率
    parser.add_argument('--marl_target_policy_noise', default=0.08, type=float, help="target action smoothing noise std for MARL")  # MARL目标动作平滑噪声标准差
    parser.add_argument('--marl_target_noise_clip', default=0.15, type=float, help="target action smoothing noise clip")  # MARL目标动作平滑噪声裁剪范围
    parser.add_argument('--same_day_train_test', default=False, type=str_to_bool, help="if True, both train and test use the same OD training day")  # 若为True，则训练和测试使用同一天OD数据
    parser.add_argument('--fixed_day_index', default=1, type=int, help="1-based training day index used when same_day_train_test=True")
    parser.add_argument('--train_day_indices', nargs='+', default=preset['train_day_indices'], type=int, help="SIP training days")
    parser.add_argument('--test_day_indices', nargs='+', default=preset['test_day_indices'], type=int, help="held-out SIP test days")
    parser.add_argument('--fleet_size', default=preset['fleet_size'], choices=[200, 1000, 2000], type=int, help="fleet size for E5")
    parser.add_argument('--charger_ratio', default=preset['charger_ratio'], type=float, help="charger-to-parking ratio for sensitivity experiments")
    parser.add_argument('--charging_assignment_mode', default=preset['charging_assignment_mode'], choices=['p1', 'legacy', 'none'], help="charging assignment implementation")
    parser.add_argument('--record_runtime_metrics', default=preset['record_runtime_metrics'], type=str_to_bool, help="record actor/environment/total decision timing")
    parser.add_argument('--method', default=preset['method'], type=str, help="method label written to evaluation outputs")
    parser.add_argument('--evaluation_experiment', default=preset['evaluation_experiment'], type=str, help="evaluation output group")  # 当same_day_train_test=True时使用的训练日编号，从1开始
    # ====== 修改结束

    # ========= 2026.2.13 修改：修复命令行参数解析 =========
    # 原代码：args = parser.parse_args([])  # 传入空列表会忽略命令行参数
    # 修复：不传参数，让argparse自动读取sys.argv
    args = parser.parse_args()  # 正确读取命令行参数
    # ====== 修改结束

    args = {**vars(args)}  # 将args转换为字典
    args['experiment_id'] = str(args['experiment_id']).upper()
    args['experiment_variant'] = str(args['experiment_variant']).lower()
    args['experiment_name'] = REVIEW_EXPERIMENT_NAMES[args['experiment_id']]
    args['experiment_variant_name'] = REVIEW_VARIANT_NAMES[
        (args['experiment_id'], args['experiment_variant'])
    ]
    train_days = [int(day) for day in args['train_day_indices']]
    test_days = [int(day) for day in args['test_day_indices']]
    if not train_days or not test_days:
        raise ValueError("train_day_indices and test_day_indices cannot be empty")
    if len(train_days) != len(set(train_days)) or len(test_days) != len(set(test_days)):
        raise ValueError("training and test day lists must not contain duplicates")
    overlap = sorted(set(train_days).intersection(test_days))
    if overlap:
        raise ValueError(f"training and test days overlap: {overlap}")
    args['train_day_indices'] = train_days
    args['test_day_indices'] = test_days

    # 打印参数
    print("Training arguments:")
    print(''.join(['='] * 80))
    tplt = "{:^20}\t{:^20}\t{:^20}"
    print(tplt.format("param", "value", "type"))
    for k, v in args.items():
        print(tplt.format(str(k), str(v), str(type(v))))
    print(''.join(['='] * 80))
    return args


def smooth(data, weight=0.9):
    """对曲线数据进行指数平滑处理，效果类似 TensorBoard 中的 smooth。

    Args:
        data (List or np.ndarray): 输入数据序列。
        weight (float): 平滑权重，取值范围通常为0-1；数值越大，曲线越平滑。

    Returns:
        List: 平滑后的数据序列。
    """
    last = data[0]  # 曲线中的第一个值
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算当前平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_rewards(rewards, cfg, path=None, tag='train'):
    """绘制训练或测试奖励曲线，并叠加平滑后的奖励曲线。"""
    sns.set()
    colors = sns.color_palette("muted")

    plt.figure(facecolor='white')  # 创建图形，并设置画布背景为白色
    # plt.savefig(path, facecolor='white')  # 若需要保存图片，应保证 path 非 None
    ax = plt.gca()
    ax.set_facecolor('white')  # 设置绘图区域背景为白色
    plt.grid(False)  # 关闭背景网格线

    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    rewards_shifted = np.array(rewards)  # 0728修改；1128修改：去掉 +6000 的平移处理

    plt.title(f"{tag}ing curve on {cfg['device']} of {cfg['algo_name']} ")
    plt.xlabel('epsiodes')
    plt.plot(rewards_shifted, label='rewards', linewidth=2.0, color=colors[0], alpha=0.15)
    plt.plot(smooth(rewards_shifted), label='smoothed', linewidth=2.0, color=colors[0])  # 绘制平滑后的奖励曲线
    plt.legend(frameon=False, fontsize=10, loc='best')
    # plt.show()  # 若在外部统一显示图像，此处保持注释


def plot_vehicle_battery_heatmap(battery_log, title="Battery Heatmap", save_path=None):
    """绘制车辆电量随时间变化的热力图。"""
    # battery_log = np.array(battery_log)  # 如有需要，可转换为矩阵格式
    plt.figure(figsize=(12, 12))  # 可根据车辆数量调整图像尺寸

    # 使用 seaborn 绘制热力图
    sns.heatmap(
        battery_log,
        cmap='Blues',  # 蓝色渐变色图
        cbar_kws={'label': 'Battery Level'},  # 颜色条标签
        xticklabels=True,
        yticklabels=True
    )

    plt.title(title, fontsize=14)
    plt.xlabel('Vehicle ID', fontsize=12)
    plt.ylabel('Timestep', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    # else:
    #     plt.show()


def plot_vehicle_battery(battery_log, save_path=None):
    """
    绘制所有车辆的电量变化曲线。

    Args:
        battery_log (list or np.ndarray): 车辆电量记录，shape=(T, N)，其中T为时间步数，N为车辆数量。
        save_path (str, optional): 图片保存路径；若提供该路径，则保存图像。
    """
    sns.set()
    colors = sns.color_palette("muted")

    plt.figure(facecolor='white')  # 创建图形，并设置画布背景为白色
    # plt.savefig(path, facecolor='white')  # 若需要保存图片，应保证 path 非 None
    ax = plt.gca()
    ax.set_facecolor('white')  # 设置绘图区域背景为白色
    plt.grid(False)  # 关闭背景网格线

    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    # sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    num_vehicles = battery_log.shape[1]
    for i in range(num_vehicles):
        plt.plot(battery_log[:, i], label=f'Vehicle {i}', linewidth=1.2)

    plt.title("Battery Level of Vehicles Over Time", fontsize=14)
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Battery Level", fontsize=12)
    plt.legend(fontsize=8, ncol=2, frameon=False, loc='upper right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    # else:
    #     plt.show()


def plot_critic_losses(critic_losses, critic2_losses=None, tag="train", show=False, save_path=None):
    """
    绘制 Critic loss 曲线。

    功能：
        - DDPG 通常绘制一条 critic loss 曲线；
        - TD3 若使用双 Critic，则可同时绘制 critic loss 和 critic2 loss；
        - show=False 表示不在函数内部阻塞显示图像，可由 main.py 统一调用 plt.show()；
        - save_path 可选，若提供路径则直接保存图片，便于实验汇报或结果留存。
    """
    if critic_losses is None or len(critic_losses) == 0:
        print(f"[plot_critic_losses] {tag}: empty losses, skip.")
        return

    plt.figure()
    plt.plot(np.asarray(critic_losses, dtype=float), label="critic loss")

    # TD3双Critic结构下可绘制第二条loss曲线
    if critic2_losses is not None and len(critic2_losses) > 0:
        plt.plot(np.asarray(critic2_losses, dtype=float), label="critic2 loss")

    plt.xlabel("update step")
    plt.ylabel("loss")
    plt.title(f"Critic Loss ({tag})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    # if show:
    #     plt.show()


import json
import os
import shutil


def cp(source_path):
    """将指定目录下的 config.json 复制到当前工作目录。"""
    target_path = 'config.json'  # 目标路径为当前文件夹下的config.json
    shutil.copy2(source_path + "config.json", target_path)
