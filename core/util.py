# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 19:29:03 2024

@author: TLab
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def get_args():
    """解析训练、测试及算法相关超参数。"""
    parser = argparse.ArgumentParser(description="hyperparameters")

    parser.add_argument('--algo_name', default='TD3', type=str, help="name of algorithm")  # 算法名称
    parser.add_argument('--env_name', default='dclily', type=str, help="name of environment")  # 环境名称
    parser.add_argument('--data_folder', default='data\\toy_network\\', type=str, help="name of environment")  # 数据路径

    # ========= 2026.3.27 修改：调整训练轮数 =========
    parser.add_argument('--train_eps', default=10, type=int, help="episodes of training")  # 训练轮数
    # ====== 修改结束

    parser.add_argument('--test_eps', default=10, type=int, help="episodes of testing")  # 测试轮数
    parser.add_argument('--max_steps', default=26, type=int, help="steps per episode")  # 每轮最大步数，与 TIME_NUM 对齐，避免无效超长仿真

    # ========= 2026.3.27 修改：MARL_v4 调参，修复 Critic 发散并降低噪声 =========
    parser.add_argument('--gamma', default=0.90, type=float, help="discounted factor")  # TD3保守配置：降低远期回报权重，使学习更平滑且整体更弱
    parser.add_argument('--critic_lr', default=1e-5, type=float, help="learning rate of critic")  # TD3保守配置：减小critic步长，降低训练波动
    # ====== 修改结束

    parser.add_argument('--actor_lr', default=1e-5, type=float, help="learning rate of actor")  # TD3保守配置：减小actor步长，避免过快冲高
    parser.add_argument('--memory_capacity', default=128000, type=int, help="memory capacity")  # 经验池容量，MARL优化默认值

    # ========= 2026.3.27 修改：提升 batch_size，增强训练稳定性 =========
    parser.add_argument('--batch_size', default=1024, type=int)  # TD3保守配置：增大batch以平滑更新
    # ====== 修改结束

    parser.add_argument('--target_update', default=2, type=int)  # 目标网络更新频率

    # ========= 2026.3.27 修改：降低 tau，减缓目标网络跟随速度 =========
    parser.add_argument('--tau', default=1e-4, type=float)  # TD3保守配置：更慢的target更新，进一步降低波动
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
    parser.add_argument('--exp_variant', default='a1', type=str, help="experiment variant key used by env")
    parser.add_argument('--reward_mode', default='base', type=str, help="reward mode: base or shaping")  # 奖励模式：base为基础奖励，shaping为奖励塑形模式
    parser.add_argument('--use_dual_head', default=False, type=bool, help="Use dual-head architecture for SAC")  # 是否使用双头架构，实验19禁用双头
    parser.add_argument('--device', default='cuda', type=str, help="cpu or cuda")  # 运行设备，2026.3.3修改：启用GPU加速
    parser.add_argument('--seed', default=2, type=int, help="random seed")  # 随机种子

    # ========= 2026.2.19 修改：MARL Phase-1，添加多智能体参数 =========
    parser.add_argument('--marl_mode', default='single', type=str, help="single or multi agent mode")  # 单智能体/多智能体模式
    parser.add_argument('--n_agents', default=50, type=int, help="number of agents (zones)")  # 智能体数量，对应区域数量
    parser.add_argument('--share_policy', default=True, type=bool, help="whether to share policy across agents")  # 是否在多智能体之间共享策略参数
    parser.add_argument('--policy_delay', default=6, type=int, help="TD3 delayed policy update frequency")  # TD3延迟策略更新频率
    parser.add_argument('--target_policy_noise', default=0.02, type=float, help="TD3 target policy smoothing noise std")  # TD3目标策略平滑噪声标准差
    parser.add_argument('--target_noise_clip', default=0.05, type=float, help="TD3 target policy noise clip")  # TD3目标策略噪声裁剪范围
    parser.add_argument('--expl_noise', default=0.02, type=float, help="TD3 exploration noise std")  # TD3探索噪声标准差
    parser.add_argument('--battery_log_keep_steps', default=2000, type=int, help="max timesteps kept for battery heatmap during training")  # 训练过程中电量热力图最多保留的时间步数
    parser.add_argument('--battery_log_stride', default=1, type=int, help="record one battery snapshot every N env steps")  # 每隔N个环境步记录一次车辆电量快照
    parser.add_argument('--marl_metric_sample_every', default=1, type=int, help="sample interval for MARL per-step metrics")  # MARL逐步指标采样间隔

    # ========= 2026.3.27 修改：进一步压缩 MARL 探索噪声，修复 train/eval gap 约22%的问题 =========
    parser.add_argument('--marl_noise_start', default=0.12, type=float, help="initial exploration noise scale for MARL")  # 初始探索噪声，50智能体叠加效应较强，因此需要降低
    parser.add_argument('--marl_noise_min', default=0.02, type=float, help="minimum exploration noise scale for MARL")  # 最小探索噪声，后期更接近确定性策略
    parser.add_argument('--marl_noise_decay_power', default=2.0, type=float, help="decay curvature for MARL exploration noise")  # 探索噪声衰减曲率，用于控制平滑衰减过程
    # ====== 修改结束

    parser.add_argument('--marl_noise_warmup_eps', default=30, type=int, help="warmup episodes before MARL noise decay")  # 探索噪声衰减前的预热轮数
    parser.add_argument('--marl_service_target', default=0.85, type=float, help="target svc used for MARL adaptive exploration")  # MARL自适应探索的服务率目标
    parser.add_argument('--marl_noise_service_boost', default=0.05, type=float, help="extra noise when svc is below target")  # 当服务率低于目标值时额外增加的探索噪声
    parser.add_argument('--marl_updates_per_step', default=4, type=int, help="number of gradient updates per env step for MARL")  # MARL每个环境步对应的梯度更新次数
    parser.add_argument('--marl_learning_starts', default=2048, type=int, help="minimum replay size before MARL updates")  # 开始MARL训练更新前所需的最小经验池样本量
    parser.add_argument('--marl_policy_delay', default=2, type=int, help="delayed actor update frequency for MARL twin-critic")  # MARL双Critic结构下的延迟Actor更新频率
    parser.add_argument('--marl_target_policy_noise', default=0.08, type=float, help="target action smoothing noise std for MARL")  # MARL目标动作平滑噪声标准差
    parser.add_argument('--marl_target_noise_clip', default=0.15, type=float, help="target action smoothing noise clip")  # MARL目标动作平滑噪声裁剪范围
    parser.add_argument('--same_day_train_test', default=False, type=bool, help="if True, both train and test use the same OD training day")  # 若为True，则训练和测试使用同一天OD数据
    parser.add_argument('--fixed_day_index', default=1, type=int, help="1-based training day index used when same_day_train_test=True")  # 当same_day_train_test=True时使用的训练日编号，从1开始
    # ====== 修改结束

    # ========= 2026.2.13 修改：修复命令行参数解析 =========
    # 原代码：args = parser.parse_args([])  # 传入空列表会忽略命令行参数
    # 修复：不传参数，让argparse自动读取sys.argv
    args = parser.parse_args()  # 正确读取命令行参数
    # ====== 修改结束

    args = {**vars(args)}  # 将args转换为字典

    # 打印参数
    print("Training arguments:")
    print(''.join(['='] * 80))
    tplt = "{:^20}\t{:^20}\t{:^20}"
    print(tplt.format("param", "value", "type"))
    for k, v in args.items():
        print(tplt.format(k, v, str(type(v))))
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
