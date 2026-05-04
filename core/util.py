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
    """ 超参数
    """
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name',default='SAC',type=str,help="name of algorithm")#算法名称
    parser.add_argument('--env_name',default='dclily',type=str,help="name of environment") # Pendulum-v1 #环境ID
    parser.add_argument('--data_folder',default='data\\toy_network\\',type=str,help="name of environment") #数据位置
    parser.add_argument('--train_eps',default=1200,type=int,help="episodes of training") #训练轮数  # 2026.2.13：默认1000轮
    parser.add_argument('--test_eps',default=10,type=int,help="episodes of testing")  #测试轮数
    parser.add_argument('--max_steps',default=100000,type=int,help="steps per episode, much larger value can simulate infinite steps") #最大步数（这个弃用了，不需要设置）
    parser.add_argument('--gamma',default=0.95,type=float,help="discounted factor")  #折扣因子（更稳的长期回报）
    parser.add_argument('--critic_lr',default=1e-4,type=float,help="learning rate of critic")  #critic学习率（稳态配置）
    parser.add_argument('--actor_lr',default=5e-5,type=float,help="learning rate of actor")  #actor学习率（稳态配置）
    parser.add_argument('--memory_capacity',default=64000,type=int,help="memory capacity")  #经验池容量
    parser.add_argument('--batch_size',default=128,type=int)  #批量学习数  # 2026.3.3修改：提升到512以利用GPU并行
    parser.add_argument('--target_update',default=2,type=int)  #目标网络更新频率
    parser.add_argument('--tau',default=5e-4,type=float)  #软更新参数（稳态配置）
    parser.add_argument('--critic_hidden_dim',default=256,type=int)  #critic的dense层神经元  # 实验18：恢复256
    parser.add_argument('--actor_hidden_dim',default=256,type=int)   #actor的dense层神经元  # 实验18：恢复256
    parser.add_argument('--critic_embed_dim',default=8,type=int)  #critic的embedding层  # 实验18：恢复8
    parser.add_argument('--actor_embed_dim',default=8,type=int)  #actor层的embedding层  # 实验18：恢复8
    parser.add_argument('--sac_alpha',default=0.2,type=float,help="SAC initial temperature")  #SAC温度参数
    parser.add_argument('--sac_alpha_lr',default=1e-4,type=float,help="SAC alpha learning rate")  #SAC alpha学习率
    parser.add_argument('--reward_scale',default=10000.0,type=float,help="reward scaling factor for SAC/DDPG updates")  # 2026.2.20：提高到10000以稳定Q值
    parser.add_argument('--eval_every',default=20,type=int,help="run deterministic eval every N episodes during training")  # 2026.3.4修改：从20改为50，减少评估次数
    parser.add_argument('--eval_eps',default=3,type=int,help="number of episodes per periodic eval")
    parser.add_argument('--exp_variant',default='a1',type=str,help="experiment variant key used by env")
    parser.add_argument('--reward_mode',default='shaping',type=str,help="reward mode: base or shaping")  # 2026.2.21: 启用shaping模式以包含充电奖励
    parser.add_argument('--use_dual_head',default=False,type=bool,help="Use dual-head architecture for SAC")  #是否使用双头架构  # 实验19：禁用双头
    parser.add_argument('--device',default='cpu',type=str,help="cpu or cuda")  #运算方法  # 2026.3.3修改：启用GPU加速
    parser.add_argument('--seed',default=2,type=int,help="random seed")  #随机种子
    parser.add_argument('--sa_init_temp',default=1.5,type=float,help="initial temperature for simulated annealing")
    parser.add_argument('--sa_final_temp',default=0.02,type=float,help="final temperature for simulated annealing")
    parser.add_argument('--sa_cooling_rate',default=0.985,type=float,help="multiplicative cooling rate for simulated annealing")
    parser.add_argument('--sa_iters_per_step',default=120,type=int,help="annealing iterations per environment timestep")
    parser.add_argument('--sa_seed_action',default='demand_charge',type=str,help="seed action strategy for simulated annealing")
    parser.add_argument('--sa_weight_dispatch_support',default=0.4,type=float,help="dispatch support weight for simulated annealing scoring")
    parser.add_argument('--sa_weight_charging_bonus',default=0.9,type=float,help="charging bonus weight for simulated annealing scoring")
    parser.add_argument('--sa_weight_dispatch_cost',default=0.15,type=float,help="dispatch cost weight for simulated annealing scoring")
    parser.add_argument('--sa_weight_low_soc_penalty',default=2.5,type=float,help="low-SOC penalty weight for simulated annealing scoring")
    parser.add_argument('--sa_weight_unmet_penalty',default=0.1,type=float,help="unmet-demand penalty weight for simulated annealing scoring")
    # =========2026.2.19修改，MARL Phase-1：添加多智能体参数==========
    parser.add_argument('--marl_mode',default='single',type=str,help="single or multi agent mode")  # 单智能体/多智能体模式
    parser.add_argument('--n_agents',default=50,type=int,help="number of agents (zones)")  # 智能体数量（区域数）
    parser.add_argument('--share_policy',default=True,type=bool,help="whether to share policy across agents")  # 是否参数共享
    parser.add_argument('--policy_delay',default=2,type=int,help="TD3 delayed policy update frequency")
    parser.add_argument('--target_policy_noise',default=0.02,type=float,help="TD3 target policy smoothing noise std")
    parser.add_argument('--target_noise_clip',default=0.05,type=float,help="TD3 target policy noise clip")
    parser.add_argument('--expl_noise',default=0.02,type=float,help="TD3 exploration noise std")
    # ======修改结束
    # =========2026.2.13修改，修复命令行参数解析==========
    # 原代码：args = parser.parse_args([])  # 传入空列表会忽略命令行参数！
    # 修复：不传参数，让argparse自动读取sys.argv
    args = parser.parse_args()  # 正确读取命令行参数
    # ======修改结束
    args = {**vars(args)} # 将args转换为字典  
    # 打印参数
    print("训练参数如下：")
    print(''.join(['=']*80))
    tplt = "{:^20}\t{:^20}\t{:^20}"
    print(tplt.format("参数名","参数值","参数类型"))
    for k,v in args.items():
        print(tplt.format(k,v,str(type(v))))   
    print(''.join(['=']*80))                  
    return args

def smooth(data, weight=0.9):  
    '''用于平滑曲线，类似于Tensorboard中的smooth

    Args:
        data (List):输入数据
        weight (Float): 平滑权重，处于0-1之间，数值越高说明越平滑，一般取0.9

    Returns:
        smoothed (List): 平滑后的数据
    '''
    last = data[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)                    
        last = smoothed_val                                
    return smoothed

def plot_rewards(rewards,cfg,path=None,tag='train'):
    sns.set()
    colors = sns.color_palette("muted")

    plt.figure(facecolor='white')  # 创建一个图形实例，方便同时多画几个图# 设置画布背景为白色
    #plt.savefig(path, facecolor='white')  # path 需要非 None
    ax = plt.gca()
    ax.set_facecolor('white')  # 设置绘图区域背景为白色
    plt.grid(False)# 关闭背景网格线（可选）

    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    rewards_shifted = np.array(rewards)   #0728修改  #1128修改 去掉+ 6000

    plt.title(f"{tag}ing curve on {cfg['device']} of {cfg['algo_name']} ")
    plt.xlabel('epsiodes')
    plt.plot(rewards_shifted, label='rewards', linewidth=2.0, color=colors[0], alpha=0.15)
    plt.plot(smooth(rewards_shifted), label='smoothed', linewidth=2.0, color=colors[0])  #0728修改 原来是1
    plt.legend(frameon=False, fontsize=10, loc='best')
    #plt.show()  #0728注释    #1118取消注释

def plot_vehicle_battery_heatmap(battery_log, title="Battery Heatmap", save_path=None):
    #battery_log = np.array(battery_log)  # 转为矩阵格式
    plt.figure(figsize=(12, 12))  # 你可以根据车辆数调整宽度

    # 使用 seaborn 热力图
    sns.heatmap(
        battery_log,
        cmap='Blues',         # 蓝绿色渐变：低值浅黄，高值深蓝
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
    #     plt.show()#0728注释  #1118取消注释


def plot_vehicle_battery(battery_log, save_path=None):
    """
    绘制所有车辆的电量变化曲线
    :param battery_log: 2D list or np.ndarray，shape=(T, N)，T为时间步数，N为车辆数
    :param save_path: 可选，若给出则保存图像
    """
    sns.set()
    colors = sns.color_palette("muted")

    plt.figure(facecolor='white')  # 创建一个图形实例，方便同时多画几个图# 设置画布背景为白色
    # plt.savefig(path, facecolor='white')  # path 需要非 None
    ax = plt.gca()
    ax.set_facecolor('white')  # 设置绘图区域背景为白色
    plt.grid(False)  # 关闭背景网格线（可选）

    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    #sns.set_style("whitegrid")
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
       作用：
       - 画 critic loss 曲线（DDPG 一条 / TD3 两条）
       - show=False：不阻塞，让 main.py 最后统一 plt.show()
       - save_path：可选，直接保存成图片（更适合组会汇报）
       """
    if critic_losses is None or len(critic_losses) == 0:
        print(f"[plot_critic_losses] {tag}: empty losses, skip.")
        return
    
    plt.figure()
    plt.plot(np.asarray(critic_losses, dtype=float), label="critic loss")
    
    # TD3 双 critic 才会有第二条
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
 
    target_path = 'config.json'  # 目标路径是当前文件夹下的config.json  
      
    shutil.copy2(source_path+"config.json", target_path) 
