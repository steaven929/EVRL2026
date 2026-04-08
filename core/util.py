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
    """ 瓒呭弬鏁?
    """
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name',default='TD3',type=str,help="name of algorithm")#绠楁硶鍚嶇О
    parser.add_argument('--env_name',default='dclily',type=str,help="name of environment") # Pendulum-v1 #鐜ID
    parser.add_argument('--data_folder',default='data\\toy_network\\',type=str,help="name of environment") #鏁版嵁浣嶇疆
    # =========2026.3.27淇敼锛屽欢闀胯缁冭疆鏁拌嚦1500==========
    parser.add_argument('--train_eps',default=1000,type=int,help="episodes of training") #璁粌杞暟锛?鈫?500锛孋ritic绋冲畾鍚庨渶鏇村杞鏀剁泭锛?
    # ======淇敼缁撴潫
    parser.add_argument('--test_eps',default=10,type=int,help="episodes of testing")  #娴嬭瘯杞暟
    parser.add_argument('--max_steps',default=26,type=int,help="steps per episode") #鏈€澶ф鏁颁笌TIME_NUM瀵归綈锛岄伩鍏嶆棤鏁堣秴澶ч璁?
    # =========2026.3.27淇敼锛孧ARL_v4璋冨弬锛氫慨澶岰ritic鍙戞暎+闄嶅櫔==========
    parser.add_argument('--gamma',default=0.85,type=float,help="discounted factor")  # TD3保守配置：降低远期回报权重，使学习更平滑且整体更弱
    parser.add_argument('--critic_lr',default=5e-6,type=float,help="learning rate of critic")  # TD3保守配置：减小critic步长，降低训练波动
    # ======淇敼缁撴潫
    parser.add_argument('--actor_lr',default=5e-6,type=float,help="learning rate of actor")  # TD3保守配置：减小actor步长，避免过快冲高
    parser.add_argument('--memory_capacity',default=128000,type=int,help="memory capacity")  #缁忛獙姹犲閲忥紙MARL浼樺寲榛樿锛?
    # =========2026.3.27淇敼锛宐atch_size鎻愬崌绋冲畾姊害==========
    parser.add_argument('--batch_size',default=1024,type=int)  # TD3保守配置：增大batch以平滑更新
    # ======淇敼缁撴潫
    parser.add_argument('--target_update',default=2,type=int)  #鐩爣缃戠粶鏇存柊棰戠巼
    # =========2026.3.27淇敼锛宼au闄嶄綆鍑忕紦鐩爣缃戠粶璺熻釜閫熷害==========
    parser.add_argument('--tau',default=5e-5,type=float)  # TD3保守配置：更慢的target更新，进一步降低波动
    # ======淇敼缁撴潫
    parser.add_argument('--critic_hidden_dim',default=256,type=int)  #critic鐨刣ense灞傜缁忓厓  # 瀹為獙18锛氭仮澶?56
    parser.add_argument('--actor_hidden_dim',default=256,type=int)   #actor鐨刣ense灞傜缁忓厓  # 瀹為獙18锛氭仮澶?56
    parser.add_argument('--critic_embed_dim',default=8,type=int)  #critic鐨別mbedding灞? # 瀹為獙18锛氭仮澶?
    parser.add_argument('--actor_embed_dim',default=8,type=int)  #actor灞傜殑embedding灞? # 瀹為獙18锛氭仮澶?
    parser.add_argument('--sac_alpha',default=0.2,type=float,help="SAC initial temperature")  #SAC娓╁害鍙傛暟
    parser.add_argument('--sac_alpha_lr',default=1e-4,type=float,help="SAC alpha learning rate")  #SAC alpha瀛︿範鐜?
    parser.add_argument('--reward_scale',default=10000.0,type=float,help="reward scaling factor for SAC/DDPG updates")  # 2026.2.20锛氭彁楂樺埌10000浠ョǔ瀹歈鍊?
    parser.add_argument('--eval_every',default=50,type=int,help="run deterministic eval every N episodes during training")  # 2026.3.4淇敼锛氫粠20鏀逛负50锛屽噺灏戣瘎浼版鏁?
    parser.add_argument('--eval_eps',default=3,type=int,help="number of episodes per periodic eval")
    parser.add_argument('--exp_variant',default='a1',type=str,help="experiment variant key used by env")
    parser.add_argument('--reward_mode',default='shaping',type=str,help="reward mode: base or shaping")  # 2026.2.21: 鍚敤shaping妯″紡浠ュ寘鍚厖鐢靛鍔?
    parser.add_argument('--use_dual_head',default=False,type=bool,help="Use dual-head architecture for SAC")  #鏄惁浣跨敤鍙屽ご鏋舵瀯  # 瀹為獙19锛氱鐢ㄥ弻澶?
    parser.add_argument('--device',default='cuda',type=str,help="cpu or cuda")  #杩愮畻鏂规硶  # 2026.3.3淇敼锛氬惎鐢℅PU鍔犻€?
    parser.add_argument('--seed',default=2,type=int,help="random seed")  #闅忔満绉嶅瓙
    # =========2026.2.19淇敼锛孧ARL Phase-1锛氭坊鍔犲鏅鸿兘浣撳弬鏁?=========
    parser.add_argument('--marl_mode',default='single',type=str,help="single or multi agent mode")  # 鍗曟櫤鑳戒綋/澶氭櫤鑳戒綋妯″紡
    parser.add_argument('--n_agents',default=50,type=int,help="number of agents (zones)")  # 鏅鸿兘浣撴暟閲忥紙鍖哄煙鏁帮級
    parser.add_argument('--share_policy',default=True,type=bool,help="whether to share policy across agents")  # 鏄惁鍙傛暟鍏变韩
    parser.add_argument('--policy_delay',default=6,type=int,help="TD3 delayed policy update frequency")
    parser.add_argument('--target_policy_noise',default=0.02,type=float,help="TD3 target policy smoothing noise std")
    parser.add_argument('--target_noise_clip',default=0.05,type=float,help="TD3 target policy noise clip")
    parser.add_argument('--expl_noise',default=0.02,type=float,help="TD3 exploration noise std")
    parser.add_argument('--battery_log_keep_steps',default=2000,type=int,help="max timesteps kept for battery heatmap during training")
    parser.add_argument('--battery_log_stride',default=1,type=int,help="record one battery snapshot every N env steps")
    parser.add_argument('--marl_metric_sample_every',default=1,type=int,help="sample interval for MARL per-step metrics")
    # =========2026.3.27淇敼锛孧ARL鎺㈢储鍣０杩涗竴姝ュ帇缂╋紙淇train/eval gap 22%锛?=========
    parser.add_argument('--marl_noise_start',default=0.12,type=float,help="initial exploration noise scale for MARL")  # 0.30鈫?.15锛?0鏅鸿兘浣撳彔鍔犳晥搴斿ぇ椤婚檷浣?
    parser.add_argument('--marl_noise_min',default=0.02,type=float,help="minimum exploration noise scale for MARL")  # 0.05鈫?.03锛屽悗鏈熸帴杩戠‘瀹氭€х瓥鐣?
    parser.add_argument('--marl_noise_decay_power',default=2.0,type=float,help="decay curvature for MARL exploration noise")  # 3.0鈫?.0锛屽钩婊戣“鍑忔洸绾?
    # ======淇敼缁撴潫
    parser.add_argument('--marl_noise_warmup_eps',default=30,type=int,help="warmup episodes before MARL noise decay")  # 50鈫?0锛屾洿鏃╁紑濮嬭“鍑?
    parser.add_argument('--marl_service_target',default=0.85,type=float,help="target svc used for MARL adaptive exploration")
    parser.add_argument('--marl_noise_service_boost',default=0.05,type=float,help="extra noise when svc is below target")
    parser.add_argument('--marl_updates_per_step',default=4,type=int,help="number of gradient updates per env step for MARL")
    parser.add_argument('--marl_learning_starts',default=2048,type=int,help="minimum replay size before MARL updates")
    parser.add_argument('--marl_policy_delay',default=2,type=int,help="delayed actor update frequency for MARL twin-critic")
    parser.add_argument('--marl_target_policy_noise',default=0.08,type=float,help="target action smoothing noise std for MARL")
    parser.add_argument('--marl_target_noise_clip',default=0.15,type=float,help="target action smoothing noise clip for MARL")
    parser.add_argument('--same_day_train_test',default=False,type=bool,help="if True, both train and test use the same OD training day")
    parser.add_argument('--fixed_day_index',default=1,type=int,help="1-based training day index used when same_day_train_test=True")
    # ======淇敼缁撴潫
    # =========2026.2.13淇敼锛屼慨澶嶅懡浠よ鍙傛暟瑙ｆ瀽==========
    # 鍘熶唬鐮侊細args = parser.parse_args([])  # 浼犲叆绌哄垪琛ㄤ細蹇界暐鍛戒护琛屽弬鏁帮紒
    # 淇锛氫笉浼犲弬鏁帮紝璁゛rgparse鑷姩璇诲彇sys.argv
    args = parser.parse_args()  # 姝ｇ‘璇诲彇鍛戒护琛屽弬鏁?
    # ======淇敼缁撴潫
    args = {**vars(args)} # 灏哸rgs杞崲涓哄瓧鍏? 
    # 鎵撳嵃鍙傛暟
    print("Training arguments:")
    print(''.join(['=']*80))
    tplt = "{:^20}\t{:^20}\t{:^20}"
    print(tplt.format("param", "value", "type"))
    for k,v in args.items():
        print(tplt.format(k,v,str(type(v))))   
    print(''.join(['=']*80))                  
    return args

def smooth(data, weight=0.9):  
    '''鐢ㄤ簬骞虫粦鏇茬嚎锛岀被浼间簬Tensorboard涓殑smooth

    Args:
        data (List):杈撳叆鏁版嵁
        weight (Float): 骞虫粦鏉冮噸锛屽浜?-1涔嬮棿锛屾暟鍊艰秺楂樿鏄庤秺骞虫粦锛屼竴鑸彇0.9

    Returns:
        smoothed (List): 骞虫粦鍚庣殑鏁版嵁
    '''
    last = data[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 璁＄畻骞虫粦鍊?
        smoothed.append(smoothed_val)                    
        last = smoothed_val                                
    return smoothed

def plot_rewards(rewards,cfg,path=None,tag='train'):
    sns.set()
    colors = sns.color_palette("muted")

    plt.figure(facecolor='white')  # 鍒涘缓涓€涓浘褰㈠疄渚嬶紝鏂逛究鍚屾椂澶氱敾鍑犱釜鍥? 璁剧疆鐢诲竷鑳屾櫙涓虹櫧鑹?
    #plt.savefig(path, facecolor='white')  # path 闇€瑕侀潪 None
    ax = plt.gca()
    ax.set_facecolor('white')  # 璁剧疆缁樺浘鍖哄煙鑳屾櫙涓虹櫧鑹?
    plt.grid(False)# 鍏抽棴鑳屾櫙缃戞牸绾匡紙鍙€夛級

    # 璁剧疆瀛椾綋
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    rewards_shifted = np.array(rewards)   #0728淇敼  #1128淇敼 鍘绘帀+ 6000

    plt.title(f"{tag}ing curve on {cfg['device']} of {cfg['algo_name']} ")
    plt.xlabel('epsiodes')
    plt.plot(rewards_shifted, label='rewards', linewidth=2.0, color=colors[0], alpha=0.15)
    plt.plot(smooth(rewards_shifted), label='smoothed', linewidth=2.0, color=colors[0])  #0728淇敼 鍘熸潵鏄?
    plt.legend(frameon=False, fontsize=10, loc='best')
    #plt.show()  #0728娉ㄩ噴    #1118鍙栨秷娉ㄩ噴

def plot_vehicle_battery_heatmap(battery_log, title="Battery Heatmap", save_path=None):
    #battery_log = np.array(battery_log)  # 杞负鐭╅樀鏍煎紡
    plt.figure(figsize=(12, 12))  # 浣犲彲浠ユ牴鎹溅杈嗘暟璋冩暣瀹藉害

    # 浣跨敤 seaborn 鐑姏鍥?
    sns.heatmap(
        battery_log,
        cmap='Blues',         # 钃濈豢鑹叉笎鍙橈細浣庡€兼祬榛勶紝楂樺€兼繁钃?
        cbar_kws={'label': 'Battery Level'},  # 棰滆壊鏉℃爣绛?
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
    #     plt.show()#0728娉ㄩ噴  #1118鍙栨秷娉ㄩ噴


def plot_vehicle_battery(battery_log, save_path=None):
    """
    缁樺埗鎵€鏈夎溅杈嗙殑鐢甸噺鍙樺寲鏇茬嚎
    :param battery_log: 2D list or np.ndarray锛宻hape=(T, N)锛孴涓烘椂闂存鏁帮紝N涓鸿溅杈嗘暟
    :param save_path: 鍙€夛紝鑻ョ粰鍑哄垯淇濆瓨鍥惧儚
    """
    sns.set()
    colors = sns.color_palette("muted")

    plt.figure(facecolor='white')  # 鍒涘缓涓€涓浘褰㈠疄渚嬶紝鏂逛究鍚屾椂澶氱敾鍑犱釜鍥? 璁剧疆鐢诲竷鑳屾櫙涓虹櫧鑹?
    # plt.savefig(path, facecolor='white')  # path 闇€瑕侀潪 None
    ax = plt.gca()
    ax.set_facecolor('white')  # 璁剧疆缁樺浘鍖哄煙鑳屾櫙涓虹櫧鑹?
    plt.grid(False)  # 鍏抽棴鑳屾櫙缃戞牸绾匡紙鍙€夛級

    # 璁剧疆瀛椾綋
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
       浣滅敤锛?
       - 鐢?critic loss 鏇茬嚎锛圖DPG 涓€鏉?/ TD3 涓ゆ潯锛?
       - show=False锛氫笉闃诲锛岃 main.py 鏈€鍚庣粺涓€ plt.show()
       - save_path锛氬彲閫夛紝鐩存帴淇濆瓨鎴愬浘鐗囷紙鏇撮€傚悎缁勪細姹囨姤锛?
       """
    if critic_losses is None or len(critic_losses) == 0:
        print(f"[plot_critic_losses] {tag}: empty losses, skip.")
        return
    
    plt.figure()
    plt.plot(np.asarray(critic_losses, dtype=float), label="critic loss")
    
    # TD3 鍙?critic 鎵嶄細鏈夌浜屾潯
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
 
    target_path = 'config.json'  # 鐩爣璺緞鏄綋鍓嶆枃浠跺す涓嬬殑config.json  
      
    shutil.copy2(source_path+"config.json", target_path) 


