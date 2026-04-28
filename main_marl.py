# -*- coding: utf-8 -*-
"""
MARL Phase-1: Main Entry Point
Created on 2026-02-19

澶氭櫤鑳戒綋寮哄寲瀛︿範涓荤▼搴?
"""
import sys
import os
import torch
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 娣诲姞椤圭洰璺緞
sys.path.append(os.getcwd())

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from core.util import get_args, plot_rewards, plot_critic_losses
from core.environment import env_agent_config
from core.marl_training import marl_train, marl_test

from gym.envs.registration import register
register(id='dclily', entry_point='specific.dclily_elec:dclily')

# =========2026.2.19淇敼锛孧ARL Phase-1锛氫富绋嬪簭==========
def main():
    # 鑾峰彇閰嶇疆鍙傛暟
    cfg = get_args()

    # 寮哄埗璁剧疆涓簃ulti妯″紡
    cfg['marl_mode'] = 'multi'
    cfg['n_agents'] = 50  # SIP鏁版嵁闆嗭細50涓尯鍩?

    print("\n" + "="*80)
    print("MARL Phase-1: Multi-agent reinforcement learning")
    print("="*80)
    print(f"Mode: {cfg['marl_mode']}")
    print(f"Agents: {cfg['n_agents']}")
    print(f"Share policy: {cfg.get('share_policy', True)}")
    print(f"Experiment variant: {cfg.get('exp_variant', 'a1')}")
    print(f"Reward mode: {cfg.get('reward_mode', 'base')}")
    print(f"Train episodes: {cfg['train_eps']}")
    print(f"Test episodes: {cfg['test_eps']}")
    print("="*80 + "\n")

    # 鍒涘缓鐜鍜屾櫤鑳戒綋
    env, agent = env_agent_config(cfg)

    # 璁剧疆淇濆瓨璺緞
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = f"./outputs/marl_{curr_time}/"
    os.makedirs(path, exist_ok=True)

    # 璁粌
    print("\n" + "="*80)
    print("Start training")
    print("="*80)
    res_dic = marl_train(env, agent, cfg, path=path)

    # 淇濆瓨鎵€鏈夋暟鎹紙涓巑ain.py涓€鑷达級
    np.save(path + 'critic_losses.npy', res_dic['critic_losses'])
    np.save(path + 'actor_losses.npy', res_dic['actor_losses'])
    np.save(path + 'rewards.npy', res_dic['rewards'])
    if 'raw_rewards' in res_dic:
        np.save(path + 'raw_rewards.npy', res_dic['raw_rewards'])
    np.save(path + 'ma_rewards.npy', res_dic['ma_rewards'])
    if 'rld_history' in res_dic:
        np.save(path + 'rld_history.npy', res_dic['rld_history'])
    if 'svc_history' in res_dic:
        np.save(path + 'svc_history.npy', res_dic['svc_history'])
    if 'srd_history' in res_dic:
        np.save(path + 'srd_history.npy', res_dic['srd_history'])
    for key in [
        'rld_last_history',
        'svc_last_history',
        'srd_last_history',
        'service_rate_history',
        'service_rate_last_history',
        'charging_efficiency_history',
        'charger_queue_delay_history',
        'energy_not_served_history',
        'safety_soc_violation_count_history',
    ]:
        if key in res_dic:
            np.save(path + f'{key}.npy', res_dic[key])
    print(f"Data saved to: {path}")
    print(f"Info瀛楁缂哄け璁℃暟: {res_dic.get('info_missing_counts', {})}")

    # 淇濆瓨妯″瀷
    agent.save_model(path + 'marl_model_final.pth')
    print(f"Model saved: {path}marl_model_final.pth")

    # 缁樺埗璁粌鏇茬嚎
    print("\n缁樺埗璁粌鏇茬嚎...")
    plot_rewards(res_dic['rewards'], cfg, path=path + 'train_rewards.png', tag='train')

    # 缁樺埗Critic Loss鏇茬嚎
    plot_critic_losses(res_dic['critic_losses'], None, tag="train", save_path=path + "critic_loss_curve.png")

    # 鐢熸垚缁煎悎鎸囨爣鍥撅紙涓巑ain.py涓€鑷寸殑8涓瓙鍥撅級
    print("鐢熸垚缁煎悎鎸囨爣鍥?..")
    fig, axes = plt.subplots(8, 1, figsize=(10, 20))

    # 瀛愬浘1: Critic Loss
    axes[0].plot(res_dic['critic_losses'], label='Critic Loss', color='blue', linewidth=1.5)
    axes[0].set_xlabel('Update Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Critic Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 瀛愬浘2: Actor Loss
    axes[1].plot(res_dic['actor_losses'], label='Actor Loss (-Q)', color='red', linewidth=1.5)
    axes[1].set_xlabel('Update Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Actor Loss (-Q Value, should decrease)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 瀛愬浘3-4: 棰勭暀锛圡ARL娌℃湁鍗曠嫭鐨凲鍊艰褰曪級
    axes[2].text(0.5, 0.5, 'Q Value tracking not available in MARL',
                ha='center', va='center', transform=axes[2].transAxes)
    axes[2].set_title('Q Value (N/A for MARL)')

    axes[3].text(0.5, 0.5, 'Target Q tracking not available in MARL',
                ha='center', va='center', transform=axes[3].transAxes)
    axes[3].set_title('Target Q Value (N/A for MARL)')

    # 瀛愬浘5: RLD
    if 'rld_history' in res_dic and len(res_dic['rld_history']) > 0:
        axes[4].plot(res_dic['rld_history'], label='RLD', color='green', linewidth=1.5)
        axes[4].set_xlabel('Step')
        axes[4].set_ylabel('RLD')
        axes[4].set_title('Repositioning Loss Distance (RLD)')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)

    # 瀛愬浘6: SVC
    if 'svc_history' in res_dic and len(res_dic['svc_history']) > 0:
        axes[5].plot(res_dic['svc_history'], label='SVC', color='purple', linewidth=1.5)
        axes[5].set_xlabel('Step')
        axes[5].set_ylabel('SVC')
        axes[5].set_title('Service Rate (SVC)')
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)

    # 瀛愬浘7: SRD
    if 'srd_history' in res_dic and len(res_dic['srd_history']) > 0:
        axes[6].plot(res_dic['srd_history'], label='SRD', color='orange', linewidth=1.5)
        axes[6].set_xlabel('Step')
        axes[6].set_ylabel('SRD')
        axes[6].set_title('Service Reward Ratio (SRD)')
        axes[6].legend()
        axes[6].grid(True, alpha=0.3)

    # 瀛愬浘8: Episode Reward
    axes[7].plot(res_dic['rewards'], label='Reward', color='gray', alpha=0.3, linewidth=1.0)
    from core.util import smooth
    axes[7].plot(smooth(res_dic['rewards']), label='Smoothed', color='blue', linewidth=2.0)
    axes[7].set_xlabel('Episode')
    axes[7].set_ylabel('Reward')
    axes[7].set_title('Episode Reward')
    axes[7].legend()
    axes[7].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path + 'final_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Final metrics figure saved: {path}final_metrics.png")

    # 娴嬭瘯
    print("\n" + "="*80)
    print("Start testing")
    print("="*80)
    test_rewards = marl_test(env, agent, cfg, path=path)

    print("\n" + "="*80)
    print("Training complete")
    print(f"Results saved to: {path}")
    print("="*80)

if __name__ == "__main__":
    main()
# ======淇敼缁撴潫




