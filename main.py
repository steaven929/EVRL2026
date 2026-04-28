# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 19:29:48 2024

@author: TLab
"""
# =========2026.2.5淇敼锛屼慨澶峎indows涓枃缂栫爜闂==========
import sys
import io
# 璁剧疆鏍囧噯杈撳嚭涓?UTF-8 缂栫爜
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# ======淇敼缁撴潫

from matplotlib import pyplot as plt

from core.util import get_args, plot_rewards,plot_vehicle_battery_heatmap,plot_vehicle_battery, cp, plot_critic_losses, smooth
import core.environment as dclilyenv
import core.training as dclilytrain
import numpy as np
import os

from gym.envs.registration import register

# 寤虹珛鐜

register(
    id='dclily',  # 鑷畾涔夌幆澧冪殑鍞竴 ID
    entry_point='specific.dclily_elec:dclily',  # 鐜绫荤殑璺緞锛氭ā鍧?绫诲悕
)

# 鑾峰彇鍙傛暟
cfg = get_args()

# 澶嶅埗鍙傛暟h

# 鑾峰彇 main.py 鎵€鍦ㄧ殑缁濆璺緞
current_dir = os.path.dirname(os.path.abspath(__file__))
# 鎷兼帴鍑?specific 鏂囦欢澶圭殑缁濆璺緞
target_dir = os.path.join(current_dir, 'specific')

# 鍒囨崲宸ヤ綔鐩綍
os.chdir(target_dir)
#os.chdir('.\\specific')
cp("..\\" + cfg['data_folder'])
os.chdir('..')

# 璁粌
env, agent = dclilyenv.env_agent_config(cfg)


# 鉁?娣诲姞锛氱粰鐜缁戝畾 ZONE_NUM 灞炴€э紙鐢ㄤ簬鍒囩墖锛?
#env.ZONE_NUM = env.action_space.shape[0] //


# 鉁?娣诲姞锛氳嚜瀹氫箟鍔ㄤ綔娴嬭瘯
#from core.environment import manual_action_test  # 璇ュ嚱鏁板姞鍦?environment.py 閲?
#manual_action_test(env)

#璁粌
res_dic = dclilytrain.train(cfg, env, agent)

# 淇濆瓨loss鏁版嵁鍜屾寚鏍囨暟鎹? # 淇敼
np.save('critic1_losses.npy', res_dic['critic1_losses'])  # 淇敼
if res_dic.get('critic2_losses'):  # 淇敼
    np.save('critic2_losses.npy', res_dic['critic2_losses'])  # 淇敼
if res_dic.get('actor_losses'):  # 淇敼
    np.save('actor_losses.npy', res_dic['actor_losses'])  # 淇敼
if res_dic.get('q_values'):  # 淇敼
    np.save('q_values.npy', res_dic['q_values'])  # 淇敼
if res_dic.get('target_q_values'):  # 淇敼
    np.save('target_q_values.npy', res_dic['target_q_values'])  # 淇敼
np.save('rewards.npy', res_dic['rewards'])  # 淇敼锛氫繚瀛?episode rewards
if res_dic.get('raw_rewards'):
    np.save('raw_rewards.npy', res_dic['raw_rewards'])
np.save('eval_rewards.npy', res_dic.get('eval_rewards', []))
np.save('eval_episodes.npy', res_dic.get('eval_episodes', []))
np.save('rld_history.npy', res_dic['rld_history'])  # 淇敼
np.save('svc_history.npy', res_dic['svc_history'])  # 淇敼
np.save('srd_history.npy', res_dic['srd_history'])  # 淇敼
for key in [
    'rld_last_history',
    'svc_last_history',
    'srd_last_history',
    'service_rate_history',
    'service_rate_last_history',
]:
    if key in res_dic:
        np.save(f'{key}.npy', res_dic[key])
for key in [
    'charging_efficiency_history',
    'charger_queue_delay_history',
    'energy_not_served_history',
    'safety_soc_violation_count_history',
]:
    if key in res_dic:
        np.save(f'{key}.npy', res_dic[key])
if 'battery_log' in res_dic:
    np.save('battery_log.npy', res_dic['battery_log'])
print(f"Loss鏁版嵁鍜屾寚鏍囧凡淇濆瓨")  # 淇敼
print(f"Info瀛楁缂哄け璁℃暟: {res_dic.get('info_missing_counts', {})}")

# =========2026.1.31淇敼锛屼繚瀛樿缁冨ソ鐨勬ā鍨?=========
agent.save_model('trained_model.pth')
print(f"璁粌瀹屾垚锛屾ā鍨嬪凡淇濆瓨鍒? trained_model.pth")
# ======淇敼缁撴潫

plot_rewards(res_dic['rewards'], cfg, tag="train")

plot_critic_losses(res_dic['critic1_losses'], res_dic.get('critic2_losses', None), tag="train", save_path="critic_loss_curve.png")

# 鐢熸垚鍖呭惈7涓瓙鍥剧殑缁煎悎鎸囨爣鍥? # 淇敼
import matplotlib  # 淇敼
matplotlib.use('Agg')  # 淇敼
fig, axes = plt.subplots(8, 1, figsize=(10, 20))  # 淇敼

# 瀛愬浘1: Critic Loss  # 淇敼
axes[0].plot(res_dic['critic1_losses'], label='Critic Loss', color='blue', linewidth=1.5)  # 淇敼
axes[0].set_xlabel('Update Step')  # 淇敼
axes[0].set_ylabel('Loss')  # 淇敼
axes[0].set_title('Critic Loss')  # 淇敼
axes[0].legend()  # 淇敼
axes[0].grid(True, alpha=0.3)  # 淇敼

# 瀛愬浘2: Actor Loss (-Q鍊硷紝搴旇涓嬮檷)  # 淇敼
if res_dic.get('actor_losses') and len(res_dic['actor_losses']) > 0:  # 淇敼
    axes[1].plot(res_dic['actor_losses'], label='Actor Loss (-Q)', color='red', linewidth=1.5)  # 淇敼
axes[1].set_xlabel('Update Step')  # 淇敼
axes[1].set_ylabel('Loss')  # 淇敼
axes[1].set_title('Actor Loss (-Q Value, should decrease)')  # 淇敼
axes[1].legend()  # 淇敼
axes[1].grid(True, alpha=0.3)  # 淇敼

# 瀛愬浘3: Q Value (搴旇涓婂崌锛屽彉寰椾笉閭ｄ箞璐?  # 淇敼
if res_dic.get('q_values') and len(res_dic['q_values']) > 0:  # 淇敼
    axes[2].plot(res_dic['q_values'], label='Q Value', color='cyan', linewidth=1.5)  # 淇敼
axes[2].set_xlabel('Update Step')  # 淇敼
axes[2].set_ylabel('Q Value')  # 淇敼
axes[2].set_title('Q Value (should increase towards 0)')  # 淇敼
axes[2].legend()  # 淇敼
axes[2].grid(True, alpha=0.3)  # 淇敼

# 瀛愬浘4: Target Q Value  # 淇敼
if res_dic.get('target_q_values') and len(res_dic['target_q_values']) > 0:  # 淇敼
    axes[3].plot(res_dic['target_q_values'], label='Target Q', color='magenta', linewidth=1.5)  # 淇敼
axes[3].set_xlabel('Update Step')  # 淇敼
axes[3].set_ylabel('Target Q Value')  # 淇敼
axes[3].set_title('Target Q Value')  # 淇敼
axes[3].legend()  # 淇敼
axes[3].grid(True, alpha=0.3)  # 淇敼

# 瀛愬浘5: RLD  # 淇敼
axes[4].plot(res_dic['rld_history'], label='RLD', color='green', linewidth=1.5)  # 淇敼
axes[4].set_xlabel('Step')  # 淇敼
axes[4].set_ylabel('RLD')  # 淇敼
axes[4].set_title('Repositioning Loss Distance (RLD)')  # 淇敼
axes[4].legend()  # 淇敼
axes[4].grid(True, alpha=0.3)  # 淇敼

# 瀛愬浘6: SVC  # 淇敼
axes[5].plot(res_dic['svc_history'], label='SVC', color='purple', linewidth=1.5)  # 淇敼
axes[5].set_xlabel('Step')  # 淇敼
axes[5].set_ylabel('SVC')  # 淇敼
axes[5].set_title('Service Rate (SVC)')  # 淇敼
axes[5].legend()  # 淇敼
axes[5].grid(True, alpha=0.3)  # 淇敼

# 瀛愬浘7: SRD  # 淇敼
axes[6].plot(res_dic['srd_history'], label='SRD', color='orange', linewidth=1.5)  # 淇敼
axes[6].set_xlabel('Step')  # 淇敼
axes[6].set_ylabel('SRD')  # 淇敼
axes[6].set_title('Service Reward Ratio (SRD)')  # 淇敼
axes[6].legend()  # 淇敼
axes[6].grid(True, alpha=0.3)  # 淇敼

# 瀛愬浘8: Reward锛堟寜 episode锛? # 淇敼
rewards = np.asarray(res_dic.get('rewards', []), dtype=float)
if rewards.size > 0:
    # 鍘熷鏁版嵁锛堢伆鑹茶儗鏅級
    axes[7].plot(rewards, label='Reward', linewidth=2.0, alpha=0.15, color='gray')  # 淇敼

    # 骞虫粦鏁版嵁锛堟繁绾㈣壊锛夛紝鐩存帴璋冪敤 util.py 閲岀殑 smooth
    if len(rewards) > 1:
        try:
            smoothed_rewards = smooth(rewards.tolist(), weight=0.9)
            axes[7].plot(smooth(rewards), label='smoothed', linewidth=2.0)  # 淇敼
        except Exception as e:
            print(f"骞虫粦璁＄畻鍑洪敊: {e}")

    eval_rewards = np.asarray(res_dic.get('eval_rewards', []), dtype=float)
    eval_episodes = np.asarray(res_dic.get('eval_episodes', []), dtype=float)
    if eval_rewards.size > 0 and eval_episodes.size == eval_rewards.size:
        axes[7].plot(eval_episodes - 1, eval_rewards, 'o-', label='eval_reward', linewidth=1.5, markersize=4)

axes[7].set_xlabel('Episode')  # 淇敼
axes[7].set_ylabel('Reward')  # 淇敼
axes[7].set_title('Episode Reward')  # 淇敼
axes[7].legend(loc='upper right')  # 淇敼
axes[7].grid(True, alpha=0.3)  # 淇敼

plt.tight_layout()  # 淇敼
plt.savefig('final_metrics.png', dpi=300, bbox_inches='tight')  # 淇敼
print("缁煎悎鎸囨爣鍥惧凡淇濆瓨: final_metrics.png")  # 淇敼
plt.close()  # 淇敼

#plot_vehicle_battery(np.array(res_dic['battery_log'][0]), cfg)



#杩欎竴閮ㄥ垎琚敞閲婄殑鏄儹鍔涘浘鍥惧儚
# plot_vehicle_battery_heatmap(res_dic['battery_log'], title="Battery Heatmap", save_path=None)
# print(res_dic['battery_log'])
# print(np.array(res_dic['battery_log']))
# plt.show()
# ==========================================
# 鉁?淇敼锛氱粯鍒剁儹鍔涘浘骞朵繚瀛?
# ==========================================
# 鐢变簬涓婇潰鐢ㄤ簡 matplotlib.use('Agg')锛宲lt.show() 鍙兘涓嶄細寮圭獥
# 鎵€浠ュ己鍒朵紶鍏?save_path 鏉ヤ繚瀛樺浘鐗?
plot_vehicle_battery_heatmap(
    res_dic['battery_log'],
    title="Battery Heatmap",
    save_path="battery_heatmap.png" # 鏄庣‘鎸囧畾淇濆瓨璺緞
)
print("鐢垫睜鐑姏鍥惧凡淇濆瓨: battery_heatmap.png")


# 娴嬭瘯
res_dic = dclilytrain.test(cfg, env, agent)
plot_rewards(res_dic['rewards'], cfg, tag="test")  # 鐢诲嚭缁撴灉
#plot_vehicle_battery(np.array(res_dic['battery_log'][0]), cfg)

# (杩欎竴瓒磋矊浼兼槸鎴戜笉灏忓績澶氬姞杩涘幓鐨勶紝閲嶅浜嗭級
#res_dic = dclilytrain.test(cfg, env, agent)
#plot_rewards(res_dic['rewards'], cfg, tag="test")




