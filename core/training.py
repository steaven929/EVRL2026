# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 19:24:55 2024

@author: TLab
"""

import numpy as np
import random

class OUNoise(object):
    '''Ornstein–Uhlenbeck噪声
    '''
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.12, min_sigma=0.01, decay_period=15000):
        self.mu           = mu # OU噪声的参数
        self.theta        = theta # OU噪声的参数
        self.sigma        = max_sigma # OU噪声的参数
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.n_actions    = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
    def reset(self):
        self.obs = np.ones(self.n_actions) * self.mu
    def evolve_obs(self):
        x  = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.n_actions)
        self.obs = x + dx
        return self.obs
    def get_action(self, action, t=0):
        ou_obs = self.evolve_obs()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period) # sigma会逐渐衰减
        if random.random()>0.99:
            print(ou_obs)
        return np.clip(action + ou_obs, self.low, self.high) # 动作加上噪声后进行剪切
def encode_custom_action(zone_to_counts, zone_from_counts=None, charge_pref=None, zone_num=4):
    """
    按照车辆调度分布生成动作向量
    :param zone_to_counts: 每个区域作为目的地的车辆数量 [zone0, zone1, ...]
    :param zone_from_counts: 每个区域作为出发地的车辆数量 [zone0, zone1, ...]，如果为空，则均匀分配
    :param charge_pref: 充电倾向分布（zone级别），如果为空，默认等于目的地分布
    :return: 长度为 3 * zone_num 的动作向量
    """
    def normalize(arr):
        total = sum(arr)
        return [x / total if total > 0 else 0 for x in arr]

    if zone_from_counts is None:
        zone_from_counts = [1] * zone_num  # 均匀分布
    if charge_pref is None:
        charge_pref = zone_to_counts  # 默认充电倾向 = 去哪充哪

    vec = normalize(zone_from_counts) + normalize(zone_to_counts) + normalize(charge_pref)
    return np.array(vec, dtype=np.float32)



def train(cfg, env, agent):
    print("开始训练！")
    ou_noise = OUNoise(env.action_space)  # 动作噪声
    rewards = [] # 记录所有回合的奖励
    critic1_losses, critic2_losses, actor_losses = [], [], []
    q_values, target_q_values = [], []  # 修改：记录Q值和目标Q值用于诊断
    rld_history, svc_history, srd_history = [], [], []  # 修改

    battery_log = []  # 修改：在所有episode之前初始化，用于存储所有episode的电池数据  # 修改

    # =========2026.2.7修改：保存探索期动作==========
    exploration_actions_saved = False  # 标记是否已保存探索期动作
    # ======修改结束

    for i_ep in range(cfg['train_eps']):
        env.set_training_mode(1)
        # =========2026.2.7修改，方案D2：设置当前epoch==========
        env.set_epoch(i_ep)  # 设置当前轮数，用于探索策略衰减

        # 在探索期（第10轮）保存详细动作
        if i_ep == 10 and not exploration_actions_saved:
            env.set_training_mode(2)  # 设置为探索期记录模式
            exploration_actions_saved = True
        # ======修改结束
        state = env.reset()
        ou_noise.reset()
        ep_reward = 0
        current_ep_battery_log = []  # 修改：每个episode初始化当前episode的电池日志  # 修改

        for i_step in range(cfg['max_steps']):
            action = agent.sample_action(state)
            # print("无噪声动作")
            # print(action)
            zone_to_counts = [5, 5, 4, 4]  # zone 0 ~ 3
            # 出发区域可以不指定，默认为均匀
            #action = encode_custom_action(zone_to_counts)  #0723修改，用来自定义action 来测试
            action = ou_noise.get_action(action, i_step+1)

            # =========2026.2.7修改，方案D2：充电维度噪声增强==========
            # 前20轮训练，对充电维度 [8:12] 添加额外噪声
            if i_ep < 20:
                charging_noise = np.random.uniform(-0.3, 0.3, size=4)
                action[8:12] = np.clip(action[8:12] + charging_noise, 0, 1)
            # ======修改结束

            next_state, reward, done, info = env.step(action)  # 修改：移除epoch参数
            ep_reward += reward

            # 记录环境指标  # 修改
            if 'rld' in info:  # 修改
                rld_history.append(info['rld'])  # 修改
            if 'svc' in info:  # 修改
                svc_history.append(info['svc'])  # 修改
            if 'srd' in info:  # 修改
                srd_history.append(info['srd'])  # 修改

            battery_snapshot = [car[1] for car in env.car]  # 20250726添加，画图1用 # 每步记录所有车辆电量
            current_ep_battery_log.append(battery_snapshot) # 20250726添加，画图1用

            agent.memory.push((state, action, reward, next_state, done))
            #agent.update()  #2026.1.22修改
            loss_info = agent.update()  # 让它返回 dict
            if loss_info is not None:
                if 'critic1_loss' in loss_info: critic1_losses.append(loss_info['critic1_loss'])
                if 'critic2_loss' in loss_info: critic2_losses.append(loss_info['critic2_loss'])
                if 'actor_loss' in loss_info:   actor_losses.append(loss_info['actor_loss'])
                if 'q_value' in loss_info:      q_values.append(loss_info['q_value'])  # 修改
                if 'target_q' in loss_info:     target_q_values.append(loss_info['target_q'])  # 修改
            state = next_state

            if done:
                break
        if 1: #(i_ep+1)%1 == 0:
            print(f"回合：{i_ep+1}/{cfg['train_eps']}，奖励：{ep_reward:.2f}")
        rewards.append(ep_reward)

        # =========2026.2.13添加：每个episode后更新学习率==========
        agent.step_schedulers()
        # ======添加结束

        battery_log.extend(current_ep_battery_log)  # 修改：将当前episode的电池数据追加到总列表  # 修改
    print("完成训练！")
    return {'rewards': rewards,
    'battery_log': battery_log,
    'critic1_losses': critic1_losses,
    'critic2_losses': critic2_losses,
    'actor_losses': actor_losses,
    'q_values': q_values,  # 修改
    'target_q_values': target_q_values,  # 修改
    'rld_history': rld_history,  # 修改
    'svc_history': svc_history,  # 修改
    'srd_history': srd_history}  # 修改

def test(cfg, env, agent):
    print("开始测试！")
    rewards = [] # 记录所有回合的奖励
    for i_ep in range(cfg['test_eps']):
        env.set_training_mode(0)
        state = env.reset() 
        ep_reward = 0
        for i_step in range(cfg['max_steps']):
            action = agent.predict_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state
            if done:
                break
        rewards.append(ep_reward)
        print(f"回合：{i_ep+1}/{cfg['test_eps']}，奖励：{ep_reward:.2f}")
    print("完成测试！")
    return {'rewards':rewards}