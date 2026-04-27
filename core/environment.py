# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 19:26:35 2024

@author: TLab
"""


import os
import torch
import numpy as np
import random
import gym

from specific.model import Actor, Critic
from core.algorithm import DDPG, SAC, TD3
from core.buffer import ReplayBuffer
# =========2026.2.19修改，MARL Phase-1：导入MARL组件==========
from core.marl_agent import IPPOAgent
# ======修改结束


class NormalizedActions(gym.ActionWrapper):
    ''' 将action范围重定在[0.1]之间，貌似不需要
    '''
    def action(self, action):
        # low_bound   = self.action_space.low
        # upper_bound = self.action_space.high
        # action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        # action = np.clip(action, low_bound, upper_bound)
        return action


# =========2026.2.20修改，MARL适配器：联合动作与局部观测==========
class MARLAdapter(gym.Wrapper):
    """将单智能体环境包装为MARL接口。"""

    def __init__(self, env, n_agents):
        super().__init__(env)
        self.n_agents = int(n_agents)
        self._last_global_state = None

    def reset(self, **kwargs):
        reset_ret = self.env.reset(**kwargs)
        global_state = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
        self._last_global_state = np.asarray(global_state)
        return self._last_global_state

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        expected_shape = (self.n_agents, 3)
        if actions.shape != expected_shape:
            raise ValueError(f"MARL actions shape {actions.shape} != {expected_shape}")

        actions = np.clip(actions, 0.0, 1.0)
        flat_action = np.concatenate([actions[:, 0], actions[:, 1], actions[:, 2]], axis=0)

        step_ret = self.env.step(flat_action)
        if len(step_ret) == 5:
            global_state, reward, terminated, truncated, info = step_ret
            done = bool(terminated or truncated)
        else:
            global_state, reward, done, info = step_ret

        self._last_global_state = np.asarray(global_state)
        return self._last_global_state, float(reward), bool(done), info

    def get_agent_obs(self, agent_id):
        if self._last_global_state is None:
            raise RuntimeError("Call reset() before get_agent_obs().")

        if agent_id < 0 or agent_id >= self.n_agents:
            raise ValueError(f"agent_id {agent_id} out of range [0, {self.n_agents})")

        state = np.asarray(self._last_global_state, dtype=np.float32)
        z = self.n_agents
        expected_min_dim = 5 * z + 1
        if state.shape[0] < expected_min_dim:
            raise ValueError(f"global_state dim {state.shape[0]} < required {expected_min_dim}")

        local_obs = np.array([
            state[agent_id],
            state[z + agent_id],
            state[2 * z + agent_id],
            state[3 * z + agent_id],
            state[4 * z + agent_id],
            state[-1],
        ], dtype=np.float32)
        return np.concatenate([local_obs, np.array([float(agent_id)], dtype=np.float32)], axis=0)
# ======修改结束

    # def reverse_action(self, action):
    #     low_bound   = self.action_space.low
    #     upper_bound = self.action_space.high
    #     action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
    #     action = np.clip(action, low_bound, upper_bound)
    #     return action
    
def all_seed(env,seed = 1):
    ''' 万能的seed函数
    '''
    env.seed(seed) # env config
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # config for CPU
    torch.cuda.manual_seed(seed) # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # config for python scripts
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
def env_agent_config(cfg):
    env = NormalizedActions(gym.make(cfg['env_name'])) # 装饰action噪声
    if cfg['seed'] !=0:
        all_seed(env,seed=cfg['seed'])
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    state_vocab_size = 64
    if hasattr(env.observation_space, "nvec"):
        state_vocab_size = int(np.max(env.observation_space.nvec))
    cfg.update({"n_states":n_states,"n_actions":n_actions, "state_vocab_size": state_vocab_size}) # ??n_states?n_actions?cfg???

    # =========2026.2.19修改，MARL Phase-1：支持multi模式==========
    marl_mode = cfg.get('marl_mode', 'single')  # 默认single模式

    if marl_mode == 'multi':
        inferred_agents = n_actions // 3
        if inferred_agents <= 0:
            raise ValueError(f"Invalid action dim for MARL: {n_actions}")
        cfg['n_agents'] = int(cfg.get('n_agents', inferred_agents))
        if cfg['n_agents'] != inferred_agents:
            print(f"[MARL] n_agents from cfg ({cfg['n_agents']}) != inferred ({inferred_agents}), use inferred.")
            cfg['n_agents'] = inferred_agents

        env = MARLAdapter(env, n_agents=cfg['n_agents'])

        # 多智能体模式：IPPO with Parameter Sharing
        agent = IPPOAgent(
            obs_dim=6,  # 局部观测维度（不含agent_id）
            n_agents=cfg.get('n_agents', 50),  # 智能体数量（区域数）
            action_dim=3,  # 每个智能体的动作维度
            global_state_dim=n_states,  # 全局状态维度
            agent_embed_dim=16,  # agent_id embedding维度
            actor_hidden_dim=cfg.get('actor_hidden_dim', 128),
            critic_hidden_dim=cfg.get('critic_hidden_dim', 256),
            actor_lr=cfg['actor_lr'],
            critic_lr=cfg['critic_lr'],
            gamma=cfg['gamma'],
            tau=cfg['tau'],
            batch_size=cfg['batch_size'],
            memory_capacity=cfg['memory_capacity'],
            device=cfg['device']
        )
        return env, agent
    # ======修改结束

    # 根据算法名称选择不同的agent（单智能体模式）
    if cfg['algo_name'] == 'SAC':
        memory = ReplayBuffer(cfg['memory_capacity'])
        # =========2026.2.19修改，增加vocab_size以支持SIP数据集==========
        # SIP数据集：OD_MAX=13, TIME_NUM=26
        # vocab_size需要 >= max(OD_MAX, TIME_NUM) = 26
        agent = SAC(
            n_states=n_states,
            n_actions=n_actions,
            memory=memory,
            vocab_size=30,  # 从12增加到30，支持SIP数据集（OD_MAX=13, TIME_NUM=26）
            # ======修改结束
            embedding_dims=cfg['actor_embed_dim'],
            hidden_dim=cfg['actor_hidden_dim'],
            use_dual_head=cfg.get('use_dual_head', False),  # 是否使用双头架构
            actor_lr=cfg['actor_lr'],
            critic_lr=cfg['critic_lr'],
            alpha_lr=cfg.get('sac_alpha_lr', 3e-4),
            gamma=cfg['gamma'],
            tau=cfg['tau'],
            alpha=cfg.get('sac_alpha', 0.2),
            batch_size=cfg['batch_size'],
            device=cfg['device']
        )
    elif cfg['algo_name'] == 'TD3':
        models = {
            "actor": Actor(n_states, n_actions, embedding_dims=cfg['actor_embed_dim'], hidden_dim=cfg['actor_hidden_dim'], vocab_size_override=cfg['state_vocab_size']),
            "critic": Critic(n_states, n_actions, embedding_dims=cfg['actor_embed_dim'], hidden_dim=cfg['critic_hidden_dim'], vocab_size_override=cfg['state_vocab_size']),
            "critic2": Critic(n_states, n_actions, embedding_dims=cfg['actor_embed_dim'], hidden_dim=cfg['critic_hidden_dim'], vocab_size_override=cfg['state_vocab_size']),
        }
        memories = {"memory": ReplayBuffer(cfg['memory_capacity'])}
        agent = TD3(models, memories, cfg)
    else:  # DDPG
        models = {
            "actor": Actor(n_states, n_actions, embedding_dims=cfg['actor_embed_dim'], hidden_dim=cfg['actor_hidden_dim'], vocab_size_override=cfg['state_vocab_size']),
            "critic": Critic(n_states, n_actions, embedding_dims=cfg['actor_embed_dim'], hidden_dim=cfg['critic_hidden_dim'], vocab_size_override=cfg['state_vocab_size']),
        }
        memories = {"memory":ReplayBuffer(cfg['memory_capacity'])}
        agent = DDPG(models,memories,cfg)

    return env,agent

#--
#--------------------5.11
import numpy as np


def manual_action_test(env, custom_action=None):
    print("\n=== 手动动作测试开始 ===")

    obs = env.reset()

    # 从动作空间推断 ZONE_NUM
    Z = env.action_space.shape[0] // 3

    # 如果未指定动作，则构造一个有代表性的动作
    if custom_action is None:
        custom_action = np.array([
            1.0, 0.0, 0.0, 0.0,  # 出发倾向：只从区域0发车
            0.0, 1.0, 0.0, 0.0,  # 目的地倾向：送去区域1
            1.0, 0.0, 0.0, 0.0  # 充电倾向：偏好去区域1充电
        ])

    print("\n[Manual Action Input]")
    print("出发倾向:", custom_action[0:Z])
    print("目的地倾向:", custom_action[Z:2 * Z])
    print("充电倾向:", custom_action[2 * Z:3 * Z])

    # 执行一步调度
    obs, reward, done, _ = env.step(custom_action)

    print(f"\n[Step 0 Completed] Reward = {reward:.2f}, Terminal = {done}")
    print("=== 手动动作测试结束 ===\n")


def test_handle_vehicle_charging(self):
    # 假设车辆编号为0
    vehicle_id = 0
    self.car = [[0, 100, 0]]  # 当前位置0，电量100，当前无任务

    self.dist = np.array([
        [0, 10, 20, 30],
        [10, 0, 15, 25],
        [20, 15, 0, 10],
        [30, 25, 10, 0]
    ])

    self.available = np.array([
        [1, 1, 0, 0],  # 当前泊位占用情况
        [2, 2, 2, 2],  # 多余行：用于pre_avail对比
    ])

    self.mem_available = np.array([
        [2, 2, 2, 2]  # 总容量
    ])

    global CHARGER_ZONES
    CHARGER_ZONES = [1, 2]  # 假设1、2区有充电桩

    # case: 目标区域 1 有桩有位，应直接派往
    print("🚗 车辆前往区域 1")
    result = self.handle_vehicle_charging(vehicle_id, 1)
    print(f"结果目标区域: {result}, 车辆状态: {self.car[vehicle_id]}")

    # case: 目标区域 2 没有位，应寻找替代区
    self.available[0][2] = 2  # 已满
    print("🚗 车辆尝试前往已满的区域 2")
    result = self.handle_vehicle_charging(vehicle_id, 2)
    print(f"结果目标区域: {result}, 车辆状态: {self.car[vehicle_id]}")

    # case: 当前电量不足以到达目标
    self.car[vehicle_id][1] = 5  # 电量过低
    print("🚗 电量不足，前往区域 3（无充电桩）")
    result = self.handle_vehicle_charging(vehicle_id, 3)
    print(f"结果目标区域: {result}, 车辆状态: {self.car[vehicle_id]}")

