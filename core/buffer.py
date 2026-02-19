# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 19:22:32 2024

@author: TLab

buffer：
经验池，批量学习必备模块；包括：
初始化
加入新的transition
采样
清空

"""

from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
    def push(self,transitions):
        self.buffer.append(transitions)
    def sample(self, batch_size: int, sequential: bool = False):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if sequential: # sequential sampling
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
            return zip(*batch)
        else:
            batch = random.sample(self.buffer, batch_size)
            return zip(*batch)
    def clear(self):
        self.buffer.clear()
    def __len__(self):
        return len(self.buffer)

# =========2026.2.19修改，MARL Phase-1：添加多智能体经验池==========
class MARLReplayBuffer:
    """
    多智能体经验池，支持joint-transition存储

    Transition结构：
        (obs_n, act_n, global_state, reward, next_obs_n, next_global_state, done)

    其中：
        obs_n: [n_agents, obs_dim] - 所有智能体的局部观测
        act_n: [n_agents, act_dim] - 所有智能体的动作
        global_state: [global_state_dim] - 全局状态（用于centralized critic）
        reward: float - 全局奖励（Phase-1共享给所有agent）
        next_obs_n: [n_agents, obs_dim] - 下一时刻所有智能体的局部观测
        next_global_state: [global_state_dim] - 下一时刻全局状态
        done: bool - 回合结束标志
    """
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def push(self, obs_n, act_n, global_state, reward, next_obs_n, next_global_state, done):
        """
        存储一条joint-transition

        Args:
            obs_n: numpy array, shape (n_agents, obs_dim)
            act_n: numpy array, shape (n_agents, act_dim)
            global_state: numpy array, shape (global_state_dim,)
            reward: float
            next_obs_n: numpy array, shape (n_agents, obs_dim)
            next_global_state: numpy array, shape (global_state_dim,)
            done: bool
        """
        transition = (obs_n, act_n, global_state, reward, next_obs_n, next_global_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size: int, sequential: bool = False):
        """
        采样一批joint-transitions

        Returns:
            tuple of (obs_n_batch, act_n_batch, global_state_batch, reward_batch,
                     next_obs_n_batch, next_global_state_batch, done_batch)
        """
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)

        if sequential:
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            batch = random.sample(self.buffer, batch_size)

        return zip(*batch)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
# ======修改结束