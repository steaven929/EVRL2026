# -*- coding: utf-8 -*-
"""
MARL Phase-1: Multi-Agent RL Agent (IPPO with Parameter Sharing)
Created on 2026-02-19

实现IPPO（Independent PPO）+ Centralized Critic
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from specific.marl_model import SharedActor, CentralizedCritic
from core.buffer import MARLReplayBuffer

# =========2026.2.19修改，MARL Phase-1：IPPO Agent==========
class IPPOAgent:
    """
    IPPO Agent with Parameter Sharing and Centralized Critic

    特点：
    - 参数共享的Actor（所有区域共享同一个网络）
    - 集中式Critic（使用全局状态）
    - 全局奖励共享（Phase-1不做credit assignment）
    """
    def __init__(self,
                 obs_dim=6,
                 n_agents=50,
                 action_dim=3,
                 global_state_dim=251,
                 agent_embed_dim=16,
                 actor_hidden_dim=128,
                 critic_hidden_dim=256,
                 actor_lr=2e-5,
                 critic_lr=3e-5,
                 gamma=0.95,
                 tau=0.0005,
                 batch_size=128,
                 memory_capacity=64000,
                 device='cpu'):
        """
        Args:
            obs_dim: 局部观测维度（不含agent_id），默认6
            n_agents: 智能体数量，默认50
            action_dim: 每个智能体的动作维度，默认3
            global_state_dim: 全局状态维度，默认251
            agent_embed_dim: agent_id embedding维度，默认16
            actor_hidden_dim: Actor隐藏层维度，默认128
            critic_hidden_dim: Critic隐藏层维度，默认256
            actor_lr: Actor学习率
            critic_lr: Critic学习率
            gamma: 折扣因子
            tau: 软更新系数
            batch_size: 批大小
            memory_capacity: 经验池容量
            device: 设备（cpu/cuda）
        """
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)

        # 创建网络
        self.actor = SharedActor(
            obs_dim=obs_dim,
            n_agents=n_agents,
            agent_embed_dim=agent_embed_dim,
            hidden_dim=actor_hidden_dim,
            action_dim=action_dim
        ).to(self.device)

        self.critic = CentralizedCritic(
            global_state_dim=global_state_dim,
            n_agents=n_agents,
            action_dim=action_dim,
            hidden_dim=critic_hidden_dim
        ).to(self.device)

        # Target networks
        self.target_critic = CentralizedCritic(
            global_state_dim=global_state_dim,
            n_agents=n_agents,
            action_dim=action_dim,
            hidden_dim=critic_hidden_dim
        ).to(self.device)

        # 初始化target网络
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 经验池
        self.memory = MARLReplayBuffer(memory_capacity)

        # 损失函数 - 使用Huber Loss对大误差不敏感（P0修复）
        self.critic_criterion = nn.SmoothL1Loss()  # Huber Loss

    def sample_action(self, obs_n, global_state=None, add_noise=False, noise_scale=0.1):
        """
        采样联合动作

        Args:
            obs_n: numpy array, shape (n_agents, obs_dim+1) - 所有智能体的局部观测（含agent_id）
            add_noise: 是否添加探索噪声
            noise_scale: 噪声尺度

        Returns:
            actions: numpy array, shape (n_agents, action_dim)
        """
        # 分离观测和agent_id
        obs = obs_n[:, :self.obs_dim]  # (n_agents, obs_dim)
        agent_ids = obs_n[:, self.obs_dim].astype(np.int64)  # (n_agents,)

        # 转换为tensor
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        agent_ids_tensor = torch.LongTensor(agent_ids).to(self.device)

        # 前向传播
        with torch.no_grad():
            actions = self.actor(obs_tensor, agent_ids_tensor)  # (n_agents, action_dim)

        # 转换为numpy
        actions = actions.cpu().numpy()

        # 添加探索噪声
        if add_noise:
            noise = np.random.normal(0, noise_scale, size=actions.shape)
            actions = np.clip(actions + noise, 0, 1)

        return actions

    def update(self):
        """
        更新网络参数
        """
        if len(self.memory) < self.batch_size:
            return None, None

        # 采样batch
        obs_n_batch, act_n_batch, global_state_batch, reward_batch, \
        next_obs_n_batch, next_global_state_batch, done_batch = self.memory.sample(self.batch_size)

        # 转换为tensor
        obs_n_batch = torch.FloatTensor(np.array(obs_n_batch)).to(self.device)  # (batch, n_agents, obs_dim+1)
        act_n_batch = torch.FloatTensor(np.array(act_n_batch)).to(self.device)  # (batch, n_agents, action_dim)
        global_state_batch = torch.FloatTensor(np.array(global_state_batch)).to(self.device)  # (batch, global_state_dim)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)  # (batch, 1)
        next_obs_n_batch = torch.FloatTensor(np.array(next_obs_n_batch)).to(self.device)  # (batch, n_agents, obs_dim+1)
        next_global_state_batch = torch.FloatTensor(np.array(next_global_state_batch)).to(self.device)  # (batch, global_state_dim)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)  # (batch, 1)

        # 展平joint action: (batch, n_agents, action_dim) -> (batch, n_agents * action_dim)
        joint_action_batch = act_n_batch.reshape(act_n_batch.shape[0], -1)

        # ===== 更新Critic =====
        # 计算target Q值
        with torch.no_grad():
            # 使用target actor采样next actions
            next_obs = next_obs_n_batch[:, :, :self.obs_dim]  # (batch, n_agents, obs_dim)
            next_agent_ids = next_obs_n_batch[:, :, self.obs_dim].long()  # (batch, n_agents)

            # 为每个agent采样next action
            next_actions_list = []
            for i in range(self.n_agents):
                next_action_i = self.actor(next_obs[:, i, :], next_agent_ids[:, i])  # (batch, action_dim)
                next_actions_list.append(next_action_i)
            next_joint_action = torch.stack(next_actions_list, dim=1)  # (batch, n_agents, action_dim)
            next_joint_action_flat = next_joint_action.reshape(next_joint_action.shape[0], -1)  # (batch, n_agents*action_dim)

            # 计算target Q
            target_q = self.target_critic(next_global_state_batch, next_joint_action_flat)
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_q

        # 计算current Q
        current_q = self.critic(global_state_batch, joint_action_batch)

        # Critic loss
        critic_loss = self.critic_criterion(current_q, target_q)

        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # P0修复：梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # ===== 更新Actor =====
        # 分离观测和agent_id
        obs = obs_n_batch[:, :, :self.obs_dim]  # (batch, n_agents, obs_dim)
        agent_ids = obs_n_batch[:, :, self.obs_dim].long()  # (batch, n_agents)

        # 为每个agent采样action
        actions_list = []
        for i in range(self.n_agents):
            action_i = self.actor(obs[:, i, :], agent_ids[:, i])  # (batch, action_dim)
            actions_list.append(action_i)
        joint_action = torch.stack(actions_list, dim=1)  # (batch, n_agents, action_dim)
        joint_action_flat = joint_action.reshape(joint_action.shape[0], -1)  # (batch, n_agents*action_dim)

        # Actor loss: -Q(s, a)
        actor_loss = -self.critic(global_state_batch, joint_action_flat).mean()

        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # P0修复：梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # ===== 软更新target网络 =====
        self._soft_update(self.critic, self.target_critic)

        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, source, target):
        """软更新target网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def save_model(self, path):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
# ======修改结束
