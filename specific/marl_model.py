# -*- coding: utf-8 -*-
"""
MARL Phase-1: Multi-Agent RL Models
Created on 2026-02-19

参数共享的Actor和Centralized Critic
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =========2026.3.5修改，方案2：添加全局上下文编码器==========
class GlobalContextEncoder(nn.Module):
    """
    全局上下文编码器
    将高维全局状态(251维)压缩为紧凑的上下文向量(32维)
    """
    def __init__(self, global_state_dim=251, context_dim=32):
        """
        Args:
            global_state_dim: 全局状态维度，默认251
            context_dim: 压缩后的上下文维度，默认32
        """
        super(GlobalContextEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(global_state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, context_dim)
        )

    def forward(self, global_state):
        """
        Args:
            global_state: tensor, shape (batch_size, global_state_dim)

        Returns:
            context: tensor, shape (batch_size, context_dim)
        """
        return self.encoder(global_state)
# ======修改结束

# =========2026.2.19修改，MARL Phase-1：参数共享Actor==========
class SharedActor(nn.Module):
    """
    参数共享的Actor网络（方案2：添加全局上下文）

    输入：局部观测 + 全局上下文 + agent_id_embedding
    输出：3维连续动作 [depart_pref, dest_pref, charge_pref]
    """
    def __init__(self, obs_dim=6, n_agents=50, agent_embed_dim=16,
                 hidden_dim=128, action_dim=3, global_state_dim=251,
                 context_dim=32, init_w=3e-3):
        """
        Args:
            obs_dim: 局部观测维度（不含agent_id），默认6
            n_agents: 智能体数量（区域数），默认50
            agent_embed_dim: agent_id embedding维度，默认16
            hidden_dim: 隐藏层维度，默认128
            action_dim: 动作维度，默认3
            global_state_dim: 全局状态维度，默认251
            context_dim: 全局上下文维度，默认32
        """
        super(SharedActor, self).__init__()

        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.agent_embed_dim = agent_embed_dim
        self.context_dim = context_dim

        # =========2026.3.5修改，方案2：添加全局上下文编码器==========
        # 全局上下文编码器
        self.global_encoder = GlobalContextEncoder(global_state_dim, context_dim)
        # ======修改结束

        # Agent ID embedding
        self.agent_embedding = nn.Embedding(n_agents, agent_embed_dim)

        # =========2026.3.5修改，方案2：输入维度增加context_dim==========
        # MLP: (obs_dim + context_dim + agent_embed_dim) -> hidden_dim -> hidden_dim -> action_dim
        input_dim = obs_dim + context_dim + agent_embed_dim
        # ======修改结束
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, action_dim)

        # 初始化权重
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, agent_ids, global_state=None):
        """
        Args:
            obs: tensor, shape (batch_size, obs_dim) - 局部观测（不含agent_id）
            agent_ids: tensor, shape (batch_size,) - agent编号
            global_state: tensor, shape (batch_size, global_state_dim) - 全局状态（可选）

        Returns:
            action: tensor, shape (batch_size, action_dim) - 动作输出 [0,1]
        """
        # Agent ID embedding
        agent_embed = self.agent_embedding(agent_ids)  # (batch_size, agent_embed_dim)

        # =========2026.3.5修改，方案2：添加全局上下文==========
        if global_state is not None:
            # 提取全局上下文
            global_context = self.global_encoder(global_state)  # (batch_size, context_dim)
            # 拼接观测、全局上下文和embedding
            x = torch.cat([obs, global_context, agent_embed], dim=-1)
        else:
            # 向后兼容：如果没有全局状态，用零向量填充
            batch_size = obs.shape[0]
            zero_context = torch.zeros(batch_size, self.context_dim, device=obs.device)
            x = torch.cat([obs, zero_context, agent_embed], dim=-1)
        # ======修改结束

        # MLP前向传播
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = torch.sigmoid(self.fc3(x))  # 输出到[0,1]

        return action
# ======修改结束

# =========2026.2.19修改，MARL Phase-1：Centralized Critic==========
class CentralizedCritic(nn.Module):
    """
    集中式Critic网络

    输入：global_state + joint_action
    输出：Q值（标量）
    """
    def __init__(self, global_state_dim=251, n_agents=50, action_dim=3,
                 hidden_dim=256, init_w=3e-3):
        """
        Args:
            global_state_dim: 全局状态维度，默认251（SIP数据集）
            n_agents: 智能体数量，默认50
            action_dim: 每个智能体的动作维度，默认3
            hidden_dim: 隐藏层维度，默认256
        """
        super(CentralizedCritic, self).__init__()

        self.global_state_dim = global_state_dim
        self.n_agents = n_agents
        self.action_dim = action_dim

        # 输入：global_state + joint_action (n_agents * action_dim)
        input_dim = global_state_dim + n_agents * action_dim

        # MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        # 初始化权重
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, global_state, joint_action):
        """
        Args:
            global_state: tensor, shape (batch_size, global_state_dim)
            joint_action: tensor, shape (batch_size, n_agents * action_dim)

        Returns:
            q_value: tensor, shape (batch_size, 1)
        """
        # 拼接全局状态和联合动作
        x = torch.cat([global_state, joint_action], dim=-1)

        # MLP前向传播
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)

        return q_value
# ======修改结束
