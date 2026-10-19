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

# =========2026.2.19修改，MARL Phase-1：参数共享Actor==========
class SharedActor(nn.Module):
    """
    参数共享的Actor网络

    输入：局部观测（去掉agent_id）+ agent_id_embedding
    输出：3维连续动作 [depart_pref, dest_pref, charge_pref]
    """
    def __init__(self, obs_dim=6, n_agents=50, agent_embed_dim=16,
                 hidden_dim=128, action_dim=3, init_w=3e-3,
                 use_zone_embedding=True):
        """
        Args:
            obs_dim: 局部观测维度（不含agent_id），默认6
            n_agents: 智能体数量（区域数），默认50
            agent_embed_dim: agent_id embedding维度，默认16
            hidden_dim: 隐藏层维度，默认128
            action_dim: 动作维度，默认3
        """
        super(SharedActor, self).__init__()

        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.agent_embed_dim = agent_embed_dim
        self.use_zone_embedding = bool(use_zone_embedding)

        # Agent ID embedding
        self.agent_embedding = nn.Embedding(n_agents, agent_embed_dim)

        # MLP: (obs_dim + agent_embed_dim) -> hidden_dim -> hidden_dim -> action_dim
        input_dim = obs_dim + agent_embed_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, action_dim)

        # 初始化权重
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, agent_ids):
        """
        Args:
            obs: tensor, shape (batch_size, obs_dim) - 局部观测（不含agent_id）
            agent_ids: tensor, shape (batch_size,) - agent编号

        Returns:
            action: tensor, shape (batch_size, action_dim) - 动作输出 [0,1]
        """
        # Agent ID embedding
        agent_embed = self.agent_embedding(agent_ids)  # (batch_size, agent_embed_dim)
        if not self.use_zone_embedding:
            # 消融时只屏蔽区域身份，仍保留历史 6+16=22 维输入。
            agent_embed = torch.zeros_like(agent_embed)

        # 拼接观测和embedding
        x = torch.cat([obs, agent_embed], dim=-1)  # (batch_size, obs_dim + agent_embed_dim)

        # MLP前向传播
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = torch.sigmoid(self.fc3(x))  # 输出到[0,1]

        return action
# ======修改结束


class IndependentActorEnsemble(nn.Module):
    """每个区域一个独立 Actor；各 Actor 的输入和输出维数与 V4.7 相同。"""

    def __init__(self, obs_dim=6, n_agents=50, agent_embed_dim=16,
                 hidden_dim=128, action_dim=3, init_w=3e-3,
                 use_zone_embedding=True):
        super().__init__()
        self.n_agents = int(n_agents)
        self.actors = nn.ModuleList([
            SharedActor(
                obs_dim=obs_dim,
                n_agents=n_agents,
                agent_embed_dim=agent_embed_dim,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                init_w=init_w,
                use_zone_embedding=use_zone_embedding,
            )
            for _ in range(self.n_agents)
        ])

    def forward(self, obs, agent_ids):
        outputs = []
        for row, agent_id in zip(obs, agent_ids):
            idx = int(agent_id.detach().item())
            outputs.append(self.actors[idx](row.unsqueeze(0), agent_id.reshape(1)).squeeze(0))
        return torch.stack(outputs, dim=0)

# =========2026.2.19修改，MARL Phase-1：Centralized Critic==========
class CentralizedCritic(nn.Module):
    """
    集中式Critic网络

    输入：global_state + joint_action
    输出：Q值（标量）
    """
    def __init__(self, global_state_dim=251, n_agents=50, action_dim=3,
                 hidden_dim=256, init_w=3e-3, use_global_context=True):
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
        self.use_global_context = bool(use_global_context)

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
        if not self.use_global_context:
            # 消融时屏蔽全局状态，Critic 输入层形状保持不变。
            global_state = torch.zeros_like(global_state)
        # 拼接全局状态和联合动作
        x = torch.cat([global_state, joint_action], dim=-1)

        # MLP前向传播
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)

        return q_value
# ======修改结束
