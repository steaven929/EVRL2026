# -*- coding: utf-8 -*-
"""
MARL Phase-1: Multi-Agent RL Agent (IPPO with Parameter Sharing)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from specific.marl_model import SharedActor, CentralizedCritic
from core.buffer import MARLReplayBuffer


class IPPOAgent:
    """
    Shared-actor MARL agent with a single centralized critic.

    This rollback version intentionally matches the earlier 200-car-era
    training dynamics while still allowing the actor to consume global state
    context through the shared actor network.
    """

    def __init__(
        self,
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
        device="cpu",
    ):
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.actor = SharedActor(
            obs_dim=obs_dim,
            n_agents=n_agents,
            agent_embed_dim=agent_embed_dim,
            hidden_dim=actor_hidden_dim,
            action_dim=action_dim,
            global_state_dim=global_state_dim,
            context_dim=32,
        ).to(self.device)

        self.critic = CentralizedCritic(
            global_state_dim=global_state_dim,
            n_agents=n_agents,
            action_dim=action_dim,
            hidden_dim=critic_hidden_dim,
        ).to(self.device)

        self.target_critic = CentralizedCritic(
            global_state_dim=global_state_dim,
            n_agents=n_agents,
            action_dim=action_dim,
            hidden_dim=critic_hidden_dim,
        ).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.memory = MARLReplayBuffer(memory_capacity)
        self.critic_criterion = nn.SmoothL1Loss()

    def sample_action(self, obs_n, global_state=None, add_noise=False, noise_scale=0.1):
        obs = obs_n[:, :self.obs_dim]
        agent_ids = obs_n[:, self.obs_dim].astype(np.int64)

        obs_tensor = torch.FloatTensor(obs).to(self.device)
        agent_ids_tensor = torch.LongTensor(agent_ids).to(self.device)

        global_state_tensor = None
        if global_state is not None:
            global_state_tensor = (
                torch.FloatTensor(global_state)
                .unsqueeze(0)
                .repeat(self.n_agents, 1)
                .to(self.device)
            )

        with torch.no_grad():
            actions = self.actor(obs_tensor, agent_ids_tensor, global_state_tensor)
        actions = actions.cpu().numpy()

        if add_noise:
            noise = np.random.normal(0, noise_scale, size=actions.shape)
            actions = np.clip(actions + noise, 0.0, 1.0)

        return actions

    def update(self):
        if len(self.memory) < self.batch_size:
            return None, None

        (
            obs_n_batch,
            act_n_batch,
            global_state_batch,
            reward_batch,
            next_obs_n_batch,
            next_global_state_batch,
            done_batch,
        ) = self.memory.sample(self.batch_size)

        obs_n_batch = torch.FloatTensor(np.array(obs_n_batch)).to(self.device)
        act_n_batch = torch.FloatTensor(np.array(act_n_batch)).to(self.device)
        global_state_batch = torch.FloatTensor(np.array(global_state_batch)).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_obs_n_batch = torch.FloatTensor(np.array(next_obs_n_batch)).to(self.device)
        next_global_state_batch = torch.FloatTensor(np.array(next_global_state_batch)).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)

        joint_action_batch = act_n_batch.reshape(act_n_batch.shape[0], -1)

        with torch.no_grad():
            next_obs = next_obs_n_batch[:, :, :self.obs_dim]
            next_agent_ids = next_obs_n_batch[:, :, self.obs_dim].long()

            next_actions_list = []
            for i in range(self.n_agents):
                next_action_i = self.actor(
                    next_obs[:, i, :],
                    next_agent_ids[:, i],
                    next_global_state_batch,
                )
                next_actions_list.append(next_action_i)

            next_joint_action = torch.stack(next_actions_list, dim=1)
            next_joint_action_flat = next_joint_action.reshape(next_joint_action.shape[0], -1)
            target_q = self.target_critic(next_global_state_batch, next_joint_action_flat)
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_q

        current_q = self.critic(global_state_batch, joint_action_batch)
        critic_loss = self.critic_criterion(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        obs = obs_n_batch[:, :, :self.obs_dim]
        agent_ids = obs_n_batch[:, :, self.obs_dim].long()

        actions_list = []
        for i in range(self.n_agents):
            action_i = self.actor(obs[:, i, :], agent_ids[:, i], global_state_batch)
            actions_list.append(action_i)

        joint_action = torch.stack(actions_list, dim=1)
        joint_action_flat = joint_action.reshape(joint_action.shape[0], -1)

        actor_loss = -self.critic(global_state_batch, joint_action_flat).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        self._soft_update(self.critic, self.target_critic)
        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save_model(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
