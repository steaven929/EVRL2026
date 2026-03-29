# -*- coding: utf-8 -*-
"""
MARL agent with shared actor and centralized twin critics (MATD3-style updates).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from specific.marl_model import SharedActor, CentralizedCritic
from core.buffer import MARLReplayBuffer


class IPPOAgent:
    """
    Multi-agent actor-critic with:
    - Shared actor
    - Centralized twin critics
    - Delayed actor update
    - Target policy smoothing
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
        policy_delay=2,
        target_policy_noise=0.08,
        target_noise_clip=0.15,
        device='cpu',
    ):
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_delay = max(1, int(policy_delay))
        self.target_policy_noise = max(0.0, float(target_policy_noise))
        self.target_noise_clip = max(0.0, float(target_noise_clip))
        self.total_updates = 0
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
        self.critic2 = CentralizedCritic(
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
        self.target_critic2 = CentralizedCritic(
            global_state_dim=global_state_dim,
            n_agents=n_agents,
            action_dim=action_dim,
            hidden_dim=critic_hidden_dim,
        ).to(self.device)

        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

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

    def _build_joint_action(self, obs_batch, agent_ids_batch, global_state_batch):
        actions_list = []
        for i in range(self.n_agents):
            action_i = self.actor(obs_batch[:, i, :], agent_ids_batch[:, i], global_state_batch)
            actions_list.append(action_i)
        joint_action = torch.stack(actions_list, dim=1)
        return joint_action.reshape(joint_action.shape[0], -1)

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
        self.total_updates += 1

        with torch.no_grad():
            next_obs = next_obs_n_batch[:, :, :self.obs_dim]
            next_agent_ids = next_obs_n_batch[:, :, self.obs_dim].long()

            next_actions_list = []
            for i in range(self.n_agents):
                next_action_i = self.actor(next_obs[:, i, :], next_agent_ids[:, i], next_global_state_batch)
                if self.target_policy_noise > 0:
                    noise = torch.randn_like(next_action_i) * self.target_policy_noise
                    noise = torch.clamp(noise, -self.target_noise_clip, self.target_noise_clip)
                    next_action_i = torch.clamp(next_action_i + noise, 0.0, 1.0)
                next_actions_list.append(next_action_i)

            next_joint_action = torch.stack(next_actions_list, dim=1)
            next_joint_action_flat = next_joint_action.reshape(next_joint_action.shape[0], -1)

            target_q1 = self.target_critic(next_global_state_batch, next_joint_action_flat)
            target_q2 = self.target_critic2(next_global_state_batch, next_joint_action_flat)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_q

        current_q1 = self.critic(global_state_batch, joint_action_batch)
        current_q2 = self.critic2(global_state_batch, joint_action_batch)

        critic_loss1 = self.critic_criterion(current_q1, target_q)
        critic_loss2 = self.critic_criterion(current_q2, target_q)
        critic_loss = critic_loss1 + critic_loss2

        self.critic_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        self.critic2_optimizer.step()

        actor_loss = None
        if self.total_updates % self.policy_delay == 0:
            obs = obs_n_batch[:, :, :self.obs_dim]
            agent_ids = obs_n_batch[:, :, self.obs_dim].long()
            joint_action_flat = self._build_joint_action(obs, agent_ids, global_state_batch)

            actor_loss = -self.critic(global_state_batch, joint_action_flat).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            self._soft_update(self.critic, self.target_critic)
            self._soft_update(self.critic2, self.target_critic2)

        return critic_loss.item(), (actor_loss.item() if actor_loss is not None else None)

    def _soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save_model(self, path):
        torch.save(
            {
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'critic2': self.critic2.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'target_critic2': self.target_critic2.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'critic2_optimizer': self.critic2_optimizer.state_dict(),
                'total_updates': self.total_updates,
            },
            path,
        )

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

        if 'critic2' in checkpoint:
            self.critic2.load_state_dict(checkpoint['critic2'])
        if 'target_critic' in checkpoint:
            self.target_critic.load_state_dict(checkpoint['target_critic'])
        else:
            self.target_critic.load_state_dict(self.critic.state_dict())

        if 'target_critic2' in checkpoint:
            self.target_critic2.load_state_dict(checkpoint['target_critic2'])
        else:
            self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if 'critic2_optimizer' in checkpoint:
            self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])

        self.total_updates = int(checkpoint.get('total_updates', 0))
