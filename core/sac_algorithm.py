import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =========2026.2.11，实验13：SAC算法实现==========

class SACActorNetwork(nn.Module):
    """SAC的Actor网络，输出动作的均值和标准差"""
    def __init__(self, n_states, n_actions, embedding_dims=8, hidden_dim=256, vocab_size=10):
        super(SACActorNetwork, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dims)
        self.linear1 = nn.Linear(n_states * embedding_dims, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        # 输出均值和log标准差
        self.mean_linear = nn.Linear(hidden_dim, n_actions)
        self.log_std_linear = nn.Linear(hidden_dim, n_actions)

        # 标准差的范围限制
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state):
        embedded_states = self.embedding(state)
        embedded_flattened = embedded_states.view(embedded_states.size(0), -1)

        x = F.relu(self.linear1(embedded_flattened))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        """采样动作"""
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # 重参数化技巧
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # 可微分采样

        # 使用tanh压缩到[-1, 1]，然后映射到[0, 1]
        action = torch.sigmoid(x_t)

        # 计算log概率
        log_prob = normal.log_prob(x_t)
        # 修正tanh变换的log概率
        log_prob -= torch.log(action * (1 - action) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, mean, log_std


class SACCriticNetwork(nn.Module):
    """SAC的Critic网络（Q网络）"""
    def __init__(self, n_states, n_actions, embedding_dims=8, hidden_dim=256, vocab_size=10):
        super(SACCriticNetwork, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dims)
        self.action_linear = nn.Linear(n_actions, embedding_dims * n_states)

        self.linear1 = nn.Linear(embedding_dims * n_states * 2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        embedded_states = self.embedding(state)
        embedded_flattened = embedded_states.view(embedded_states.size(0), -1)

        action_embedded = F.relu(self.action_linear(action))
        x = torch.cat([embedded_flattened, action_embedded], dim=1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        q_value = self.linear3(x)

        return q_value


class SAC:
    """SAC算法实现"""
    def __init__(self, n_states, n_actions, vocab_size=10,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.95, tau=0.005, alpha=0.2, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions

        # Actor网络
        self.actor = SACActorNetwork(n_states, n_actions, vocab_size=vocab_size).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # 两个Critic网络（Twin Q）
        self.critic1 = SACCriticNetwork(n_states, n_actions, vocab_size=vocab_size).to(device)
        self.critic2 = SACCriticNetwork(n_states, n_actions, vocab_size=vocab_size).to(device)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # 目标Critic网络
        self.target_critic1 = SACCriticNetwork(n_states, n_actions, vocab_size=vocab_size).to(device)
        self.target_critic2 = SACCriticNetwork(n_states, n_actions, vocab_size=vocab_size).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # 自动调整温度参数α
        self.target_entropy = -n_actions  # 目标熵
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = alpha
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    def select_action(self, state, evaluate=False):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).long().to(self.device)

        if evaluate:
            # 评估时使用确定性策略（均值）
            mean, _ = self.actor(state)
            action = torch.sigmoid(mean)
        else:
            # 训练时采样
            action, _, _, _ = self.actor.sample(state)

        return action.detach().cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=64):
        """更新网络"""
        # 从经验池采样
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).long().to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).long().to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # 更新Critic
        with torch.no_grad():
            next_action, next_log_prob, _, _ = self.actor.sample(next_state)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # 更新Actor
        new_action, log_prob, _, _ = self.actor.sample(state)
        q1 = self.critic1(state, new_action)
        q2 = self.critic2(state, new_action)
        q = torch.min(q1, q2)

        actor_loss = (self.alpha * log_prob - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新温度参数α
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().item()

        # 软更新目标网络
        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)

        return critic1_loss.item(), actor_loss.item(), self.alpha

    def soft_update(self, source, target):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self, filename):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'log_alpha': self.log_alpha,
        }, filename)

    def load(self, filename):
        """加载模型"""
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp().item()
