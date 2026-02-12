# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 19:23:20 2024

@author: TLab
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

class DDPG:
    def __init__(self, models, memories, cfg):
        self.device = torch.device(cfg['device'])
        self.critic = models['critic'].to(self.device)
        self.target_critic = models['critic'].to(self.device)
        self.actor = models['actor'].to(self.device)
        self.target_actor = models['actor'].to(self.device)
        
        # 复制参数到目标网络
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),  lr=cfg['critic_lr'])
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg['actor_lr'])
        self.memory = memories['memory']
        self.batch_size = cfg['batch_size']
        self.gamma = cfg['gamma']
        self.tau = cfg['tau']  # 软更新参数
    def sample_action(self, state):
        state = torch.LongTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]
    @torch.no_grad()
    def predict_action(self, state):
        ''' 用于预测，不需要计算梯度
        '''
        state = torch.LongTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        return action.cpu().numpy()[0]
    def update(self):
        if len(self.memory) < self.batch_size: # 当memory中不满足一个批量时，不更新策略
            return None  # 修改
        # 从经验回放中中随机采样一个批量的transition
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        # 转变为张量
        state = torch.LongTensor(np.array(state)).to(self.device)
        next_state = torch.LongTensor(np.array(next_state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        # ========== 更新 Critic ==========
        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())

        # 这里的expected_value就是伪代码中间的y_i
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        # 修改：限制target value范围，防止Q值爆炸
        expected_value = torch.clamp(expected_value, -1000, 1000)

        actual_value = self.critic(state, action)
        critic_loss = nn.MSELoss()(actual_value, expected_value.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # 添加梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # ========== 更新 Actor ==========
        # 注意看伪代码，这里的actor损失就是对应策略即actor输出的action下对应critic值的负均值
        policy_q_values = self.critic(state, self.actor(state))  # 修改
        actor_loss = - policy_q_values.mean()  # 修改

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # 添加梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # 各自目标网络的参数软更新
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) +
                param.data * self.tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) +
                param.data * self.tau
            )
        return {
            "critic1_loss": float(critic_loss.item()),  # 修改
            "actor_loss": float(-policy_q_values.mean().item()),  # 修改：记录-Q值（优化目标），训练良好时应下降
            "q_value": float(policy_q_values.mean().item()),  # 修改：记录Q值本身，训练良好时应上升（变得不那么负）
            "target_q": float(target_value.mean().item()),  # 修改：记录目标Q值，用于诊断
        }

    # =========2026.1.31修改，添加模型保存和加载功能==========
    def save_model(self, path):
        """保存模型权重"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
        }, path)
        print(f"模型已保存到: {path}")

    def load_model(self, path):
        """加载模型权重"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        print(f"模型已从 {path} 加载")
    # ======修改结束


# 其他的算法可以自己按照参考书写
class DQN:
    def __init__(self,model,memory,cfg):

        self.n_actions = cfg['n_actions']
        self.device = torch.device(cfg['device'])
        self.gamma = cfg['gamma'] # 奖励的折扣因子
        # e-greedy策略相关参数
        self.sample_count = 0  # 用于epsilon的衰减计数
        self.epsilon = cfg['epsilon_start']
        self.sample_count = 0
        self.epsilon_start = cfg['epsilon_start']
        self.epsilon_end = cfg['epsilon_end']
        self.epsilon_decay = cfg['epsilon_decay']
        self.batch_size = cfg['batch_size']
        self.policy_net = model.to(self.device)
        self.target_net = model.to(self.device)
        # 复制参数到目标网络
        for target_param, param in zip(self.target_net.parameters(),self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg['lr']) # 优化器
        self.memory = memory # 经验回放
    def sample_action(self, state):
        ''' 采样动作
        '''
        self.sample_count += 1
        # epsilon指数衰减
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay)
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item() # choose action corresponding to the maximum q value
        else:
            action = random.randrange(self.n_actions)
        return action
    @torch.no_grad() # 不计算梯度，该装饰器效果等同于with torch.no_grad()：
    def predict_action(self, state):
        ''' 预测动作
        '''
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        q_values = self.policy_net(state)
        action = q_values.max(1)[1].item() # choose action corresponding to the maximum q value
        return action
    def update(self):
        if len(self.memory) < self.batch_size: # 当经验回放中不满足一个批量时，不更新策略
            return
        # 从经验回放中随机采样一个批量的转移(transition)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        # 将数据转换为tensor
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch) # 计算当前状态(s_t,a)对应的Q(s_t, a)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach() # 计算下一时刻的状态(s_t_,a)对应的Q值
        # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expected_q_values = reward_batch + self.gamma * next_q_values * (1-done_batch)
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))  # 计算均方根损失
        # 优化更新模型
        self.optimizer.zero_grad()
        loss.backward()
        # clip防止梯度爆炸
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return {"dqn_loss": float(loss.item())}


class TD3(object):
    def __init__(self,cfg):
        self.gamma = cfg.gamma # 奖励的折扣因子
        self.actor_lr = cfg.actor_lr # actor学习率
        self.critic_lr = cfg.critic_lr # critic学习率
        self.policy_noise = cfg.policy_noise
        self.noise_clip = cfg.noise_clip
        self.expl_noise = cfg.expl_noise
        self.policy_freq = cfg.policy_freq
        self.batch_size =  cfg.batch_size 
        self.tau = cfg.tau
        self.sample_count = 0
        self.policy_freq = cfg.policy_freq
        self.explore_steps = cfg.explore_steps
        self.device = torch.device(cfg.device)
        self.n_actions = cfg.n_actions
        self.action_space = cfg.action_space
        self.action_scale = torch.tensor((self.action_space.high - self.action_space.low)/2, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        self.action_bias = torch.tensor((self.action_space.high + self.action_space.low)/2, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        self.actor = Actor(cfg.n_states, cfg.n_actions, hidden_dim = cfg.actor_hidden_dim).to(self.device)
        self.actor_target = Actor(cfg.n_states, cfg.n_actions, hidden_dim = cfg.actor_hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict()) # 复制参数到目标网络

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = self.actor_lr)

        self.critic_1 = Critic(cfg.n_states, cfg.n_actions, hidden_dim = cfg.critic_hidden_dim).to(self.device)
        self.critic_2 = Critic(cfg.n_states, cfg.n_actions, hidden_dim = cfg.critic_hidden_dim).to(self.device)
        self.critic_1_target = Critic(cfg.n_states, cfg.n_actions, hidden_dim = cfg.critic_hidden_dim).to(self.device)
        self.critic_2_target = Critic(cfg.n_states, cfg.n_actions, hidden_dim = cfg.critic_hidden_dim).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict()) # 复制参数到目标网络
        self.critic_2_target.load_state_dict(self.critic_2.state_dict()) # 复制参数到目标网络
        
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr = self.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr = self.critic_lr)
        self.memory = ReplayBufferQue(cfg.buffer_size)
        # self.memory = ReplayBuffer(n_states, n_actions)

    def sample_action(self, state):
        ''' 采样动作
        '''
        self.sample_count += 1
        if self.sample_count < self.explore_steps:
            return self.action_space.sample()
        else:
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            action = self.actor(state)
            action = self.action_scale * action + self.action_bias
            action = action.detach().cpu().numpy()[0]
            action_noise = np.random.normal(0, self.action_scale.cpu().numpy()[0] * self.expl_noise, size=self.n_actions)
            action = (action + action_noise).clip(self.action_space.low, self.action_space.high)
            return action

    @torch.no_grad()
    def predict_action(self, state):
        ''' 预测动作
        '''
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        action = self.actor(state)
        action = self.action_scale * action + self.action_bias # 对actor计算的动作分布放缩
        return action.detach().cpu().numpy()[0]

    def update(self):
        # if len(self.memory) < self.batch_size:
        #     return
        if len(self.memory) < self.explore_steps: # 当经验回放中不满足一个批量时，不更新策略
            return
        state, action, reward, next_state, done = self.memory.sample(self.batch_size) # 从经验回放中随机采样一个批量的转移(transition)
         # 将数据转换为tensor
        state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32)
        action = torch.tensor(np.array(action), device=self.device, dtype=torch.float32)
        next_state = torch.tensor(np.array(next_state), device=self.device, dtype=torch.float32)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32).unsqueeze(1)
        done = torch.tensor(done, device=self.device, dtype=torch.float32).unsqueeze(1)
        # update critic
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip) # 构造加入目标动作的噪声
        # 计算加入了噪声的目标动作
        next_action = (self.actor_target(next_state) + noise).clamp(-self.action_scale+self.action_bias, self.action_scale+self.action_bias)
        # 计算两个critic网络对t+1时刻的状态动作对的评分，并选取更小值来计算目标q值
        target_q1, target_q2 = self.critic_1_target(next_state, next_action).detach(), self.critic_2_target(next_state, next_action).detach()
        target_q = torch.min(target_q1, target_q2)
        target_q = reward + self.gamma * target_q * (1 - done)
        # 计算两个critic网络对t时刻的状态动作对的评分
        current_q1, current_q2 = self.critic_1(state, action), self.critic_2(state, action)
        # 计算均方根损失
        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        actor_loss_value = None
        if self.sample_count % self.policy_freq == 0:
            # 延迟策略更新，actor的更新频率低于critic
            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.item())
            #目标网络软更新
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return {
            "critic1_loss": float(critic_1_loss.item()),
            "critic2_loss": float(critic_2_loss.item()),
            "actor_loss": actor_loss_value,  # 可能为 None
        }

    # =========2026.1.31修改，添加模型保存和加载功能==========
    def save_model(self, path):
        """保存模型权重"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_1_target': self.critic_1_target.state_dict(),
            'critic_2_target': self.critic_2_target.state_dict(),
        }, path)
        print(f"模型已保存到: {path}")

    def load_model(self, path):
        """加载模型权重"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_1_target.load_state_dict(checkpoint['critic_1_target'])
        self.critic_2_target.load_state_dict(checkpoint['critic_2_target'])
        print(f"模型已从 {path} 加载")
    # ======修改结束


# =========2026.2.11，实验13：SAC算法实现==========

class SACDualHeadActorNetwork(nn.Module):
    """SAC的双头Actor网络，分别处理调度和充电决策"""
    def __init__(self, n_states, n_actions, embedding_dims=8, hidden_dim=256, vocab_size=10):
        super(SACDualHeadActorNetwork, self).__init__()

        # 动作维度分解：前8维调度，后4维充电
        self.dispatch_dims = 8
        self.charging_dims = 4
        assert n_actions == self.dispatch_dims + self.charging_dims, "n_actions must be 12"

        # 共享层
        self.embedding = nn.Embedding(vocab_size, embedding_dims)
        self.shared_linear1 = nn.Linear(n_states * embedding_dims, hidden_dim)
        self.shared_linear2 = nn.Linear(hidden_dim, hidden_dim)

        # 调度头（8维）
        self.dispatch_mean = nn.Linear(hidden_dim, self.dispatch_dims)
        self.dispatch_log_std = nn.Linear(hidden_dim, self.dispatch_dims)

        # 充电头（4维）- 增强充电决策能力
        self.charging_hidden = nn.Linear(hidden_dim, 128)
        self.charging_mean = nn.Linear(128, self.charging_dims)
        self.charging_log_std = nn.Linear(128, self.charging_dims)

        # 标准差范围限制
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state):
        # 共享特征提取
        embedded_states = self.embedding(state)
        embedded_flattened = embedded_states.view(embedded_states.size(0), -1)

        shared = F.relu(self.shared_linear1(embedded_flattened))
        shared = F.relu(self.shared_linear2(shared))

        # 调度头
        dispatch_mean = self.dispatch_mean(shared)
        dispatch_log_std = self.dispatch_log_std(shared)
        dispatch_log_std = torch.clamp(dispatch_log_std, self.log_std_min, self.log_std_max)

        # 充电头（额外的隐藏层）
        charging_features = F.relu(self.charging_hidden(shared))
        charging_mean = self.charging_mean(charging_features)
        charging_log_std = self.charging_log_std(charging_features)
        charging_log_std = torch.clamp(charging_log_std, self.log_std_min, self.log_std_max)

        # 合并输出
        mean = torch.cat([dispatch_mean, charging_mean], dim=1)
        log_std = torch.cat([dispatch_log_std, charging_log_std], dim=1)

        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        """采样动作"""
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # 重参数化技巧
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()

        # 使用sigmoid映射到[0, 1]
        action = torch.sigmoid(x_t)

        # 计算log概率
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(action * (1 - action) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, mean, log_std


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
    def __init__(self, n_states, n_actions, memory, vocab_size=10,
                 embedding_dims=8, hidden_dim=256, use_dual_head=False,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.95, tau=0.005, alpha=0.2, batch_size=128, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.memory = memory  # 经验回放池

        # Actor网络 - 根据配置选择单头或双头
        if use_dual_head:
            self.actor = SACDualHeadActorNetwork(n_states, n_actions, embedding_dims=embedding_dims, hidden_dim=hidden_dim, vocab_size=vocab_size).to(device)
        else:
            self.actor = SACActorNetwork(n_states, n_actions, embedding_dims=embedding_dims, hidden_dim=hidden_dim, vocab_size=vocab_size).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # 两个Critic网络（Twin Q）
        self.critic1 = SACCriticNetwork(n_states, n_actions, embedding_dims=embedding_dims, hidden_dim=hidden_dim, vocab_size=vocab_size).to(device)
        self.critic2 = SACCriticNetwork(n_states, n_actions, embedding_dims=embedding_dims, hidden_dim=hidden_dim, vocab_size=vocab_size).to(device)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # 目标Critic网络
        self.target_critic1 = SACCriticNetwork(n_states, n_actions, embedding_dims=embedding_dims, hidden_dim=hidden_dim, vocab_size=vocab_size).to(device)
        self.target_critic2 = SACCriticNetwork(n_states, n_actions, embedding_dims=embedding_dims, hidden_dim=hidden_dim, vocab_size=vocab_size).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # 自动调整温度参数α
        self.target_entropy = -n_actions  # 目标熵
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = alpha
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    def sample_action(self, state):
        """采样动作（训练时使用，兼容DDPG接口）"""
        state = torch.FloatTensor(state).unsqueeze(0).long().to(self.device)
        action, _, _, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    @torch.no_grad()
    def predict_action(self, state):
        """预测动作（测试时使用，兼容DDPG接口）"""
        state = torch.FloatTensor(state).unsqueeze(0).long().to(self.device)
        mean, _ = self.actor(state)
        action = torch.sigmoid(mean)
        return action.cpu().numpy()[0]

    def update(self):
        """更新网络（兼容DDPG接口）"""
        if len(self.memory) < self.batch_size:
            return None

        # 从经验池采样
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

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

        return {
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q_value": float(q.mean().item()),
            "target_q": float(target_q.mean().item()),
            "alpha": float(self.alpha)
        }

    def soft_update(self, source, target):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save_model(self, path):
        """保存模型权重"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'log_alpha': self.log_alpha,
        }, path)
        print(f"SAC模型已保存到: {path}")

    def load_model(self, path):
        """加载模型权重"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp().item()
        print(f"SAC模型已从 {path} 加载")
