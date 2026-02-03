# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 19:19:58 2024

@author: TLab
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from collections import Dict
import json  
  
# 加载JSON文件
# with open('.\\specific\\config.json', 'r') as file:
#     config = json.load(file)

import os

current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 拼接出 config.json 的绝对路径
config_path = os.path.join(current_dir, 'config.json')

# 3. 使用绝对路径打开文件 (注意：这里不需要 os.chdir)
with open(config_path, 'r') as file:
    # --- 注意这里有缩进 (4个空格) ---
    config = json.load(file)     # 读取配置
    PL_NUM = config['PL_NUM']    # 获取参数
    # -----------------------------
# === 修复代码结束 ===

PL_NUM = config['PL_NUM']  
ZONE_NUM = config['ZONE_NUM']  
CAR_NUM = config['CAR_NUM']  
TIME_NUM = config['TIME_NUM']  
TIME_LENGTH = config['TIME_LENGTH']  
TIME_SLICE = config['TIME_SLICE']  
od_max = config['OD_MAX'] * (ZONE_NUM-1)
DIST_MAX = config['DIST_MAX']  
car_speed_min = config['car_speed_min']  
manzu_min = config['manzu_min']  
diaodu_min = config['diaodu_min']  
elec_min = config['elec_min']  
BATTERY_CAPACITY = config['BATTERY_CAPACITY']  
path = config['path']  
training_set_number = config['training_set_number']  

class Actor(nn.Module):
    def __init__(self, n_states, n_actions, embedding_dims=8, hidden_dim=256, init_w=3e-3):
        super(Actor, self).__init__()
        
        # 嵌入层，适用于 MultiDiscrete 状态空间
        self.embedding = nn.Embedding(od_max, embedding_dims)
        
        # 全连接层定义
        self.linear1 = nn.Linear(n_states * embedding_dims, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出的动作空间是展平后的矩阵
        self.linear3 = nn.Linear(hidden_dim, n_actions)  # n_actions 是矩阵的总元素数

        # 初始化权重
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        # 嵌入层处理离散状态
        embedded_states = self.embedding(x)  # [batch_size, n_states, embedding_dim]
        embedded_flattened = embedded_states.view(embedded_states.size(0), -1)
        # 前向传播经过全连接层
        x = F.relu(self.linear1(embedded_flattened))
        x = F.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))  # 使用 tanh 保证输出范围在 [-1, 1]

        # 将 tanh 输出映射到 [0, 1] 之间
        
        # print("神经网络动作")
        # print(x)
        
        return x

class Critic(nn.Module):
    def __init__(self, n_states, n_actions, embedding_dims=8, hidden_dim=256, init_w=3e-3):
        super(Critic, self).__init__()

        # 嵌入层，适用于 MultiDiscrete 状态空间
        self.embeddings = nn.Embedding(od_max, embedding_dims)
        self.linearmod = nn.Linear(n_actions, embedding_dims * n_states)

        # 全连接层定义，输入状态和动作拼接后的向量
        self.linear1 = nn.Linear(embedding_dims * n_states * 2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # 初始化权重
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        # 嵌入层处理离散状态
        embedded_states = self.embeddings(state)
        embedded_flattened = embedded_states.view(embedded_states.size(0), -1)
        
        # 将状态和展平后的动作拼接
        action = F.relu(self.linearmod(action))
        # action_flattened = action.view(action.size(0), -1)  # 展平矩阵动作为一维向量
        x = torch.cat([embedded_flattened, action], dim=1)  # 拼接状态和动作

        # 前向传播经过全连接层
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        
        return x
