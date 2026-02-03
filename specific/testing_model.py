# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 18:41:00 2024

@author: TLab
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, embedding_dim=16):
        super(MyModel, self).__init__()
        # num_embeddings=6 代表离散值的总数 (0, 1, 2, 3, 4, 5)
        self.embedding = nn.Embedding(num_embeddings=6, embedding_dim=embedding_dim)
        self.fc = nn.Linear(8 * embedding_dim, 10)  # 假设最后需要输出10维

    def forward(self, x):
        # x 的 shape 是 [batch_size, 8]
        x = self.embedding(x)  # 将每个离散值映射到 embedding_dim 维的向量
        x = x.view(x.size(0), -1)  # 展平为 [batch_size, 8 * embedding_dim]
        x = F.relu(self.fc(x))  # 通过全连接层
        return x

# 创建一个假输入，范围在 [0, 5]，形状为 [batch_size, 8]
dummy_input = torch.randint(0, 6, (2, 8))  # 2 samples, each with 8 elements

model = MyModel(embedding_dim=16)
output = model(dummy_input)
print(output)
