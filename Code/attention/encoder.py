import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.activation as act

class EncoderAtt(nn.Module):
    def __init__(self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1, dropout=0.0):
        super(EncoderAtt, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # self.self_attention_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.self_attention = act.MultiheadAttention(embedding_dim, 8, dropout=dropout)
        # self.self_attention_dropout = nn.Dropout(dropout)

        self.spatial_embedding = nn.Linear(2, embedding_dim)

    def forward(self, obs_traj):  # pylint: disable=arguments-differ
        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(
            -1, batch, self.embedding_dim
        )
        
        attn_output, attn_output_weights = self.self_attention(obs_traj_embedding, obs_traj_embedding, obs_traj_embedding)
        # feat_h = torch.sum(attn_output, dim=0, keepdims=True)
        feat_h = attn_output[-1].unsqueeze(0)
        return feat_h
