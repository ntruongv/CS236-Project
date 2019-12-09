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

class EncoderAttDis(nn.Module):
    def __init__(self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1, dropout=0.0):
        super(EncoderAttDis, self).__init__()

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
        # feat_h = attn_output[-1].unsqueeze(0)
        return attn_output

class TrajectoryDiscriminatorPiecewise(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        d_type='local'
    ):
        super(TrajectoryDiscriminatorPiecewise, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.encoder = EncoderAttDis(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm
            )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        batch = traj_rel.size(1)
        multi_h = self.encoder(traj_rel)

        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        if self.d_type == 'local':
            classifier_input = multi_h.view(-1, self.h_dim)
        else:
            raise NotImplementedError
        scores = self.real_classifier(classifier_input)
        return scores
