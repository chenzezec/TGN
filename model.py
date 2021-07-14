import copy

import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)

def linear(input, weight, bias=None):
    output = torch.matmul(input, weight)
    if bias is not None:
        output = output + bias
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout=0.1, bias=True):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.dropout = dropout
        self.w_q = nn.Parameter(torch.tensor(d_model, d_k * n_heads))
        self.w_k = nn.Parameter(torch.tensor(d_model, d_k * n_heads))
        self.w_v = nn.Parameter(torch.tensor(d_model, d_v * n_heads))
        if bias:
            self.bias_q = nn.Parameter(torch.empty(d_k * n_heads))
            self.bias_k = nn.Parameter(torch.empty(d_k * n_heads))
            self.bias_v = nn.Parameter(torch.empty(d_v * n_heads))
        else:
            self.bias_q = self.bias_k = self.bias_v = None
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)
        nn.init.zeros_(self.bias_q)
        nn.init.zeros_(self.bias_k)
        nn.init.zeros_(self.bias_v)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_v * n_heads, d_model)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, q, k, v, attn_mask):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_k x d_model]
        b_size = q.size(0)
        # Q: [b_size x n_heads x len_q x d_k]
        # K: [b_size x n_heads x len_k x d_k]
        # V: [b_size x n_heads x len_k x d_v]
        Q = linear(q, self.w_q, bias=self.bias_q).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = linear(k, self.w_k, bias=self.bias_k).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = linear(v, self.w_v, bias=self.bias_v).view(b_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        scale_factor = np.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / scale_factor
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
            assert attn_mask.size() == scores.size()
            scores.masked_fill_(attn_mask, -1e9)
        attn = self.dropout(self.softmax(scores))
        attn_output = torch.matmul(attn, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)
        output = self.out_proj(attn_output)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        output = self.relu(self.linear1(input))
        output = self.dropout(output)
        output = self.linear2(output)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff=2048, dropout=0):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads=n_heads, dropout=dropout)
        self.ff_layer = FeedForward(d_model, d_ff, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, attn_mask=None):
        out1 = self.self_attn(src, src, src, attn_mask=attn_mask)
        out1 = self.norm1(src + self.dropout1(out1))
        out2 = self.ff_layer(out1)
        out2 = self.norm2(out1 + self.dropout2(out2))

        return out2

class Encoder(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_layers, n_heads, d_ff=2048, dropout=0):
        super(Encoder, self).__init__()
        encoder_layer = EncoderLayer(d_model, d_k, d_v, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer for i in range(n_layers))])
        self.n_layers = n_layers

    def forward(self, src, attn_mask):
        out = src
        for i in range(self.n_layers):
            out = self.layers[i](out, attn_mask=attn_mask)

        return out

class TrajectoryModel(nn.Module):
    def __init__(self, args, dropout):
        super(TrajectoryModel, self).__init__()
        self.args = args
        d_model = 64
        d_ff = 2048
        n_layers = 6
        n_heads = 8
        dropout = 0.1

        d_k = d_v = d_model // n_heads
        self.spatial_encoder_1 = Encoder(d_model, d_k, d_v, n_layers, d_ff, dropout)
        self.spatial_encoder_2 = Encoder(d_model, d_k, d_v, n_layers, d_ff, dropout)

        self.temporal_encoder = Encoder(d_model, d_k, d_v, n_layers, d_ff, dropout)

        self.abs_embedding = nn.Linear(2, 64)
        self.rel_embedding = nn.Linear(2, 64)

        self.spatial_fusion = nn.Linear(128, 64)
        self.fusion_layer = nn.Linear(128, 64)
        self.output_layer = nn.Linear(80, 2)

    def normalize_input(self, seq):
        # seq = seq.permute(1, 0, 2)
        mean_x = torch.mean(seq[:, :, 0])
        mean_y = torch.mean(seq[:, :, 1])

        seq[:, :, 0] = seq[:, :, 0] - mean_x
        seq[:, :, 1] = seq[:, :, 1] - mean_y

        return seq

    def forward(self, inputs, iftest=False):
        traj, traj_rel, social_adj = inputs
        output = torch.zeros(traj.shape[0], traj.shape[1], 2).cuda()
        noise = get_noise((1, 16), 'gaussian')
        for curr_frame in range(self.args.seq_len - 1):
            if curr_frame >= self.args.obs_len and iftest:
                traj_input = output[self.obs_len - 1:curr_frame]
                traj_input = torch.cat((traj[:self.obs_len], traj_input))
                traj_rel_input = torch.zeros(traj_input.shape)
                traj_rel_input[1:] = traj_input[1:] - traj_input[:-1]
                traj_input = self.normalize_input(traj_input)
            else:
                traj_input = traj[:curr_frame + 1]
                traj_rel_input = traj_rel[:curr_frame + 1]
                traj_input = self.normalize_input(traj_input)

            spatial_input_embedding = self.abs_embedding(traj_input)
            temporal_input_embedding = self.rel_embedding(traj_rel_input)

            spatial_embedding_1 = self.spatial_encoder_1(spatial_input_embedding[-1].unsqueeze(0), social_adj)
            if curr_frame >= 1:
                spatial_embedding_2 = self.spatial_encoder_2(spatial_input_embedding[-2].unsqueeze(0), social_adj)
            else:
                spatial_embedding_2 = torch.zeros_like(spatial_embedding_1).cuda()
            temporal_embedding = self.temporal_encoder(temporal_input_embedding.permute(1, 0, 2))
            temporal_embedding = temporal_embedding.permute(1, 0, 2)[-1]
            spatial_embedding = torch.cat((spatial_embedding_1, spatial_encoder_2), dim=2)
            spatial_embedding = self.spatial_fusion(spatial_embedding)[-1]
            st_fusion = torch.cat((spatial_embedding, temporal_embedding), dim=1)
            st_fusion = self.fusion_layer(st_fusion)
            noise = noise.repeat(st_fusion.shape[0], 1)
            output_next = self.output_layer(torch.cat((st_fusion, noise), dim=1))
            output[curr_frame] = output_next

        return output
