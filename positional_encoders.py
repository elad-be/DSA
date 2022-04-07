# Based on:
# https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
import torch
from einops import rearrange
from torch import nn
from collections.abc import Iterable
import math


def _get_spatial_dim(x):
    if isinstance(x, Iterable):
        assert len(x) == 2, 'Dimension should be a 2-Tuple or a single integer, got {}'.format(x)
        return x[0], x[1]
    else:
        assert isinstance(x, int), 'Dimension should be a 2-Tuple or a single integer, got {}'.format(x)
        return x, x


class SpatialPositionalEncoderLearned(nn.Module):
    def __init__(self, img_size, dim, channels_first=True):
        super(SpatialPositionalEncoderLearned, self).__init__()
        self.H, self.W = _get_spatial_dim(img_size)
        self.img_size = img_size
        self.rows_embedding = nn.Embedding(self.H, dim // 2)
        self.cols_embedding = nn.Embedding(self.W, dim // 2)
        self.channels_first = channels_first
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.rows_embedding.weight)
        nn.init.normal_(self.cols_embedding.weight)
        # nn.init.kaiming_uniform_(self.rows_embedding.weight, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.cols_embedding.weight, a=math.sqrt(5))

    def forward(self, x):
        # x: B C H W
        H, W = x.shape[2::] if self.channels_first else x.shape[1:3]
        i = torch.arange(self.H, device=x.device)
        j = torch.arange(self.W, device=x.device)
        x_emb = self.cols_embedding(j)
        y_emb = self.rows_embedding(i)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(H, 1, 1),
            y_emb.unsqueeze(1).repeat(1, W, 1),
        ], dim=2)
        if self.channels_first:
            pos = pos.permute(2, 0, 1)
        x = x + pos
        return x


class SpatialPositionalEncoderSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, img_size, dim, channels_first=True, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.dim = dim
        self.H, self.W = _get_spatial_dim(img_size)
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        pos = self.get_embedding()
        if channels_first:
            pos = pos.permute(0, 3, 1, 2)
        self.register_buffer('pos', pos)

    def get_embedding(self):
        mask = torch.ones([1, self.H, self.W])
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.dim // 2, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.dim // 2))

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)
        return pos

    def forward(self, x):
        return x + self.pos


class TokenPositionalEncoder(nn.Module):
    def __init__(self, sequence_length, dim):
        super(TokenPositionalEncoder, self).__init__()
        self.sequence_length = sequence_length
        self.token_embedding = nn.Embedding(sequence_length, dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.token_embedding.weight, a=math.sqrt(5))

    def forward(self, x):
        N = x.size(1)
        i = torch.arange(N, device=x.device)
        pos = self.token_embedding(i).unsqueeze(0).repeat(x.size(0), 1, 1)
        x = x + pos
        return x
