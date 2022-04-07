import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math
import os
import torch
from torch import nn
from attention_augmented_conv import AugmentedConv

from multihead_attention import MultiHeadAttention
from positional_encoders import TokenPositionalEncoder
from utils import FoldingLayer

from positional_encoders import SpatialPositionalEncoderLearned, SpatialPositionalEncoderSine
from torch.nn.init import kaiming_uniform_
import math

class Upformer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1, kernel_size=3, scale=2, post_norm=True, pos_embedding=None,
                 masked=False, bias=True, img_size=None, use_relative_pos_embedding=False, rescale_attention=False):
        super(Upformer, self).__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.up_query = nn.Parameter(torch.randn([1, scale ** 2, in_dim]), requires_grad=True)
        # kaiming_uniform_(self.up_query, a=math.sqrt(5))
        self.att = MultiHeadAttention(in_dim, heads, project_dim=out_dim, rescale_attention=rescale_attention)
        self.to_k = nn.Conv2d(in_dim, in_dim, 1, 1, 0, bias=bias)
        self.to_v = nn.Conv2d(in_dim, in_dim, 1, 1, 0, bias=bias)
        self.img_size = img_size
        self.kernel_size = kernel_size
        self.scale = scale
        self.unfold = FoldingLayer(kernel_size, stride=1, padding=(self.kernel_size - 1) // 2)
        self.pe_k = self._get_pos_embedding(pos_embedding)
        self.global_pos = (pos_embedding is not None and pos_embedding.startswith('global'))
        self.pre_norm = nn.GroupNorm(1, in_dim) if not post_norm else nn.Identity()
        self.post_norm = nn.LayerNorm(out_dim) if post_norm else nn.Identity()

        one_row = torch.Tensor([10e5 for _ in range(kernel_size ** 2)])
        one_row[[0, 1, kernel_size, kernel_size + 1]] = 0
        mask = torch.stack([one_row.roll(i, dims=0) for i in range(scale ** 2)], dim=0).view(scale ** 2,
                                                                                             kernel_size ** 2)
        self.mask = nn.Parameter(mask, requires_grad=False) if masked else None
        if not self.global_pos:
            self.aux_zeros = nn.Parameter(torch.zeros([1, self.kernel_size, self.kernel_size, in_dim]),
                                          requires_grad=False)
        if use_relative_pos_embedding:
            self.relative_pos_embedding = nn.Parameter(torch.randn([1, heads, scale ** 2, kernel_size ** 2]),
                                                       requires_grad=True)
        else:
            self.relative_pos_embedding = None

    def _get_pos_embedding(self, pos_embedding):
        if pos_embedding == 'global_learned':
            return SpatialPositionalEncoderLearned(self.img_size, self.in_dim, channels_first=True)
        if pos_embedding == 'global_sine':
            return SpatialPositionalEncoderSine(self.img_size, self.in_dim, channels_first=True)
        if pos_embedding == 'local_learned':
            return SpatialPositionalEncoderLearned(self.kernel_size, self.in_dim, channels_first=False)
        if pos_embedding == 'local_sine':
            return SpatialPositionalEncoderSine(self.kernel_size, self.in_dim, channels_first=False)
        return nn.Identity()

    def forward(self, x, debug=False):
        debug_ret = []
        x = self.pre_norm(x)
        if self.global_pos:
            x_k = self.pe_k(x).cuda()
            x_v = x.cuda()
            x_k = self.unfold(self.to_k(x_k))
            x_v = self.unfold(self.to_v(x_v))
        else:
            x_k = (self.unfold(self.to_k(x)) +
                   self.to_k(self.pe_k(self.aux_zeros).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).view(1,
                                                                                                     self.kernel_size ** 2,
                                                                                                     self.in_dim))
            x_v = self.unfold(self.to_v(x))
            # x_unfold = self.unfold(x).view(-1, self.kernel_size, self.kernel_size, self.in_dim)
            # x_k = self.to_k(self.pe_k(x_unfold)).view(-1, self.kernel_size ** 2, self.in_dim)
            # x_v = self.to_v(x_unfold).view(-1, self.kernel_size ** 2, self.in_dim)
        x_q = self.up_query.repeat(x_v.size(0), 1, 1)
        print("x_q shape", x_q.shape)

        up_sampled = self.att(x_q, x_k, x_v, mask=self.mask, relative_pos=self.relative_pos_embedding, return_attention=debug)
        print(up_sampled.shape)
        if debug:
            debug_ret.append(up_sampled[1])
            up_sampled = self.post_norm(up_sampled[0])
        else:
            up_sampled = self.post_norm(up_sampled)

        B, _, H, W = x.size()
        up_sampled = up_sampled.view(B, H,
                                     W,
                                     self.scale,
                                     self.scale,
                                     self.out_dim).permute(0, 5, 1, 3, 2, 4).reshape(B, self.out_dim, H * self.scale,
                                                                                     W * self.scale)
        if debug:
            return (up_sampled, *debug_ret)
        else:
            return up_sampled
            
            
            
#tmp = torch.randn((4, 4, 128,128)).cuda()
#augmented_conv1 = AugmentedConv(in_channels=4, out_channels=256, kernel_size=3, dk=256, dv=16, Nh=4, relative=True, stride=2, shape=64).cuda()
#conv_out1 = augmented_conv1(tmp)
#augmented_conv2 = AugmentedConv(in_channels=256, out_channels=512, kernel_size=3, dk=512, dv=16, Nh=4, relative=True, stride=2, shape=32).cuda()
#conv_out1 = augmented_conv2(conv_out1)
#print(conv_out1.shape)


#from self_attention_cv import MultiHeadSelfAttention
#model = MultiHeadSelfAttention(dim=64)
#x = torch.rand(16, 10, 64)  # [batch, tokens, dim]
#mask = torch.zeros(10, 10)  # tokens X tokens
#mask[5:8, 5:8] = 1
#y = model(x, mask)
#print('Shape of output is: ', y.shape)
#print('-'*70)
#print('Output corresponding to the first token/patch in the first batch \n')
#print(y.detach().numpy()[0][0]) 


from self_attention_cv.bottleneck_transformer import BottleneckBlock
x = torch.rand(4, 512, 32, 32)
bottleneck_block = BottleneckBlock(in_channels=512, fmap_size=(32, 32), heads=4, out_channels=512, pooling=False)
y = bottleneck_block(x)
print('Shape of output is: ', y.shape)
print('-'*70)
print('Output corresponding to the first patch in the first head, first batch \n')
print(y.detach().numpy()[0][0][0]) 

#augmented_conv1 = AugmentedConv(in_channels=64, out_channels=128, kernel_size=3, dk=40, dv=4, Nh=4, relative=True, stride=2, shape=16)


#x = torch.rand(8, 64, 256, 256)
#tr = augmented_conv1(in_dim=64, out_dim=64,heads = 8, scale=1, pos_embedding='local_sine')
#x = augmented_conv1(x)
#print(x.shape)