import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math
import os
import torch
from torch import autograd
from torch import nn
import pytorch_lightning as pl
from SparseConv import SparseConv
import cspn as post_process
from multihead_attention import MultiHeadAttention
from positional_encoders import TokenPositionalEncoder
from utils import FoldingLayer
from attention_augmented_conv import AugmentedConv

from positional_encoders import SpatialPositionalEncoderLearned, SpatialPositionalEncoderSine
from torch.nn.init import kaiming_uniform_
import math
from self_attention_cv.bottleneck_transformer import BottleneckBlock
from blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock
import numpy as np


class Upformer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=16, kernel_size=3, scale=2, post_norm=True, pos_embedding=None,
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

        up_sampled = self.att(x_q, x_k, x_v, mask=self.mask, relative_pos=self.relative_pos_embedding,
                              return_attention=debug)
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


class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

        # create kernel [1, 0; 0, 0]
        self.weights = torch.autograd.Variable(
            torch.zeros(num_channels, 1, stride, stride).cuda())  # currently not compatible with running on CPU
        self.weights[:, :, 0, 0] = 1

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights, stride=self.stride, groups=self.num_channels)


def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class Decoder_ref(nn.Module):
    # Decoder is the base class for all decoders

    names = ['deconv2', 'deconv3', 'upconv', 'upproj']

    def __init__(self):
        super(Decoder_ref, self).__init__()

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

        self.tr0 = Upformer(in_dim=1024, out_dim=512, pos_embedding='local_sine')
        self.tr1 = Upformer(in_dim=1536, out_dim=512, pos_embedding='local_sine')
        self.tr2 = Upformer(in_dim=1024, out_dim=256, pos_embedding='local_sine')
        self.tr3 = Upformer(in_dim=512, out_dim=128, pos_embedding='local_sine')
        self.tr4 = Upformer(in_dim=128, out_dim=64, pos_embedding='local_sine')
        self.relu = nn.ReLU()

        self.dec0 = Upformer(in_dim=512, out_dim=256, pos_embedding='local_sine')
        self.dec1 = Upformer(in_dim=512, out_dim=256, pos_embedding='local_sine')
        self.dec2 = Upformer(in_dim=384, out_dim=128, pos_embedding='local_sine')
        self.dec3 = Upformer(in_dim=192, out_dim=64, pos_embedding='local_sine')
        # self.dec4 = Upformer(in_dim=64, out_dim=32, pos_embedding='local_sine')
        self.conv4 = nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(8)
        self.bn5 = nn.BatchNorm2d(16)
        self.bn6 = nn.BatchNorm2d(8)
        # self.dec5 = Upformer(in_dim=16, out_dim=8, pos_embedding='local_sine')
        self.conv4.apply(weights_init)
        self.conv5.apply(weights_init)
        self.conv6.apply(weights_init)
        self.bn4.apply(weights_init)
        self.bn5.apply(weights_init)
        self.bn6.apply(weights_init)

    def merge_rgb_depth(self, rgb, depth):
        new_x = []
        new_x = new_x
        for i in range(len(rgb)):
            new_x.append(torch.cat((rgb[i], depth[i]), 0).cuda())
        x = torch.stack(new_x).cuda()
        # l = torch.nn.Conv2d(2048, 1024, kernel_size=3, padding=1).cuda()
        # x = l(x).cuda()
        return x

    def forward(self, x, m0, m1, m2, m3):
        mrg = torch.cat((x, m0), 1).cuda()
        x = self.relu(mrg)
        x = self.dec0(x)
        x = self.relu(x)
        mrg = torch.cat((x, m1), 1).cuda()
        x = self.relu(mrg)
        x = self.dec1(x)
        x = self.relu(x)
        mrg = torch.cat((x, m2), 1).cuda()
        x = self.relu(mrg)
        x = self.dec2(x)
        x = self.relu(x)
        mrg = torch.cat((x, m3), 1).cuda()
        x = self.relu(mrg)
        x = self.dec3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        return x
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    # Decoder is the base class for all decoders
    names = ['deconv2', 'deconv3', 'upconv', 'upproj']

    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

        self.tr0 = Upformer(in_dim=1024, out_dim=512, pos_embedding='local_sine')
        self.tr1 = Upformer(in_dim=1536, out_dim=512, pos_embedding='local_sine')
        self.tr2 = Upformer(in_dim=1024, out_dim=256, pos_embedding='local_sine')
        self.tr3 = Upformer(in_dim=512, out_dim=128, pos_embedding='local_sine')
        self.tr4 = Upformer(in_dim=128, out_dim=64, pos_embedding='local_sine')
        self.relu = nn.ReLU()

        self.dec0 = Upformer(in_dim=512, out_dim=256, pos_embedding='local_sine')
        self.dec1 = Upformer(in_dim=512, out_dim=256, pos_embedding='local_sine')
        self.dec2 = Upformer(in_dim=384, out_dim=128, pos_embedding='local_sine')
        self.dec3 = Upformer(in_dim=192, out_dim=64, pos_embedding='local_sine')
        self.dec4 = Upformer(in_dim=64, out_dim=16, pos_embedding='local_sine')
        self.conv4 = nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(16)
        self.bn6 = nn.BatchNorm2d(8)
        # self.dec5 = Upformer(in_dim=16, out_dim=8, pos_embedding='local_sine')
        self.conv4.apply(weights_init)
        self.conv5.apply(weights_init)
        self.conv6.apply(weights_init)
        self.bn4.apply(weights_init)
        self.bn5.apply(weights_init)
        self.bn6.apply(weights_init)

    def merge_rgb_depth(self, rgb, depth):
        new_x = []
        new_x = new_x
        for i in range(len(rgb)):
            new_x.append(torch.cat((rgb[i], depth[i]), 0).cuda())
        x = torch.stack(new_x).cuda()
        # l = torch.nn.Conv2d(2048, 1024, kernel_size=3, padding=1).cuda()
        # x = l(x).cuda()
        return x

    def forward(self, x, m0, m1, m2, m3):
        mrg = torch.cat((x, m0), 1).cuda()
        x = self.relu(mrg)
        x = self.dec0(x)
        f3 = x
        x = self.relu(x)
        mrg = torch.cat((x, m1), 1).cuda()
        x = self.relu(mrg)
        x = self.dec1(x)
        f2 = x
        x = self.relu(x)
        mrg = torch.cat((x, m2), 1).cuda()
        x = self.relu(mrg)
        x = self.dec2(x)
        f1 = x
        x = self.relu(x)
        mrg = torch.cat((x, m3), 1).cuda()
        x = self.relu(mrg)
        x = self.dec3(x)
        f0 = x
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        return x, f3, f2, f1, f0
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        return x, f3, f2, f1, f0


class DeConv(Decoder):
    def __init__(self, in_channels, kernel_size):
        assert kernel_size >= 2, "kernel_size out of range: {}".format(kernel_size)
        super(DeConv, self).__init__()

        def convt(in_channels):
            stride = 2
            padding = (kernel_size - 1) // 2
            output_padding = kernel_size % 2
            assert -2 - 2 * padding + kernel_size + output_padding == 0, "deconv parameters incorrect"

            module_name = "deconv{}".format(kernel_size)
            return nn.Sequential(collections.OrderedDict([
                (module_name, nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size,
                                                 stride, padding, output_padding, bias=False)),
                ('batchnorm', nn.BatchNorm2d(in_channels // 2)),
                ('relu', nn.ReLU(inplace=True)),
            ]))

        self.layer1 = convt(in_channels)
        self.layer2 = convt(in_channels // 2)
        self.layer3 = convt(in_channels // (2 ** 2))
        self.layer4 = convt(in_channels // (2 ** 3))


class DeConv_ref(Decoder_ref):
    def __init__(self, in_channels, kernel_size):
        assert kernel_size >= 2, "kernel_size out of range: {}".format(kernel_size)
        super(DeConv_ref, self).__init__()

        def convt(in_channels):
            stride = 2
            padding = (kernel_size - 1) // 2
            output_padding = kernel_size % 2
            assert -2 - 2 * padding + kernel_size + output_padding == 0, "deconv parameters incorrect"

            module_name = "deconv{}".format(kernel_size)
            return nn.Sequential(collections.OrderedDict([
                (module_name, nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size,
                                                 stride, padding, output_padding, bias=False)),
                ('batchnorm', nn.BatchNorm2d(in_channels // 2)),
                ('relu', nn.ReLU(inplace=True)),
            ]))

        self.layer1 = convt(in_channels)
        self.layer2 = convt(in_channels // 2)
        self.layer3 = convt(in_channels // (2 ** 2))
        self.layer4 = convt(in_channels // (2 ** 3))


class UpConv(Decoder):
    # UpConv decoder consists of 4 upconv modules with decreasing number of channels and increasing feature map size
    def upconv_module(self, in_channels):
        # UpConv module: unpool -> 5*5 conv -> batchnorm -> ReLU
        upconv = nn.Sequential(collections.OrderedDict([
            ('unpool', Unpool(in_channels)),
            ('conv', nn.Conv2d(in_channels, in_channels // 2, kernel_size=5, stride=1, padding=2, bias=False)),
            ('batchnorm', nn.BatchNorm2d(in_channels // 2)),
            ('relu', nn.ReLU()),
        ]))
        return upconv

    def __init__(self, in_channels):
        super(UpConv, self).__init__()
        self.layer1 = self.upconv_module(in_channels)
        self.layer2 = self.upconv_module(in_channels // 2)
        self.layer3 = self.upconv_module(in_channels // 4)
        self.layer4 = self.upconv_module(in_channels // 8)


class UpProj(Decoder):
    # UpProj decoder consists of 4 upproj modules with decreasing number of channels and increasing feature map size

    class UpProjModule(nn.Module):
        # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
        #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
        #   bottom branch: 5*5 conv -> batchnorm

        def __init__(self, in_channels):
            super(UpProj.UpProjModule, self).__init__()
            out_channels = in_channels // 2
            self.unpool = Unpool(in_channels)
            self.upper_branch = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)),
                ('batchnorm1', nn.BatchNorm2d(out_channels)),
                ('relu', nn.ReLU()),
                ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
                ('batchnorm2', nn.BatchNorm2d(out_channels)),
            ]))
            self.bottom_branch = nn.Sequential(collections.OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)),
                ('batchnorm', nn.BatchNorm2d(out_channels)),
            ]))
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.unpool(x)
            x1 = self.upper_branch(x)
            x2 = self.bottom_branch(x)
            x = x1 + x2
            x = self.relu(x)
            return x

    def __init__(self, in_channels):
        super(UpProj, self).__init__()
        self.layer1 = self.UpProjModule(in_channels)
        self.layer2 = self.UpProjModule(in_channels // 2)
        self.layer3 = self.UpProjModule(in_channels // 4)
        self.layer4 = self.UpProjModule(in_channels // 8)


def choose_decoder(decoder, in_channels):
    # iheight, iwidth = 10, 8
    if decoder[:6] == 'deconv':
        assert len(decoder) == 7
        kernel_size = int(decoder[6])
        return DeConv(in_channels, kernel_size)
    elif decoder == "upproj":
        return UpProj(in_channels)
    elif decoder == "ref":
        return DeConv_ref(in_channels, 3)
    elif decoder == "upconv":
        return UpConv(in_channels)
    else:
        assert False, "invalid option for decoder: {}".format(decoder)


class SparseDownSampleClose(nn.Module):
    def __init__(self, stride):
        super(SparseDownSampleClose, self).__init__()
        self.pooling = nn.MaxPool2d(stride, stride)
        self.large_number = 600

    def forward(self, d, mask):
        encode_d = - (1 - mask) * self.large_number - d

        d = - self.pooling(encode_d)
        mask_result = self.pooling(mask)
        d_result = d - (1 - mask_result) * self.large_number

        return d_result, mask_result


class ResNet(pl.LightningModule):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):
        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError(
                'Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet, self).__init__()
        os.environ['TORCH_HOME'] = 'models\\resnet'  # elad edit
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)
        in_channels = 4
        cspn_config_default = {'step': 24, 'kernel': 3, 'norm_type': '8sum'}

        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
            self.bn3 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1,
                                   bias=False)  ###change - small kernel size
            self.conv1_ref = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.conv1_ref)
            weights_init(self.bn1)
            weights_init(self.bn3)

        self.output_size = output_size

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        self.layer1_ref = pretrained_model._modules['layer1']
        self.layer2_ref = pretrained_model._modules['layer2']
        self.layer3_ref = pretrained_model._modules['layer3']
        self.layer4_ref = pretrained_model._modules['layer4']
        self.softmax = nn.Softmax(dim=1)
        self.post_process_layer = self._make_post_process_layer(cspn_config_default)

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels, num_channels // 2, kernel_size=1,
                               bias=False)  # nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.conv2_ref = nn.Conv2d(num_channels, num_channels // 2, kernel_size=1,
                                   bias=False)  # nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels // 2)
        self.bn4 = nn.BatchNorm2d(num_channels // 2)
        self.decoder = choose_decoder(decoder, num_channels // 2)

        self.decoder_ref = choose_decoder('ref', num_channels // 2)

        # setting bias=true doesn't improve accuracy
        self.conv3 = nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.middle = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_ref = nn.Conv2d(8, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='nearest')

        # weight init
        self.conv2.apply(weights_init)
        self.conv2_ref.apply(weights_init)
        self.bn2.apply(weights_init)
        self.bn4.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3.apply(weights_init)
        self.conv3_ref.apply(weights_init)

        self.btl1 = BottleneckBlock(in_channels=256, fmap_size=(16, 16), heads=4, out_channels=256, pooling=False)
        self.mrg1_conv = nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.mrg2_conv = nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.mrg3_conv = nn.Conv2d(384, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.mrg4_conv = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.btl2 = BottleneckBlock(in_channels=256, fmap_size=(16, 16), heads=4, out_channels=256, pooling=False)

        self.mrg1_conv.apply(weights_init)
        self.mrg2_conv.apply(weights_init)
        self.mrg3_conv.apply(weights_init)
        self.mrg4_conv.apply(weights_init)

        ######## PENet
        self.rgb_conv_init = convbnrelu(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2)

        self.rgb_encoder_layer1 = BasicBlockGeo(inplanes=32, planes=64, stride=2, geoplanes=1)
        self.rgb_encoder_layer2 = BasicBlockGeo(inplanes=64, planes=64, stride=1, geoplanes=1)
        self.rgb_encoder_layer3 = BasicBlockGeo(inplanes=64, planes=128, stride=2, geoplanes=1)
        self.rgb_encoder_layer4 = BasicBlockGeo(inplanes=128, planes=128, stride=1, geoplanes=1)
        self.rgb_encoder_layer5 = BasicBlockGeo(inplanes=128, planes=256, stride=2, geoplanes=1)
        self.rgb_encoder_layer6 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=1)
        self.rgb_encoder_layer7 = BasicBlockGeo(inplanes=256, planes=512, stride=2, geoplanes=1)
        self.rgb_encoder_layer8 = BasicBlockGeo(inplanes=512, planes=512, stride=1, geoplanes=1)
        self.rgb_encoder_layer9 = BasicBlockGeo(inplanes=512, planes=1024, stride=2, geoplanes=1)
        self.rgb_encoder_layer10 = BasicBlockGeo(inplanes=1024, planes=1024, stride=1, geoplanes=1)

        self.depth_conv_init = convbnrelu(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=2)

        self.depth_layer1 = BasicBlockGeo(inplanes=32, planes=64, stride=2, geoplanes=1)
        self.depth_layer2 = BasicBlockGeo(inplanes=64, planes=64, stride=1, geoplanes=1)
        self.depth_layer3 = BasicBlockGeo(inplanes=128, planes=128, stride=2, geoplanes=1)
        self.depth_layer4 = BasicBlockGeo(inplanes=128, planes=128, stride=1, geoplanes=1)
        self.depth_layer5 = BasicBlockGeo(inplanes=256, planes=256, stride=2, geoplanes=1)
        self.depth_layer6 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=1)
        self.depth_layer7 = BasicBlockGeo(inplanes=512, planes=512, stride=2, geoplanes=1)
        self.depth_layer8 = BasicBlockGeo(inplanes=512, planes=512, stride=1, geoplanes=1)
        self.depth_layer9 = BasicBlockGeo(inplanes=1024, planes=1024, stride=2, geoplanes=1)
        self.depth_layer10 = BasicBlockGeo(inplanes=1024, planes=1024, stride=1, geoplanes=1)
        self.sparsepooling = SparseDownSampleClose(stride=2)
        ###########################

    def _make_post_process_layer(self, cspn_config=None):
        return post_process.Affinity_Propagate(cspn_config['step'],
                                               cspn_config['kernel'],
                                               norm_type=cspn_config['norm_type'])

    def forward(self, x):
        input_depth = x[:, 3:, :, :]

        ### preperations for the PENet encoder
        valid_mask = torch.where(input_depth > 0, torch.full_like(input_depth, 1.0), torch.full_like(input_depth, 0.0))
        d_s2, vm_s2 = self.sparsepooling(input_depth, valid_mask)
        d_s3, vm_s3 = self.sparsepooling(d_s2, vm_s2)
        d_s4, vm_s4 = self.sparsepooling(d_s3, vm_s3)
        d_s5, vm_s5 = self.sparsepooling(d_s4, vm_s4)
        d_s6, vm_s6 = self.sparsepooling(d_s5, vm_s5)
        geo_s1 = input_depth
        geo_s2 = d_s2
        geo_s3 = d_s3
        geo_s4 = d_s4
        geo_s5 = d_s5
        geo_s6 = d_s6
        ###

        # ##################### PENet encoder #1
        # # b 1 352 1216
        # rgb_feature = self.rgb_conv_init(x)
        # rgb_feature1 = self.rgb_encoder_layer1(rgb_feature, geo_s1, geo_s2)  # b 32 176 608
        # rgb_feature2 = self.rgb_encoder_layer2(rgb_feature1, geo_s2, geo_s2)  # b 32 176 608
        # rgb_feature3 = self.rgb_encoder_layer3(rgb_feature2, geo_s2, geo_s3)  # b 64 88 304
        # rgb_feature4 = self.rgb_encoder_layer4(rgb_feature3, geo_s3, geo_s3)  # b 64 88 304
        # rgb_feature5 = self.rgb_encoder_layer5(rgb_feature4, geo_s3, geo_s4)  # b 128 44 152
        # rgb_feature6 = self.rgb_encoder_layer6(rgb_feature5, geo_s4, geo_s4)  # b 128 44 152
        # rgb_feature7 = self.rgb_encoder_layer7(rgb_feature6, geo_s4, geo_s5)  # b 256 22 76
        # rgb_feature8 = self.rgb_encoder_layer8(rgb_feature7, geo_s5, geo_s5)  # b 256 22 76
        # rgb_feature9 = self.rgb_encoder_layer9(rgb_feature8, geo_s5, geo_s6)  # b 512 11 38
        # rgb_feature10 = self.rgb_encoder_layer10(rgb_feature9, geo_s6, geo_s6)  # b 512 11 38
        # ####
        # m3 = rgb_feature2
        # m2 = rgb_feature4
        # m1 = rgb_feature6
        # m0 = rgb_feature8
        # x = self.btl1(rgb_feature10)
        # btl = x
        # ####
        # #####################

        # resnet
        # b 4 256 256
        x = self.conv1(x)  # b 64 256 256
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # b 64 128 128
        x = self.layer1(x)  # b 64 128 128
        m3 = x
        x = self.layer2(x)  # b 128 64 64
        m2 = x
        x = self.layer3(x)  # b 256 32 32
        m1 = x
        x = self.layer4(x)  # b 256 16 16
        x = self.conv2(x)  # b 256 16 16
        x = self.bn2(x)
        m0 = x
        x = self.btl1(x)  # b 256 16 16
        btl = x

        # decoder
        x, f3, f2, f1, f0 = self.decoder(x, m0, m1, m2, m3)
        x = self.conv3(x)
        # x = self.bilinear(x)

        pred_depth_1 = x[:, 0:1, :, :]
        depth_conf_1 = x[:, 1:2, :, :]

        x = torch.cat((pred_depth_1, input_depth), dim=1)

        ################# PENet encoder #2
        sparsed_feature = self.depth_conv_init(x)
        sparsed_feature1 = self.depth_layer1(sparsed_feature, geo_s1, geo_s2)  # b 32 176 608
        sparsed_feature2 = self.depth_layer2(sparsed_feature1, geo_s2, geo_s2)  # b 32 176 608

        sparsed_feature2_plus = torch.cat([f0, sparsed_feature2], 1)
        sparsed_feature3 = self.depth_layer3(sparsed_feature2_plus, geo_s2, geo_s3)  # b 64 88 304
        sparsed_feature4 = self.depth_layer4(sparsed_feature3, geo_s3, geo_s3)  # b 64 88 304

        sparsed_feature4_plus = torch.cat([f1, sparsed_feature4], 1)
        sparsed_feature5 = self.depth_layer5(sparsed_feature4_plus, geo_s3, geo_s4)  # b 128 44 152
        sparsed_feature6 = self.depth_layer6(sparsed_feature5, geo_s4, geo_s4)  # b 128 44 152

        sparsed_feature6_plus = torch.cat([f2, sparsed_feature6], 1)
        sparsed_feature7 = self.depth_layer7(sparsed_feature6_plus, geo_s4, geo_s5)  # b 256 22 76
        sparsed_feature8 = self.depth_layer8(sparsed_feature7, geo_s5, geo_s5)  # b 256 22 76

        sparsed_feature8_plus = torch.cat([f3, sparsed_feature8], 1)
        sparsed_feature9 = self.depth_layer9(sparsed_feature8_plus, geo_s5, geo_s6)  # b 512 11 38
        sparsed_feature10 = self.depth_layer10(sparsed_feature9, geo_s6, geo_s6)  # b 512 11 38
        ####
        m3 = sparsed_feature2
        m2 = sparsed_feature4
        m1 = sparsed_feature6
        m0 = sparsed_feature8
        x = sparsed_feature10 + btl
        x = self.btl2(x)
        ####
        #################

        # ###ref_net
        # x = self.conv1_ref(x)
        # x = self.bn3(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        # mrg = torch.cat((x,f1),dim=1)
        # x = self.mrg1_conv(mrg)
        # x = self.layer1_ref(x)
        # m3 = x
        # mrg = torch.cat((x, f1), dim=1)
        # x = self.mrg2_conv(mrg)
        # x = self.layer2_ref(x)
        # m2 = x
        # mrg = torch.cat((x, f2), dim=1)
        # x = self.mrg3_conv(mrg)
        # x = self.layer3_ref(x)
        # m1 = x
        # mrg = torch.cat((x, f3), dim=1)
        # x = self.mrg4_conv(mrg)
        # x = self.layer4_ref(x)
        # x = self.conv2_ref(x)
        # x = self.bn4(x)
        # m0 = x
        # x = x + btl
        # x = self.btl2(x)

        # decoder
        x = self.decoder_ref(x, m0, m1, m2, m3)
        guidance = x
        x = self.conv3_ref(x)
        # x = self.bilinear(x)
        d_depth, d_conf = torch.chunk(x, 2, dim=1)
        depth_conf_1, d_conf = torch.chunk(self.softmax(torch.cat((depth_conf_1, d_conf), dim=1)), 2, dim=1)
        output = depth_conf_1 * pred_depth_1 + d_conf * d_depth

        # output = self.post_process_layer(guidance, output, input_depth)

        return output, pred_depth_1, d_depth

    def forward_works_bext(self, x):  ##1603 ###this is the working version!
        n = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1, stride=2).cuda()
        attn = n(x).cuda()
        augmented_conv1 = AugmentedConv(in_channels=4, out_channels=512, kernel_size=3, dk=512, dv=32, Nh=4,
                                        relative=True, stride=2, shape=64).cuda()
        attn = augmented_conv1(attn).cuda()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        m3 = x
        x = self.layer2(x)
        m2 = x
        x = self.layer3(attn)  ##attn - replace x and attn
        m1 = x
        x = self.layer4(x)
        x = self.conv2(x)
        x = self.bn2(x)
        m0 = x
        bottleneck_block = BottleneckBlock(in_channels=1024, fmap_size=(16, 16), heads=4, out_channels=1024,
                                           pooling=False).cuda()
        x = bottleneck_block(x)
        # x = x + 0.7 * torch.rand(1,1024,16,16).cuda()

        # decoder
        x = self.decoder(x, m0, m1, m2, m3)
        x = self.conv3(x)
        x = self.bilinear(x)
        return x

    def forward_sp(self, x):  # sparse_conv
        n = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1, stride=2).cuda()

        mask = x[:, 3:, :, :]
        mask = (mask > -1.8035560846328735).float()
        mask = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2).cuda()(mask)
        attn = n(x).cuda()
        augmented_conv1 = AugmentedConv(in_channels=4, out_channels=512, kernel_size=3, dk=512, dv=32, Nh=4,
                                        relative=True, stride=2, shape=64).cuda()

        # mask = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2).cuda()(mask)

        attn = augmented_conv1(attn).cuda()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        ##x = self.layer1(x)

        x, mask = SparseConv(64, 64, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(64, 64, 3).cuda()(x, mask)
        x = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(64, 256, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x = nn.ReLU(inplace=True).cuda()(x)
        x, mask = SparseConv(256, 64, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(64, 64, 3).cuda()(x, mask)
        x = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(64, 256, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x = nn.ReLU(inplace=True).cuda()(x)
        x, mask = SparseConv(256, 64, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(64, 64, 3).cuda()(x, mask)
        x = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(64, 256, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x = nn.ReLU(inplace=True).cuda()(x)
        m3 = x

        ###layer 2
        # x = self.layer2(x)
        x, mask = SparseConv(256, 128, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(128, 128, 3, stride=2).cuda()(x, mask)
        x = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(128, 512, 1, stride=1).cuda()(x, mask)
        x = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x = nn.ReLU(inplace=True).cuda()(x)
        x, mask = SparseConv(512, 128, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(128, 128, 3).cuda()(x, mask)
        x = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(128, 512, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x = nn.ReLU(inplace=True).cuda()(x)
        x, mask = SparseConv(512, 128, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(128, 128, 3).cuda()(x, mask)
        x = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(128, 512, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x = nn.ReLU(inplace=True).cuda()(x)
        x, mask = SparseConv(512, 128, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(128, 128, 3).cuda()(x, mask)
        x = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(128, 512, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x = nn.ReLU(inplace=True).cuda()(x)

        m2 = x

        ###layer 3
        # x = self.layer3(attn) ##attn - replace x and attn

        x, mask = SparseConv(512, 256, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(256, 256, 3, stride=2).cuda()(x, mask)
        x = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(256, 1024, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x = nn.ReLU(inplace=True).cuda()(x)
        x, mask = SparseConv(1024, 256, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(256, 256, 3).cuda()(x, mask)
        x = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(256, 1024, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x = nn.ReLU(inplace=True).cuda()(x)
        x, mask = SparseConv(1024, 256, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(256, 256, 3).cuda()(x, mask)
        x = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(256, 1024, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x = nn.ReLU(inplace=True).cuda()(x)
        x, mask = SparseConv(1024, 256, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(256, 256, 3).cuda()(x, mask)
        x = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(256, 1024, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x = nn.ReLU(inplace=True).cuda()(x)
        x, mask = SparseConv(1024, 256, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(256, 256, 3).cuda()(x, mask)
        x = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(256, 1024, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x = nn.ReLU(inplace=True).cuda()(x)
        x, mask = SparseConv(1024, 256, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).cuda()(x)
        x, mask = SparseConv(256, 256, 3).cuda()(x, mask)
        x = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(256, 1024, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x = nn.ReLU(inplace=True).cuda()(x)

        m1 = x
        ##layer 4
        # x = self.layer4(x)
        x, mask = SparseConv(1024, 512, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(512, 512, 3, stride=2).cuda()(x, mask)
        x = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(512, 2048, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x = nn.ReLU(inplace=True).cuda()(x)
        x, mask = SparseConv(2048, 512, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(512, 512, 3).cuda()(x, mask)
        x = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(512, 2048, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x = nn.ReLU(inplace=True).cuda()(x)
        x, mask = SparseConv(2048, 512, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(512, 512, 3).cuda()(x, mask)
        x = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x, mask = SparseConv(512, 2048, 1).cuda()(x, mask)
        x = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()(x)
        x = nn.ReLU(inplace=True).cuda()(x)

        x = self.conv2(x)
        x = self.bn2(x)
        m0 = x
        bottleneck_block = BottleneckBlock(in_channels=1024, fmap_size=(16, 16), heads=4, out_channels=1024,
                                           pooling=False).cuda()
        x = bottleneck_block(x)
        # x = x + 0.7 * torch.rand(1,1024,16,16).cuda()

        # decoder
        x = self.decoder(x, m0, m1, m2, m3)
        x = self.conv3(x)
        x = self.bilinear(x)
        return x


class GPPatchMcResDis(pl.LightningModule):
    def __init__(self, hp):
        super(GPPatchMcResDis, self).__init__()
        assert hp['n_res_blks'] % 2 == 0, 'n_res_blk must be multiples of 2'
        self.n_layers = hp['n_res_blks'] // 2
        nf = hp['nf']

        cnn_f = [Conv2dBlock(1, nf, 7, 1, 3,
                             pad_type='reflect',
                             norm='none',
                             activation='none')]

        for i in range(self.n_layers - 1):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = np.min([nf * 2, 1024])

        nf_out = np.min([nf * 2, 1024])
        cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
        cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
        cnn_c = [Conv2dBlock(nf_out, hp['num_classes'], 1, 1,
                             norm='none',
                             activation='lrelu',
                             activation_first=True)]

        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_c = nn.Sequential(*cnn_c)

    def forward(self, x, y):
        assert (x.size(0) == y.size(0))
        feat = self.cnn_f(x)
        # print("@@@@f1")
        out = self.cnn_c(feat)
        # print(out.size())
        # print("@@@@f2")
        index = torch.LongTensor(range(out.size(0))).cuda()
        # print("@@@@f3")
        # print(index)
        # print(y)
        out = out[index, y, :, :]
        # print(out.size())
        # print("@@@@f4")
        # print(feat.size())
        # print("@@@@5")
        return out, feat

    def calc_dis_fake_loss(self, input_fake, input_label):
        resp_fake, gan_feat = self.forward(input_fake, input_label)
        total_count = torch.tensor(np.prod(resp_fake.size()),
                                   dtype=torch.float).cuda()
        fake_loss = torch.nn.ReLU()(1.0 + resp_fake).mean()
        correct_count = (resp_fake < 0).sum()
        fake_accuracy = correct_count.type_as(fake_loss) / total_count
        return fake_loss, fake_accuracy, resp_fake

    def calc_dis_real_loss_old(self, input_real, input_label):
        resp_real, gan_feat = self.forward(input_real, input_label)
        total_count = torch.tensor(np.prod(resp_real.size()), dtype=torch.float).cuda()
        real_loss = torch.nn.ReLU()(1.0 - resp_real).mean()
        correct_count = (resp_real >= 0).sum()
        real_accuracy = correct_count.type_as(real_loss) / total_count
        return real_loss, real_accuracy, resp_real

    def calc_dis_real_loss(self, input_real, input_label):  # shahaf version
        input_real.requires_grad = True
        resp_real, gan_feat = self.forward(input_real, input_label)
        # print("***1")
        # print(resp_real)
        # print(resp_real.size())
        total_count = torch.tensor(np.prod(resp_real.size()), dtype=torch.float).cuda()
        # print(total_count)
        # print("***2")
        real_loss = torch.nn.ReLU(inplace=False)(1.0 - resp_real).mean()
        correct_count = (resp_real >= 0).sum()
        real_accuracy = correct_count.type_as(real_loss) / total_count

        reg = self.calc_grad2(resp_real, input_real)
        return real_loss, real_accuracy, resp_real, reg

    def calc_gen_loss(self, input_fake, input_fake_label):
        resp_fake, gan_feat = self.forward(input_fake, input_fake_label)
        total_count = torch.tensor(np.prod(resp_fake.size()),
                                   dtype=torch.float).cuda()
        loss = -resp_fake.mean()
        correct_count = (resp_fake >= 0).sum()
        accuracy = correct_count.type_as(loss) / total_count
        return loss, accuracy, gan_feat

    def calc_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = \
            autograd.grad(outputs=d_out.mean(), inputs=x_in, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.sum() / batch_size
        return reg


class BasicBlockGeo(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, geoplanes=3):
        super(BasicBlockGeo, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            # norm_layer = encoding.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes + geoplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes + geoplanes, planes)
        self.bn2 = norm_layer(planes)
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes + geoplanes, planes, stride),
                norm_layer(planes),
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, g1=None, g2=None):
        identity = x
        if g1 is not None:
            x = torch.cat((x, g1), 1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if g2 is not None:
            out = torch.cat((g2, out), 1)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def convbnrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False, padding=1):
    """3x3 convolution with padding"""
    if padding >= 1:
        padding = dilation
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, groups=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=bias)
