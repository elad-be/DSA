import os
import time
import csv
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn import functional as F
import cv2
from torchvision import transforms as T
from PIL import Image
import imageio
from torchvision.utils import save_image

def get_gaussian_kernel(size=3):
    kernel = cv2.getGaussianKernel(size, 0).dot(cv2.getGaussianKernel(size, 0).T)
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
    return kernel

def depth2norm_torch(d_im, ksize=3):
    """
    calculate the normals from a depth map for batched tensors
    :param d_im: input tensor of shape (\text{minibatch} , \text{in\_channels} , iH , iW)(minibatch,in_channels,iH,iW)
    :param ksize:
    :return: an RGB image. each pixel represents the surface normal direction (3 coordinates => R, G, B)
    """

    kerx = torch.Tensor((
        [0, 0, 0],
        [-800, 0, 800],
        [0, 0, 0])).unsqueeze(0).unsqueeze(0)

    kery = torch.Tensor(np.array((
        [0, -800, 0],
        [0, 0, 0],
        [0, 800, 0]))).unsqueeze(0).unsqueeze(0)

    gaussian_kernel = get_gaussian_kernel()
    if d_im.is_cuda:
        kerx = kerx.cuda()
        kery = kery.cuda()
        gaussian_kernel = gaussian_kernel.cuda()

    zx = F.conv2d(d_im, weight=kerx, padding='same')
    zy = F.conv2d(d_im, weight=kery, padding='same')
    #     print(zx)
    #     plt.imshow(zx)
    #     plt.show()
    #     plt.imshow(zy)
    #     plt.show()

    normal = torch.cat((-zx, -zy, torch.ones_like(d_im)), dim=1)
    normal = normal.transpose(0, 1).div(torch.norm(normal, dim=1)).transpose(0, 1)
    # offset and rescale values to be in 0-255
    normal = normal.flip(1)
    normal = F.conv2d(normal.view(-1, 1, normal.size(2), normal.size(3)),
                      weight=gaussian_kernel, padding='same').view(-1, 3, normal.size(2), normal.size(3))
    return normal



depth = "/home/ML_courses/03683533_2021/elad_almog_david/s2d/datasets/lidar_preped/41048190_3419.682.exr" #os.path.join("datasets","lidar_preped",self.imgs[index][20:-4] + ".exr")
rgb = "/home/ML_courses/03683533_2021/elad_almog_david/s2d/datasets/rgb_preped/41048190_3419.682.jpg"

rgb = Image.open(rgb)
rgb = T.ToTensor()(rgb)

depth = cv2.imread(depth,-1)
depth = T.ToTensor()(depth)


rgbd = torch.cat((rgb, depth),0)
depth = depth.unsqueeze(0)
norm = depth2norm_torch(depth)
norm = norm.squeeze(0)


print(norm.shape)
save_image(norm, 'normtry.jpg')