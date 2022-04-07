import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
import os
import time
import csv
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloaders.nyu_dataloader import NYUDataset
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn import functional as F
import cv2
import torchvision
import wandb
from pytorch_lightning.loggers import WandbLogger
from os import listdir
from os.path import isfile, join

cudnn.benchmark = True
import os
from random import randrange
from datetime import datetime

from models import ResNet, GPPatchMcResDis
from metrics import AverageMeter, Result
from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo
from torch.utils.data import DataLoader, Dataset
import criteria
import utils
from torch.utils.data import random_split

os.environ["WANDB_CONFIG_DIR"] = 'config\\wandb'  # r"/home/ML_courses/03683533_2021/elad_almog_david/.config"
# wandb.init(project="lidar2ipad", entity="eldalm")
wandb_logger = WandbLogger(project="lidar2ipad", entity="eldalm")
# wandb.run.name = "full deep unet + normal loss"
# wandb.run.save()
global args, best_result, output_directory, train_csv, test_csv
IMG_MERGE = None

fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
              'delta1', 'delta2', 'delta3',
              'data_time', 'gpu_time', 'ssim_val']
best_result = Result()
best_result.set_to_worst()

import os

os.environ['TRANSFORMERS_CACHE'] = 'cache'


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


class GanModel(pl.LightningModule):
    def __init__(self, output_directory, cr):
        super().__init__()
        self.autoencoder = ResNet(layers=50, decoder="deconv3", output_size=(256, 256), in_channels=4, pretrained=True)
        hp = {'nf': 64, 'n_res_blks': 10, 'num_classes': 1}
        self.dis = GPPatchMcResDis(hp)

        self.criterion = cr
        self.img_merge = None
        self.output_directory = output_directory
        #self.automatic_optimization = False

    def forward(self, x):
        return self.autoencoder(x)

    def configure_optimizers(self):
        gen_optimizer = torch.optim.AdamW(self.autoencoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        dis_optimizer = torch.optim.AdamW(self.dis.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return gen_optimizer, dis_optimizer

    def freeze(self, model):
        for p in model.parameters():
            p.requires_grad = False

    def unfreeze(self, model):
        for p in model.parameters():
            p.requires_grad = True

    def training_step(self, batch, batch_idx, optimizer_idx):
        gen_optimizer, dis_optimizer = self.optimizers()
        average_meter = AverageMeter()
        end = time.time()
        # self.autoencoder.train()  # switch to train mode
        input, target = batch
        # input, target = input.cuda(), target.cuda()
        # torch.cuda.synchronize()
        data_time = time.time() - end

        # compute pred
        end = time.time()



        if optimizer_idx == 1:
            # dis update
            # dis_optimizer.zero_grad()
            # target.requires_grad_()
            l_real_pre, acc_r, resp_r, reg = self.dis.calc_dis_real_loss(target, torch.zeros(target.size(0)).long())  # .cuda())  ##label real = 1
            l_real = 0.5 * l_real_pre
            # self.manual_backward(l_real, retain_graph=True)
            reg = 10 * reg
            # self.manual_backward(reg)
            # reg.backward()

            with torch.no_grad():
                pred = self.autoencoder(input)
            l_fake_p, acc_f, resp_f = self.dis.calc_dis_fake_loss(pred.detach(), torch.zeros(pred.size(0)).long())  # .cuda())  ##label real = 1
            l_fake = 0.5 * l_fake_p  ##0.1 is parameter
            # self.manual_backward(l_fake)
            l_total = l_fake + l_real + reg
            acc = 0.5 * (acc_f + acc_r)
            self.log("dis_acc_real", acc_r)
            self.log("dis_acc_fake", acc_f)
            # self.log("dis_loss", l_total)
            # dis_optimizer.step()
            # torch.cuda.synchronize()
            self.log("dis_loss", l_total)
            return l_total

        if optimizer_idx == 0:
            ###gen update
            pred = self.autoencoder(input)
            #self.freeze(self.dis)
            l_adv_r, gacc_r, xr_gan_feat = self.dis.calc_gen_loss(pred, torch.zeros(pred.size(0)).long())  # .cuda())  ##label fake = 0
            #self.unfreeze(self.dis)
            # pred = pred.float()
            pred_normal = depth2norm_torch(pred)  # .cuda()
            target_normal = depth2norm_torch(target)  # .cuda()
            self.log("gen_discriminator_loss", l_adv_r)
            self.log("gen_acc", gacc_r)
            loss_gen = self.criterion(pred, target, pred_normal, target_normal, l_adv_r)
            self.log("gan_loss", loss_gen)
            return loss_gen

        # self.log("gan_loss", loss_gen)
        # gen_optimizer.zero_grad()
        # self.manual_backward(loss_gen)
        # loss_gen.backward()  # compute gradient and do SGD step
        # gen_optimizer.step()
        # torch.cuda.synchronize()

        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()
        return {"l_fake": l_fake, "l_real": l_real, "reg": reg, "loss_gen": loss_gen}

    def training_step_end2(self, training_step_output):
        with torch.autograd.set_detect_anomaly(True):

            gen_optimizer, dis_optimizer = self.optimizers()
            l_fake = training_step_output["l_fake"].mean()
            l_real = training_step_output["l_real"].mean()
            reg = training_step_output["reg"].mean()
            loss_gen = training_step_output["loss_gen"].mean()
            # l_fake,l_real, reg, loss_gen = training_step_output
            l_total = l_fake + l_real
            self.log("gan_loss", loss_gen)
            self.log("dis_loss", l_total)

            dis_optimizer.zero_grad()
            self.manual_backward(l_real, retain_graph=True)
            self.manual_backward(reg)
            self.manual_backward(l_fake)

            gen_optimizer.zero_grad()
            self.manual_backward(loss_gen)
            dis_optimizer.step()
            gen_optimizer.step()
    def set_image(self, input):
        self._img_merge = input

    def validation_step(self, val_batch, batch_idx):
        average_meter = AverageMeter()
        end = time.time()
        input, target = val_batch
        # input, target = input.cuda(), target.cuda()
        # torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = self.autoencoder(input)
        # torch.cuda.synchronize()
        gpu_time = time.time() - end

        # dis_loss = dis_loss + get_dis_loss(pred.data, target.data, dis)

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # save 8 images for visualization
        skip = 50  # randrange(30,50) # 50

        rgb = input[:, :3, :, :]
        depth = input[:, 3:, :, :]
        normal = depth2norm_torch(pred)
        input_normal = depth2norm_torch(depth)
        target_normal = depth2norm_torch(target)
        # print("normal")
        # print(normal.size())
        # print(depth.size())

        if batch_idx == 0:
            yyy = utils.merge_into_row_with_gt(rgb, depth, target, pred, normal, input_normal, target_normal)
            self.img_merge = yyy
            print("h")

        elif (batch_idx < 8 * skip) and (batch_idx % skip == 0):
            row = utils.merge_into_row_with_gt(rgb, depth, target, pred, normal, input_normal, target_normal)
            #self.img_merge = utils.add_row(self.img_merge, row)
            filename = self.output_directory + '/comparison_' + str(self.global_step) +'_'+str(batch_idx) + '_'+ '.png'
            utils.save_image(row, filename)
            #grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.log({"image_comparison": [wandb.Image(row)]})
            #elif batch_idx == 8 * skip:
        #    filename = self.output_directory + '/comparison_' + str(self.current_epoch) + '.png'
        #    utils.save_image(self.img_merge, filename)

        avg = average_meter.average()
        self.log("performance", {'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                                 'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                                 'data_time': avg.data_time, 'gpu_time': avg.gpu_time})

        metrics_str = "delta3={d3}_rmse={rmse}_delta1={d1}_delta2={d2}".format(d3 =round(avg.delta3,3), rmse = round(avg.rmse,3), d2= round(avg.delta2,3), d1=round(avg.delta1,3))
        #if self.global_step % 1000 == 0:
        #

        files_in_dir = [f for f in listdir(self.output_directory) if isfile(join(self.output_directory, f))]
        is_first = True
        for f in files_in_dir:
            rmse_idx = f.find("rmse")
            if rmse_idx > -1:
                is_first = False
                last_rmse_str = f[rmse_idx+5:rmse_idx+9]
                if self.is_float(last_rmse_str):
                    last_rmse = float(last_rmse_str)
                    if avg.rmse < last_rmse:
                        self.trainer.save_checkpoint(self.output_directory + "/cp_" + str(self.global_step) + "_" + metrics_str + ".ckpt")
                        os.remove(self.output_directory +"/"+f)
        if is_first:
            self.trainer.save_checkpoint(self.output_directory + "/cp_" + str(self.global_step) + "_" + metrics_str + ".ckpt")

    def is_float(self,string):
        try:
            return float(string) and '.' in string  # True if string is a number contains a dot
        except ValueError:  # String is not a number
            return False



        #self.trainer.save_checkpoint(self.output_directory +"/cp_"+ str(self.global_step) +"_"+ metrics_str + ".ckpt")
        #
        # self.logger.experiment.add_image("comparison", self.img_merge)
        # filename = output_directory + '/comparison_' + str(epoch) + '_outside.png'
        # utils.save_image(self.img_merge, filename)
        #img_merge = utils.merge_into_row_with_gt(rgb, depth, target, pred, normal, input_normal, target_normal)
        #return {"loss" : 1, "avg":avg}






def main(args):
    traindir = os.path.join('new_data', 'ARKit', 'train')  ##elad edit
    valdir = os.path.join('new_data', 'ARKit', 'train')  ##elad edit
    train_loader = None
    val_loader = None
    max_depth = args.max_depth if args.max_depth >= 0.0 else np.inf
    sparsifier = UniformSampling(num_samples=args.num_samples, max_depth=max_depth)

    if not args.evaluate:
        train_dataset = NYUDataset(traindir, type='train', modality=args.modality, sparsifier=sparsifier)
    val_dataset = NYUDataset(valdir, type='val', modality=args.modality, sparsifier=sparsifier)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, sampler=None, )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.workers,
                                             pin_memory=True)

    criterion = criteria.MaskedL1Loss()

    # create results folder, if not already exists
    output_directory = utils.get_output_directory(args)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # print("**********output directory")
    # print(output_directory)

    in_channels = len(args.modality)
    # model = ResNet(layers=50, decoder=args.decoder, output_size=train_loader.dataset.output_size, in_channels=in_channels, pretrained=args.pretrained)
    # hp = {'nf': 64, 'n_res_blks': 10, 'num_classes': 1}
    # dis = GPPatchMcResDis(hp)
    # model = model.cuda()
    # dis = dis.cuda()
    model = GanModel(output_directory, criterion)
    # model = model.cuda()

    #checkpoint_callback = ModelCheckpoint(dirpath=output_directory, monitor="")

    # training
    trainer = pl.Trainer(gpus=[1,2, 3], num_nodes=1, val_check_interval=0.1, logger=wandb_logger,accelerator="dp")
    # print("4")
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    args = utils.parse_command()
    main(args)
