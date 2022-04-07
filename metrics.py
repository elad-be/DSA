import torch
import math
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2
import math
import lpips
import matplotlib.pyplot as plt



def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)

lpips_obj = lpips.LPIPS(net='alex', version='0.1').cuda()

class Result(object):
    def __init__(self):
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0
        self.ssim = 0.0
        self.psnr = 0.0
        self.loss_fn = lpips_obj
        self.lpips = 0.0

    def set_to_worst(self):
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0
        self.psnr = 0.0

    def update(self, irmse, imae, mse, rmse, mae, absrel, lg10, delta1, delta2, delta3, gpu_time, data_time, ssim_val,
               psnr,lpips):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
        self.data_time, self.gpu_time = data_time, gpu_time
        self.ssim_val = ssim_val
        self.psnr = psnr
        self.lpips = lpips


    def unnormalize_ARKit(self, im):
        mean = 1282.0 # 0.525 #
        std = 306.0 # 0.28 #
        return im * std + mean

    def unnormalize_hyper(self, im):
        mean = 0.525 #
        std = 0.28 #
        return im * std + mean * 1e3

    def evaluate(self, valid_mask, output, target):
        output = self.unnormalize_ARKit(output)
        target = self.unnormalize_ARKit(target)
        # valid_mask = target > 0
        # output = output[valid_mask]
        # target = target[valid_mask]
        abs_diff = (output - target).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())

        self.rmse = math.sqrt(self.mse)

        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output) - log10(target)).abs().mean())
        self.absrel = float((abs_diff / target).mean())

        maxRatio = torch.max(output / (target+1e-6), target / (output+1e-6))
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())

        output_numpy = output.cpu().detach().numpy()
        target_numpy = target.cpu().detach().numpy()
        self.ssim_val = ssim(target_numpy.squeeze(), output_numpy.squeeze(),
                             data_range=output_numpy.max() - output_numpy.min(),
                             channel_axis=0)
        self.psnr = calculate_psnr(output_numpy, target_numpy)
        self.data_time = 0
        self.gpu_time = 0

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())

        # Load images
        img0 = output  # RGB image from [-1,
        img1 = target

        img0 = img0.cuda()
        img1 = img1.cuda()

        # Compute distance
        dist01 = self.loss_fn.forward(img0, img1)
        self.lpips = dist01.mean().item()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
        self.sum_data_time, self.sum_gpu_time = 0, 0
        self.sum_ssim_val = 0.0
        self.sum_psnr = 0.0
        self.sum_lpips = 0.0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n

        self.sum_irmse += n * result.irmse
        self.sum_imae += n * result.imae
        self.sum_mse += n * result.mse
        self.sum_rmse += n * result.rmse
        self.sum_mae += n * result.mae
        self.sum_absrel += n * result.absrel
        self.sum_lg10 += n * result.lg10
        self.sum_delta1 += n * result.delta1
        self.sum_delta2 += n * result.delta2
        self.sum_delta3 += n * result.delta3
        self.sum_data_time += n * data_time
        self.sum_gpu_time += n * gpu_time
        self.sum_ssim_val += n * result.ssim_val
        self.sum_psnr += n * result.psnr
        self.sum_lpips += n* result.lpips

    def average(self):
        avg = Result()
        avg.update(
            self.sum_irmse / self.count, self.sum_imae / self.count,
            self.sum_mse / self.count, self.sum_rmse / self.count, self.sum_mae / self.count,
            self.sum_absrel / self.count, self.sum_lg10 / self.count,
            self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count,
            self.sum_gpu_time / self.count, self.sum_data_time / self.count,
            self.sum_ssim_val / self.count,
            self.sum_psnr / self.count,
            self.sum_lpips / self.count)

        return avg
