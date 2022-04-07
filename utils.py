import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
import torchvision
from torchvision import transforms as T
from datetime import datetime
import png
cmap = plt.cm.turbo

class FoldingLayer(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(FoldingLayer, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.unfold = nn.Unfold(self.kernel_size, stride=self.stride, padding=self.padding)

    def forward(self, x):
        B, C, H, W = x.size()
        unfold_x = self.unfold(x).view(B, C, self.kernel_size ** 2, -1).permute(0, 3, 2, 1).contiguous().view(-1,
                                                                                                              self.kernel_size ** 2,
                                                                                                              C)
        return unfold_x

def parse_command():
    model_names = ['resnet18', 'resnet50']
    loss_names = ['l1', 'l2']
    data_names = ['nyudepthv2', 'kitti']
    from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo
    sparsifier_names = [x.name for x in [UniformSampling, SimulatedStereo]]
    from models import Decoder
    decoder_names = Decoder.names
    from dataloaders.dataloader import MyDataloader
    modality_names = MyDataloader.modality_names

    import argparse
    parser = argparse.ArgumentParser(description='Sparse-to-Dense')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
    parser.add_argument('--data', metavar='DATA', default='nyudepthv2',
                        choices=data_names,
                        help='dataset: ' + ' | '.join(data_names) + ' (default: nyudepthv2)')
    parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb', choices=modality_names,
                        help='modality: ' + ' | '.join(modality_names) + ' (default: rgb)')
    parser.add_argument('-s', '--num-samples', default=0, type=int, metavar='N',
                        help='number of sparse depth samples (default: 0)')
    parser.add_argument('--max-depth', default=-1.0, type=float, metavar='D',
                        help='cut-off depth of sparsifier, negative values means infinity (default: inf [m])')
    parser.add_argument('--sparsifier', metavar='SPARSIFIER', default=UniformSampling.name, choices=sparsifier_names,
                        help='sparsifier: ' + ' | '.join(sparsifier_names) + ' (default: ' + UniformSampling.name + ')')
    parser.add_argument('--decoder', '-d', metavar='DECODER', default='deconv2', choices=decoder_names,
                        help='decoder: ' + ' | '.join(decoder_names) + ' (default: deconv2)')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run (default: 20)')
    parser.add_argument('-c', '--criterion', metavar='LOSS', default='l1', choices=loss_names,
                        help='loss function: ' + ' | '.join(loss_names) + ' (default: l1)')
    parser.add_argument('-b', '--batch-size', default=8, type=int, help='mini-batch size (default: 8)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate (default 0.01)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=1000, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', type=str, default='',
                        help='evaluate model on validation set')
    parser.add_argument('-n', '--exp-name', type=str, default=None,
                        help='name of experiment')
    parser.add_argument('--no-pretrain', dest='pretrained', action='store_false',
                        help='not to use ImageNet pre-trained weights')
    parser.set_defaults(pretrained=True)
    args = parser.parse_args()
    if args.modality == 'rgb' and args.num_samples != 0:
        print("number of samples is forced to be 0 when input modality is rgb")
        args.num_samples = 0
    if args.modality == 'rgb' and args.max_depth != 0.0:
        print("max depth is forced to be 0.0 when input modality is rgb/rgbd")
        args.max_depth = 0.0
    return args

def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch-1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)

def adjust_learning_rate(optimizer, epoch, lr_init):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    lr = lr_init * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_output_directory(args):
    output_directory = os.path.join('results',
        '{}.sparsifier={}.samples={}.modality={}.arch={}.decoder={}.criterion={}.lr={}.bs={}.pretrained={}'.
        format(args.data, args.sparsifier, args.num_samples, args.modality, \
            args.arch, args.decoder, args.criterion, args.lr, args.batch_size, \
            args.pretrained))
    return output_directory


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C


def merge_into_row(input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])
    
    return img_merge


def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred, normal, input_normal, target_normal):
    invTrans = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/51.75028439, 1/50.10456233, 1/50.28064866 ]), T.Normalize(mean = [ -148.47785343, -129.2452198, -105.22572954 ], std = [ 1., 1., 1. ]),])
    #inv_normalize = T.Normalize(mean=[ -148.47785343/51.75028439, -129.2452198/50.10456233, -105.22572954/50.28064866],std=[1/51.75028439, 1/50.10456233, 1/50.28064866])
    #inv_normalize = T.Normalize(mean=[ -0.4297/81.4061, -0.3969/80.8498, -0.3645/80.4092],std=[1/81.4061, 1/80.8498, 1/80.4092])
    inv_normalize = T.Normalize(mean=[ -0.4255/0.4144, -0.3926/0.4110, -0.3604/0.4086],std=[1/0.4144, 1/0.4110, 1/0.4086])

    inv_tensor = inv_normalize(input)
    rgb = 255* np.transpose(np.squeeze(inv_tensor.cpu().numpy()), (1,2,0)) # H, W, C
    #rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0))

    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())

    #inv_normalize = T.Normalize(mean=[-0.4255 / 0.4144, -0.3926 / 0.4110, -0.3604 / 0.4086],td=[1 / 0.4144, 1 / 0.4110, 1 / 0.4086])
    #inv_tensor = inv_normalize(input)



    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())
    #a = depth_pred.data.cpu().numpy()
    normal_cpu = 255 * np.transpose(np.squeeze(normal.cpu().numpy()), (1,2,0)) # H, W, C
    input_normal_cpu = 255 * np.transpose(np.squeeze(input_normal.cpu().numpy()), (1,2,0)) # H, W, C
    target_normal_cpu = 255 * np.transpose(np.squeeze(target_normal.cpu().numpy()), (1,2,0)) # H, W, C
    substraction = depth_target_cpu - depth_pred_cpu
    
    save_depth("pc/dd.png", depth_pred_cpu)
    



    d_min = min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    #d_min = -3.6960785388946533
    #d_max = 60.85293960571289
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    sub_col = colored_depthmap(substraction, d_min, d_max)
    
    
    now =datetime.now()
    ts = str(datetime.timestamp(now))
    #save_image(rgb, "pc/rgb_" + ts + ".png")
    #save_image(depth_pred_col, "pc/depth_pred_" + ts + ".png")
    #save_image(depth_target_col, "pc/depth_target_" + ts + ".png")
    
    

    img_merge = np.hstack([rgb, depth_input_col, depth_target_col, depth_pred_col,input_normal_cpu, target_normal_cpu, normal_cpu, sub_col])

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])

def save_depth(path, im):
    """Saves a depth image (16-bit) to a PNG file.
    From the BOP toolkit (https://github.com/thodan/bop_toolkit).

    :param path: Path to the output depth image file.
    :param im: ndarray with the depth image to save.
    """
    if not path.endswith(".png"):
        raise ValueError('Only PNG format is currently supported.')

    im[im > 65535] = 65535
    im_uint16 = np.round(im).astype(np.uint16)

    # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
    w_depth = png.Writer(im.shape[1], im.shape[0], greyscale=True, bitdepth=16)
    with open(path, 'wb') as f:
        w_depth.write(f, np.reshape(im_uint16, (-1, im.shape[1]))) 
        
        
def save_image2(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint16'))
    img_merge.save(filename)


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)