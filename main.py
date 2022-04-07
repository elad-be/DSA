import os
import time
import csv
import numpy as np
from os import listdir
from os.path import isfile, join
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn import functional as F
import cv2
import wandb
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
print("try")
os.environ["WANDB_CONFIG_DIR"] = 'config\\wandb'#r"/home/ML_courses/03683533_2021/elad_almog_david/.config"

#wandb.run.name = "full deep unet + normal loss"
#wandb.run.save()
args = utils.parse_command()
wandb.init(project="lidar2ipad", entity="eldalm", name=args.exp_name)
print(args)

fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
                'delta1', 'delta2', 'delta3',
                'data_time', 'gpu_time', 'ssim_val', 'psnr']
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
    
    
def create_data_loaders(args):
    # Data loading code
    print("=> creating data loaders ...")
    traindir = os.path.join("/home/cgvlab972/miniconda3/s2d/new_data/ARKit/train/rgb")####os.path.join("..","..","..","..","ssd","ARKit","train")##elad edit
    valdir = os.path.join("/home/cgvlab972/miniconda3/s2d/new_data/ARKit/train/rgb")#####os.path.join("..","..","..","..","ssd","ARKit","train")   ##elad edit
    train_loader = None
    val_loader = None

    # sparsifier is a class for generating random sparse depth input from the ground truth
    sparsifier = None
    max_depth = args.max_depth if args.max_depth >= 0.0 else np.inf
    if args.sparsifier == UniformSampling.name:
        sparsifier = UniformSampling(num_samples=args.num_samples, max_depth=max_depth)
    elif args.sparsifier == SimulatedStereo.name:
        sparsifier = SimulatedStereo(num_samples=args.num_samples, max_depth=max_depth)
    print("***************************")
    if args.data == 'nyudepthv2':
        from dataloaders.nyu_dataloader import NYUDataset
        if not args.evaluate:
            train_dataset = NYUDataset(traindir, type='train',
                modality=args.modality, sparsifier=sparsifier)
        val_dataset = NYUDataset(valdir, type='val',
            modality=args.modality, sparsifier=sparsifier)

    elif args.data == 'kitti':
        from dataloaders.kitti_dataloader import KITTIDataset
        if not args.evaluate:
            train_dataset = KITTIDataset(traindir, type='train',
                modality=args.modality, sparsifier=sparsifier)
        val_dataset = KITTIDataset(valdir, type='val',
            modality=args.modality, sparsifier=sparsifier)
            
    
    #ds_folder = os.path.join('datasets','ipad_prepared')        
    #train_ds = UniDataset([os.path.join(ds_folder,i) for i in sorted(os.listdir(ds_folder))[1001:]],transform=transform)

    else:
        raise RuntimeError('Dataset not found.' +
                           'The dataset must be either of nyudepthv2 or kitti.')
    #ds_gwb = ImageFolder("dataset/ipad_preped", tf)
    #split_lengths = [16650,1850]
    
    #train_dataset, val_dataset = random_split(ds_gwb, split_lengths)
    

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    # put construction of train loader here, for those who are interested in testing only
    if not args.evaluate:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None,
            worker_init_fn=lambda work_id:np.random.seed(work_id))
            # worker_init_fn ensures different sampling patterns for each data loading thread

    print("=> data loaders created.")
    return train_loader, val_loader
    
    
def main():
    global args, best_result, output_directory, train_csv, test_csv
    
    #print("4444444444444444444444444")
    # evaluation mode
    start_epoch = 0
    
    if args.evaluate:
        assert os.path.isfile(args.evaluate), \
        "=> no best model found at '{}'".format(args.evaluate)
        print("=> loading best model '{}'".format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        output_directory = os.path.dirname(args.evaluate)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        _, val_loader = create_data_loaders(args)
        args.evaluate = True
        validate(val_loader, model, checkpoint['epoch'], write_to_file=False)
        return

    # optionally resume from a checkpoint
    elif args.resume:
        chkpt_path = args.resume
        assert os.path.isfile(chkpt_path), \
            "=> no checkpoint found at '{}'".format(chkpt_path)
        print("=> loading checkpoint '{}'".format(chkpt_path))
        checkpoint = torch.load(chkpt_path)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        output_directory = os.path.dirname(os.path.abspath(chkpt_path))
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        train_loader, val_loader = create_data_loaders(args)
        args.resume = True

    # create new model
    else:
        
        train_loader, val_loader = create_data_loaders(args)
        print("=> creating Model ({}-{}) ...".format(args.arch, args.decoder))
        in_channels = len(args.modality)
        if args.arch == 'resnet50':
            model = ResNet(layers=18, decoder=args.decoder, output_size=train_loader.dataset.output_size,
                in_channels=in_channels, pretrained=args.pretrained)
            hp = {'nf': 64,'n_res_blks': 10, 'num_classes': 1} 
            dis = GPPatchMcResDis(hp)
        elif args.arch == 'resnet18':
            model = ResNet(layers=18, decoder=args.decoder, output_size=train_loader.dataset.output_size,
                in_channels=in_channels, pretrained=args.pretrained)
        print("=> model created.")


        # chkpt_path = "/home/cgvlab972/miniconda3/s2d/results/nyudepthv2.sparsifier=uar.samples=10.modality=rgbd.arch=resnet50.decoder=deconv3.criterion=l1.lr=1e-05.bs=2.pretrained=True/cp_delta3=0.97_rmse=0.148_delta1=0.892_delta2=0.95.pth.tar"
        # checkpoint = torch.load(chkpt_path)
        # model = checkpoint['generator']
        # dis = checkpoint['discriminator']


        optimizer = torch.optim.Adam(model.parameters(),lr = args.lr,weight_decay = args.weight_decay)
        gen_optimizer = torch.optim.AdamW(model.parameters(),lr = args.lr,weight_decay = args.weight_decay)
        dis_optimizer = torch.optim.AdamW(dis.parameters(),lr = args.lr,weight_decay = args.weight_decay)


        #model = torch.nn.DataParallel(model,device_ids=[1,2]).cuda() # for multi-gpu training
        #dis = torch.nn.DataParallel(dis,device_ids=[1,2]).cuda()
        model = model.cuda()
        dis = dis.cuda()
    print("111111111111111111111111111")

    # define loss function (criterion) and optimizer
    if args.criterion == 'l2':
        criterion = criteria.MaskedMSELoss().cuda()
    elif args.criterion == 'l1':
        criterion = criteria.MaskedL1Loss().cuda()

    # create results folder, if not already exists
    output_directory = utils.get_output_directory(args)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print("**********output directory")
    print(output_directory)
    
    train_csv = os.path.join(output_directory, 'train.csv')
    test_csv = os.path.join(output_directory, 'test.csv')
    best_txt = os.path.join(output_directory, 'best.txt')
    # create new csv files with only header
    if not args.resume:
        with open(train_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        with open(test_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    for epoch in range(start_epoch, args.epochs):
        
        utils.adjust_learning_rate(optimizer, epoch, args.lr)
        wandb.config = {
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "decoder" : args.decoder,
        "criteria" :args.criterion
        }
        
        train(train_loader, model,dis, criterion, gen_optimizer, dis_optimizer, optimizer, epoch, val_loader) # train for one epoch

        result, img_merge = validate(val_loader, model, epoch,dis,gen_optimizer, dis_optimizer, during_trainig = False) # evaluate on validation set
        

        # remember best rmse and save checkpoint
        is_best = result.rmse < best_result.rmse
        if is_best:
            best_result = result
            with open(best_txt, 'w') as txtfile:
                txtfile.write("epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
                    format(epoch, result.mse, result.rmse, result.absrel, result.lg10, result.mae, result.delta1, result.gpu_time))
            if img_merge is not None:
                img_filename = output_directory + '/comparison_best.png'
                utils.save_image(img_merge, img_filename)

        utils.save_checkpoint({
            'args': args,
            'epoch': epoch,
            'arch': args.arch,
            'model': model,
            'best_result': best_result,
            'optimizer' : optimizer,
        }, is_best, epoch, output_directory)


def train(train_loader, model,dis, criterion, gen_optimizer, dis_optimizer, optimizer, epoch, val_loader):
    print("start train")
    average_meter = AverageMeter()
    model.train() # switch to train mode
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        rnd_num = randrange(1,10)
        #if(rnd_num % 3 == 0):
            #rgb_tmp = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0))
            #utils.save_image(rgb_tmp, "aug_{0}.png".format(datetime.timestamp(datetime.now())))
            
        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end
        valid_mask = (input[:, 3:, :, :] > -4.189542293548584).detach()
        

        # compute pred
        end = time.time()
        
        #dis update

        #target = target * valid_mask
        rgb = input[:, :3, :, :]
        print(rgb.size())
        target_to_dis = torch.cat((rgb, target),1)

        
        dis_optimizer.zero_grad()
        target.requires_grad_()
        l_real_pre, acc_r, resp_r, reg = dis.calc_dis_real_loss(target, torch.zeros(target.size(0)).long().cuda()) ##label real = 1
        
        l_real = 0.5 * l_real_pre
        l_real.backward(retain_graph=True)
        reg = 10*reg
        reg.backward()
        pred, pred_prop1, pred_prop2 = model(input)
        pred_to_dis = torch.cat((rgb, pred),1)
        #pred = pred * valid_mask
        l_fake_p, acc_f, resp_f = dis.calc_dis_fake_loss((pred).detach(),torch.zeros(pred.size(0)).long())##label real = 1
        l_fake = 0.5  * l_fake_p ##0.1 is parameter
        l_fake.backward()
        l_total = l_fake + l_real
        acc = 0.5 * (acc_f + acc_r)
        wandb.log({"dis_acc_real": acc_r})
        wandb.log({"dis_acc_fake": acc_f})
        wandb.log({"dis_loss": l_total})
        
        #loss_dis.backward() # compute gradient and do SGD step
        dis_optimizer.step()
        torch.cuda.synchronize()
        
        
        ###gen update
        #pred = model(input)

        l_adv_r, gacc_r, xr_gan_feat = dis.calc_gen_loss(pred, torch.zeros(pred.size(0)).long()) ##label fake = 0
            #pred = pred.float()
        pred_normal = depth2norm_torch(pred).cuda()
        target_normal = depth2norm_torch(target).cuda()
        wandb.log({"gen_discriminator_loss": l_adv_r})
        wandb.log({"gen_acc": gacc_r})
        valid_mask = (input[:, 3:, :, :] > -4.189542293548584).detach()
        loss_gen = criterion(valid_mask, pred, target, pred_normal, target_normal, l_adv_r, pred_prop1, pred_prop2)
        
        wandb.log({"gan_loss": loss_gen})
        gen_optimizer.zero_grad()
        loss_gen.backward() # compute gradient and do SGD step
        gen_optimizer.step()
        torch.cuda.synchronize()
        
        
        
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()

        result.evaluate(valid_mask, pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()
        
        
        

        print("*****************before val",i)
        if (i + 1) % args.print_freq == 0:
            print("****************in val")
            result2, img_merge2 = validate(val_loader, model, str(epoch) + " " + str(i+1), dis,gen_optimizer, dis_optimizer, during_trainig =False)
            print('=> output: {}'.format(output_directory))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                  epoch, i+1, len(train_loader), data_time=data_time,
                  gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()
    with open(train_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
            'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
            'gpu_time': avg.gpu_time, 'data_time': avg.data_time, 'ssim_val': avg.ssim_val, 'psnr': avg.psnr})
        
def get_dis_loss(pred, target, dis):
    l_real_pre, acc_r, resp_r, reg = dis.calc_dis_real_loss(target, torch.tensor((0,0,)).cuda()) ##label real = 1
    l_real = 0.5 * l_real_pre
    reg = 10*reg
    l_fake_p, acc_f, resp_f = dis.calc_dis_fake_loss(pred,torch.tensor((0,0,)).cuda())##label real = 1
    l_fake = 0.5  * l_fake_p ##0.1 is parameter
    l_total = l_fake + l_real
    return l_total
    #wandb.log({"dis_loss_validation": l_total})

def validate(val_loader, model, epoch,dis,gen_optimizer, dis_optimizer, write_to_file=True, during_trainig = False):
    average_meter = AverageMeter()
    model.eval() # switch to evaluate mode
    end = time.time()
    dis_loss = 0
    for i, (input, target) in enumerate(val_loader):
        if (i > int(len(val_loader)/3) and during_trainig):
            break
        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end
        valid_mask = (input[:, 3:, :, :] > -4.189542293548584).detach()
        ##target = target* valid_mask
        
                   

        # compute output
        end = time.time()
        with torch.no_grad():
            pred, pred_prop1, pred_prop2 = model(input)
            # pred, pred_prop1, pred_prop2  = model(input)  # with cspn
            #pred = pred * valid_mask
        torch.cuda.synchronize()
        gpu_time = time.time() - end
        #dis_loss = dis_loss + get_dis_loss(pred.data, target.data, dis)
        
        # measure accuracy and record loss
        result = Result()
        result.evaluate(valid_mask, pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()


        # save 8 images for visualization
        skip = 1 #randrange(30,50) # 50
        if args.modality == 'd':
            img_merge = None
        else:
            if args.modality == 'rgb':
                rgb = input
            elif args.modality == 'rgbd':
                rgb = input[:,:3,:,:]
                depth = input[:,3:,:,:]
                normal = depth2norm_torch(pred)
                input_normal = depth2norm_torch(depth)
                target_normal = depth2norm_torch(target)
                #print("normal")
                #print(normal.size())
                #print(depth.size())
            

            if i == 0:
                if args.modality == 'rgbd':
                    img_merge = utils.merge_into_row_with_gt(rgb, depth, target, pred, normal, input_normal, target_normal)

                else:
                    img_merge = utils.merge_into_row(rgb, target, pred)
            elif (i < 8*skip) and (i % skip == 0):
                if args.modality == 'rgbd':
                    row = utils.merge_into_row_with_gt(rgb, depth, target, pred, normal, input_normal, target_normal)
                else:
                    row = utils.merge_into_row(rgb, target, pred)
                img_merge = utils.add_row(img_merge, row)
            elif i == 8*skip:
                filename = output_directory + '/comparison_' + str(epoch) + '.png'
                utils.save_image(img_merge, filename)

        if (i+1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))


    avg = average_meter.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))
    
    dis_loss = 0#dis_loss / 1000.0
    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
        
        wandb.log({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time, 'ssim': avg.ssim_val, 'psnr': avg.psnr, "dis_loss_validation": dis_loss, "comparison": wandb.Image(img_merge)})
        filename = output_directory + '/comparison_' + str(epoch) + '_outside.png'
        utils.save_image(img_merge, filename)



    ###save checkpoint
    metrics_str = "delta3={d3}_rmse={rmse}_delta1={d1}_delta2={d2}".format(d3=round(avg.delta3, 3),
                                                                           rmse=round(avg.rmse, 3),
                                                                           d2=round(avg.delta2, 3),
                                                                           d1=round(avg.delta1, 3))
    files_in_dir = [f for f in listdir(output_directory) if isfile(join(output_directory, f))]
    is_first = True
    for f in files_in_dir:
        rmse_idx = f.find("rmse")
        if rmse_idx > -1:
            is_first = False
            last_rmse_str = f[rmse_idx + 5:rmse_idx + 9]
            if is_float(last_rmse_str):
                last_rmse = float(last_rmse_str)
                if avg.rmse < last_rmse:
                    state_dict={"generator": model, "discriminator":dis, "gen_optimizer":gen_optimizer, "dis_oprimizer": dis_optimizer}
                    torch.save(state_dict, output_directory + "/cp_" + metrics_str + '.pth.tar')
                    os.remove(output_directory + "/" + f)

    if is_first:
        state_dict = {"generator": model, "discriminator": dis, "gen_optimizer": gen_optimizer,
                      "dis_oprimizer": dis_optimizer}
        torch.save(state_dict, output_directory + "/cp_" + metrics_str + '.pth.tar')

    return avg, img_merge

def is_float(string):
    try:
        return float(string) and '.' in string  # True if string is a number contains a dot
    except ValueError:  # String is not a number
        return False


if __name__ == '__main__':
    main()
