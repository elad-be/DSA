import numpy as np
import dataloaders.transforms as transforms
from dataloaders.dataloader import MyDataloader
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import torch
import torchvision
from .noises import compose_noises, TRANSFORM_DICT

iheight, iwidth = 512, 512  # raw image size


class NYUDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgb'):
        super(NYUDataset, self).__init__(root, type, sparsifier, modality)
        self.output_size = (256, 256)

    def train_transform(self, rgb, depth, target):
        # torchvision.utils.save_image(rgb, 'rgbbbbbb.png')

        a = torch.rand(1).item()
        b = torch.rand(1).item()
        c = torch.rand(1).item()
        d = torch.rand(1).item()
        p = 0.5

        l = 256 * a
        u = 256 * b
        w = (256 - a * 256) * c
        h = (256 - b * 256) * d
        # print("!~!~!~!~ h: " + h)
        # print("!~!~!~!~! w: " + w)

        s = np.random.uniform(1.0, 1.5)  # random scaling
        s = int(256 * s)
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = float(torch.randint(0, 2, (1,)).item())

        tr_rgb = T.Compose([

            # T.Resize(250.0 / iheight),  # this is for computational efficiency, since rotation can be slow
            # T.RandomRotation(abs(angle)),
            T.Resize((s, s), TF.InterpolationMode.NEAREST),  ###add nearest
            T.CenterCrop(self.output_size),
            T.RandomHorizontalFlip(do_flip),
            T.Normalize(mean=(148.47785343, 129.2452198, 105.22572954), std=(51.75028439, 50.10456233, 50.28064866))
            # T.Normalize(mean=(0.4255, 0.3926, 0.3604), std=(0.4144, 0.4110, 0.4086))
        ])

        tr_d = T.Compose([
            # T.Resize(250.0 / iheight),  # this is for computational efficiency, since rotation can be slow
            # T.RandomRotation(abs(angle)),
            T.Resize((s, s), TF.InterpolationMode.NEAREST),  ###add nearest
            T.CenterCrop(self.output_size),
            T.RandomHorizontalFlip(do_flip),
            T.Normalize((1282.0,), (306.0,)),
            # T.Normalize((0.525,), (0.28,))

        ])

        rgb = TF.rotate(rgb, angle)
        # print("1111111111111111111")
        rgb = tr_rgb(rgb)
        depth = TF.rotate(depth, angle)
        target = TF.rotate(target, angle)
        depth = tr_d(depth)
        target = tr_d(target)

        # ####### depth aug ######
        # aug_intnsty = {
        #     "distort_edges": (0, 0.6),
        #     "black_edges": (0, 0.5),
        #     "speckle": (0, 0.03),
        #     "guassian_blur": (0, 0.2)
        # }
        # depth = depth.squeeze().numpy()
        # for name, intensity_range in aug_intnsty.items():
        #     aug = TRANSFORM_DICT[name]
        #     intensity = np.random.rand() * intensity_range[1] * 0.1
        #     if name == "black_edges":
        #         depth = aug(depth, intensity, 1282/306)
        #     else:
        #         depth = aug(depth, intensity)
        # depth = torch.tensor(depth, dtype=torch.float32).unsqueeze(0)
        # ########################

        # rgb = T.ColorJitter()(rgb)

        # ###mix aug
        # rgb[1, int(l):int(l + w), int(u):int(u + h)] = target[0, int(l):int(l + w), int(u):int(u + h)]
        # rgb[0, int(l):int(l + w), int(u):int(u + h)] = target[0, int(l):int(l + w), int(u):int(u + h)]
        # rgb[2, int(l):int(l + w), int(u):int(u + h)] = target[0, int(l):int(l + w), int(u):int(u + h)]
        #
        # ####cutout aug
        # a = torch.rand(1).item()
        # b = torch.rand(1).item()
        # c = torch.rand(1).item()
        # d = torch.rand(1).item()
        # p = 0.5
        #
        # l = 256 * a
        # u = 256 * b
        # w = (256 - a * 256) * c * 0.75
        # h = (256 - b * 256) * d * 0.75
        #
        # rgb[1, int(l):int(l + w), int(u):int(u + h)] = 0
        # rgb[0, int(l):int(l + w), int(u):int(u + h)] = 0
        # rgb[2, int(l):int(l + w), int(u):int(u + h)] = 0
        # rgb[0, int(l):int(l + w), int(u):int(u + h)] = 0



        # rgb = rgb / 10000.0
        # depth = depth / 10000.0
        # target = target / 10000.0

        return rgb, depth, target

    def train_transform3(self, rgb, depth, target):
        s = np.random.uniform(1.0, 1.5)  # random scaling
        s = int(512 * s)
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = float(torch.randint(0, 2, (1,)).item())

        tr = T.Compose([
            # T.Resize(250.0 / iheight),  # this is for computational efficiency, since rotation can be slow
            T.RandomRotation(abs(angle)),
            T.Resize((s, s)),
            T.CenterCrop(self.output_size),
            T.RandomHorizontalFlip(do_flip)
        ])
        rgb = tr(rgb)
        depth = tr(depth)
        target = tr(target)
        rgb = T.ColorJitter()(rgb)
        return rgb, depth, target

    def train_transform2(self, rgb, depth, target):
        s = np.random.uniform(1.0, 1.5)  # random scaling
        depth_np = depth / s
        target_np = target / s
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize(250.0 / iheight),  # this is for computational efficiency, since rotation can be slow
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop(self.output_size),
            transforms.HorizontalFlip(do_flip)
        ])
        # print("-----rgb np")
        # print(rgb)
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np)  # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)
        target_np = transform(target_np)

        return rgb_np, depth_np  # , target_np

    def val_transform(self, rgb, depth, target):
        transform_rgb = T.Compose([
            T.Resize((256, 256), TF.InterpolationMode.NEAREST),
            T.Normalize(mean=(148.47785343, 129.2452198, 105.22572954), std=(51.75028439, 50.10456233, 50.28064866))
            # T.Normalize(mean=(0.4255, 0.3926, 0.3604), std=(0.4144, 0.4110, 0.4086))
        ])

        transform_d = transforms.Compose([
            T.Resize((256, 256), TF.InterpolationMode.NEAREST),
            T.Normalize((1282.0,), (306.0,)),
            # T.Normalize((0.525,), (0.28,))
        ])

        rgb = transform_rgb(rgb)
        target = transform_d(target)

        depth = transform_d(depth)

        return rgb, depth, target

        transform = transforms.Compose([
            T.Resize((256, 256), TF.InterpolationMode.NEAREST)
        ])

        rgb = transform(rgb)
        target = transform(target)

        depth = transform(depth)

        return rgb, depth, target

    def val_transform2(self, rgb, depth, target):
        return rgb, depth, target
        depth_np = depthw
        transform = transforms.Compose([
            T.Resize((256, 256), TF.InterpolationMode.NEAREST)
        ])

        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb, depth, target
