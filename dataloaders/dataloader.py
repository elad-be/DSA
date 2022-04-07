import os
import os.path
import numpy as np
import torch.utils.data as data
import h5py
import dataloaders.transforms as transforms
from PIL import Image
import imageio
import cv2
import torch
from torchvision import transforms as T
from dataloaders.depth_map_utils import fill_in_multiscale
from .noises import compose_noises

IMG_EXTENSIONS = ['.h5', '.jpg']  # elad edit add .png
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def no_depth_completion(depth_image):
    h, w = depth_image.shape
    depth_image = cv2.copyMakeBorder(depth_image, h, h, w, w, cv2.BORDER_REFLECT)
    projected_depths = np.float32(depth_image / 256.0)
    final_depths, process_dict = fill_in_multiscale(projected_depths)
    final_depths = final_depths[h:-h, w:-w]
    return final_depths


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C


def inpainting_fillna(im, value=None):
    h, w = im.shape
    im_ = cv2.copyMakeBorder(im, h, h, w, w, cv2.BORDER_REFLECT)
    if value is None:
        mask = np.isnan(im_).astype('uint8')
    else:
        mask = (im_ == value).astype('uint8')
    dst = cv2.inpaint(im_.astype('float32'), mask, 3, cv2.INPAINT_NS)
    res = dst[h:-h, w:-w]
    return res


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        os.path.join("new_data/ARKit/train/", "rgb", i)
        print(d)
        if not os.path.isdir(d):
            print("HHHH")
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images


def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    return rgb, depth


# def rgb2grayscale(rgb):
#     return rgb[:,:,0] * 0.2989 + rgb[:,:,1] * 0.587 + rgb[:,:,2] * 0.114

to_tensor = transforms.ToTensor()


class MyDataloader(data.Dataset):
    modality_names = ['rgb', 'rgbd', 'd']  # , 'g', 'gd'
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    def __init__(self, root, type, sparsifier=None, modality='rgb', loader=h5_loader):
        classes, class_to_idx = find_classes(root)
        self.type = type
        # imgs = make_dataset(root, class_to_idx)
        # start elad edit
        if type == 'train':
            # imgs = [os.path.join("..","..","..","..","ssd","hypersim_sample","hypersim_sample","rgb",i) for i in sorted(os.listdir("../../../../ssd/hypersim_sample/hypersim_sample/rgb"))[:37000]] #elad edit
            imgs = [os.path.join("/ssd/ARKit/train/rgb", i) for i in
                    sorted(os.listdir("/ssd/ARKit/train/rgb"))[
                    :1000]]  # elad edit
        if type == 'val':
            # imgs = [os.path.join("..","..","..","..","ssd","hypersim_sample","hypersim_sample","rgb",i) for i in sorted(os.listdir("../../../../ssd/hypersim_sample/hypersim_sample/rgb"))[37000:]] #elad edit
            imgs = [os.path.join("/ssd/ARKit/valid/rgb", i) for i in
                    sorted(os.listdir("/ssd/ARKit/valid/rgb"))]  # elad edit
        assert len(imgs) > 0, "Found 0 images in subfolders of: " + root + "\n"
        print("Found {} images in {} folder.".format(len(imgs), type))
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        if type == 'train':
            self.transform = self.train_transform
        elif type == 'val':
            self.transform = self.val_transform
        else:
            raise (RuntimeError("Invalid dataset type: " + type + "\n"
                                                                  "Supported dataset types are: train, val"))
        self.loader = loader
        self.sparsifier = sparsifier

        assert (modality in self.modality_names), "Invalid modality type: " + modality + "\n" + \
                                                  "Supported dataset types are: " + ''.join(self.modality_names)
        self.modality = modality

    def train_transform(self, rgb, depth):
        raise (RuntimeError("train_transform() is not implemented. "))

    def val_transform(rgb, depth):
        raise (RuntimeError("val_transform() is not implemented."))

    def create_sparse_depth(self, rgb, depth):
        if self.sparsifier is None:
            return depth
        else:
            mask_keep = self.sparsifier.dense_to_sparse(rgb, depth)
            sparse_depth = np.zeros(depth.shape)
            sparse_depth[mask_keep] = depth[mask_keep]
            return sparse_depth

    def create_rgbd(self, rgb, depth):
        sparse_depth = self.create_sparse_depth(rgb, depth)
        rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
        return rgbd

    def __getraw2__(self, index):  ##this is the good and regular version
        """
        Args:
            index (int): Index

        Returns:
            tuple: (rgb, depth) the raw data.
        """
        # path, target = self.imgs[index]
        # rgb, depth = self.loader(path)
        # return rgb, depth

        ####elad edit start###

        # print(self.imgs[index])

        im_frame = Image.open(self.imgs[index])
        rgb = T.ToTensor()(np.float32(im_frame))

        # new_data/ARKit/train/rgb

        depth = os.path.join("../../../../ssd/ARKit/train", "lidar", self.imgs[index][32:-4] + ".png")
        depth = Image.open(depth)

        # depth_indicator = np.where(depth == 0, 1, 0)
        # depth_filled = no_depth_completion(depth)
        # depth = depth + 256 * depth_indicator * depth_filled

        depth = T.ToTensor()(np.float32(depth))

        # depth.float()

        target = os.path.join("../../../../ssd/ARKit/train", "ipad", self.imgs[index][32:-4] + ".png")

        target = Image.open(target)
        target = T.ToTensor()(np.float32(target))

        return rgb, depth, target
        ####elad edit end

    def __getraw__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (rgb, depth) the raw data.
        """

        #########apple synth data
        # im_frame = Image.open(self.imgs[index])
        # rgb = T.ToTensor()(im_frame)
        # depth = os.path.join("../../../../ssd/hypersim_sample/hypersim_sample/depth_preped",
        #                      self.imgs[index][52:-4] + ".exr")
        # depth = cv2.imread(depth, -1)
        # depth = T.ToTensor()(np.float32(depth))
        # target = os.path.join("../../../../ssd/hypersim_sample/hypersim_sample/compose_noises2",
        #                       self.imgs[index][52:-4] + ".exr")
        # target = cv2.imread(target, -1)
        # target = T.ToTensor()(np.float32(target))
        ##########

        im_frame = Image.open(self.imgs[index])
        rgb = T.ToTensor()(np.float32(im_frame))
        if self.type == "train":
            depth = os.path.join("/ssd/ARKit/train", "lidar",
                                 self.imgs[index].split("/")[-1][:-4] + ".png")
            target = os.path.join("/ssd/ARKit/train", "ipad",
                                  self.imgs[index].split("/")[-1][:-4] + ".png")
        else:
            depth = os.path.join("/ssd/ARKit/valid", "lidar",
                                 self.imgs[index].split("/")[-1][:-4] + ".png")
            target = os.path.join("/ssd/ARKit/valid", "ipad",
                                  self.imgs[index].split("/")[-1][:-4] + ".png")

        depth = Image.open(depth)
        depth = T.ToTensor()(np.float32(depth))
        target = Image.open(target)
        target = T.ToTensor()(np.float32(target))

        return rgb, depth, target
        ####elad edit end

    def __transform_data__(self, rgb, depth, target):
        s = np.random.uniform(1.0, 1.5)  # random scaling
        s = int(512 * s)
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        # do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        tr = T.Compose([
            # T.Resize(250.0 / iheight),  # this is for computational efficiency, since rotation can be slow
            T.RandomRotation(abs(angle)),
            T.Resize((s, s)),
            T.CenterCrop(self.output_size),
            T.RandomHorizontalFlip()
        ])
        rgb = tr(rgb)
        depth = tr(depth)
        target = tr(target)
        rgb = T.ColorJitter()(rgb)

        return rgb, depth, target

    def __getitem__(self, index):  # elad edit

        rgb, depth, target = self.__getraw__(index)
        rgb, depth, target = self.transform(rgb, depth, target)

        # print("------------")
        # rgb = inpainting_fillna(rgb)

        # rgb = T.Resize((768,768))(rgb)
        # rgb = T.CenterCrop((512,512))(rgb)
        # depth = T.Resize((768, 768))(depth)
        # depth = T.CenterCrop((512, 512))(depth)
        # target = T.Resize((768, 768))(target)
        # target = T.CenterCrop((512, 512))(target)
        # target = inpainting_fillna(target)
        return torch.cat((rgb, depth), 0), target

    def __len__(self):
        return len(self.imgs)
