from glob import glob
import os
from abc import abstractmethod, ABC

import cv2
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
from tqdm import tqdm


# DISTORTION_MAPS = "/disk2/shahaf/Apple/data_processing/distortion_maps/256x256_7x7"
DISTORTION_MAPS = "/ssd/distortion_maps/256x256_7x7"


class BaseNoiseAugmentation(ABC):

    def __init__(self, intensity: int, out_folder: str):
        self.out_folder = out_folder
        self.intensity = intensity
        if not os.path.exists(out_folder):
            os.makedirs(out_folder, exist_ok=True)

    def get_out_folder(self):
        return self.out_folder

    @abstractmethod
    def __call__(self, im: np.ndarray) -> np.ndarray:
        pass


def noisy(noise_typ, image, **kwargs):
    """
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance."""

    std = kwargs['std'] if 'std' in kwargs else 0.2
    mean = kwargs['mean'] if 'mean' in kwargs else 0.0

    #     print("applying", noise_typ, "with mean", mean, "and std", std)
    if noise_typ == "gauss":
        gauss = np.random.normal(mean, std, image.shape)
        noisy = image + gauss
        return noisy

    elif noise_typ == "s&p":
        s_vs_p = 0.5
        amount = 0.05
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)).tolist()
                  for i in image.shape]
        coords = tuple(coords)
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)).tolist()
                  for i in image.shape]
        coords = tuple(coords)
        out[coords] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 1.5 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    elif noise_typ == "speckle":
        gauss = np.random.normal(mean, std, image.shape)
        noisy = (image + (2 ** image * gauss))
        return noisy


def speckle(im, intensity=0.5):
    max_std = 0.4
    std = intensity * max_std
    return noisy('speckle', im, std=std)


# def blur_around_edges(im, canny_params=(10, 10), dilation_params=(8, 8), blur_size=21):
#     edges = cv2.Laplacian(im, cv2.CV_64F, ksize=5)
#     edges = float2int(cv2.convertScaleAbs(edges) > 100)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilation_params)
#     thick_edges = cv2.dilate(edges, kernel)
#     edge_noise = noisy("speckle", thick_edges / 255, mean=0, std=0.1)
#     blurred_edges = median_filter(edge_noise, blur_size)
#     im_edge_blur = im * (1 - blurred_edges)
#     return im_edge_blur

def guassian_blur(im, intensity):
    sig = 2 ** (intensity * 3) - 1  # exponential in range [0-7]
    imb = gaussian_filter(im, sigma=sig, truncate=int(np.ceil(sig * 3)))
    return imb


def get_neighborhood(im, x, y, dx=1, dy=1):
    x_left = max(x - dx, 0)
    x_right = min(x + dx + 1, im.shape[1])
    y_bottom = max(y - dy, 0)
    y_top = min(y + dy + 1, im.shape[0])
    return im[y_bottom:y_top, x_left:x_right]


def morph_by_neighborhood(im, operator, dx=5, dy=5, **kwargs):
    new_im = np.zeros(im.shape)
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            neighborhood = get_neighborhood(im, x, y, dx, dy)
            new_im[y, x] = operator(neighborhood, **kwargs)
    return new_im


def sample_neighbor(nbrhod):
    nbr_idx = np.random.randint([0, 0], nbrhod.shape[:2])
    return nbrhod[nbr_idx[0], nbr_idx[1]]


def create_distrotion_maps(shape, output_folder, k, count=1000):
    pixel_map = np.indices((shape[0], shape[1])).transpose()
    for i in tqdm(range(count)):
        distortion_map = morph_by_neighborhood(pixel_map, sample_neighbor, k, k).astype('uint16')
        np.save(os.path.join(output_folder, f"{i}"), distortion_map)


def distort_edges(im, intensity, distort_map=None):
    if distort_map is not None:
        distort_map = distort_map.astype(int)
        distorted = im[distort_map[:, :, 1], distort_map[:, :, 0]]
    else:
        distorted = morph_by_neighborhood(im, sample_neighbor, int(7 * intensity), int(7 * intensity))
    avg_distort = median_filter(distorted, 5)
    return avg_distort


def distort_edges_from_distortion_map(im, intensity, distort_map_dir=DISTORTION_MAPS):
    if int(7 * intensity) == 0:
        return im
    distort_map_dir = distort_map_dir.replace("7x7", f"{int(7 * intensity)}x{int(7 * intensity)}")
    distort_map_paths = glob(os.path.join(distort_map_dir, "*"))
    dist_map = np.load(np.random.choice(distort_map_paths))
    return distort_edges(im, intensity, dist_map)


def bin_image(im, intensity):
    bins = 2 ** (16 - int(intensity * 12 + 4))
    minv = im.min()
    maxv = im.max()
    res = (np.digitize(im, np.linspace(minv, maxv, bins), right=False) / bins) * (maxv - minv) + minv
    return res


def black_edges(im, intensity, zero_val):
    edges = cv2.Sobel(im, cv2.CV_64F, 1, 1, ksize=3, scale=8)
    edges = (edges > 0.2).astype(float)
    k = int(intensity * 10)
    if k == 0:
        return im
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    thick_edges = median_filter(cv2.dilate(edges, kernel), 7)
    res = im * (1 - thick_edges) + zero_val * thick_edges
    return res


def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def neighborhood_dist_aware_smoothing(nbrs, sig=1):
    nbr_h, nbr_w = nbrs.shape
    center = nbrs[nbr_h // 2, nbr_w // 2]
    sig_ = 2 ** center * sig + 1e-6
    kern = gkern(max(nbr_h, nbr_w), sig_)
    kern = kern[:nbr_h, :nbr_w]
    return (kern * nbrs).sum()


def dist_aware_smoothing(im, sig=1):
    im_ = cv2.copyMakeBorder(im, 6, 6, 6, 6, cv2.BORDER_REFLECT)
    im_ = morph_by_neighborhood(im_, neighborhood_dist_aware_smoothing, 6, 6, sig=sig)
    return im_[6:-6,6:-6]


def compose_noises(im,
                   noises=["distort_edges", "black_edges", "speckle", "dist_aware_smoothing"],
                   intensities=[0.3, 0.25, 0.015, 0.1],
                   distortion_map_folder=DISTORTION_MAPS,
                   # zero_val=-1282 / 306):
                   zero_val=-0.525 / 0.28):
    noise_dict = {"speckle": speckle,
                  "guassian_blur": guassian_blur,
                  "distort_edges": lambda im, intsty: distort_edges_from_distortion_map(im, intsty,
                                                                                        distortion_map_folder),
                  "dist_aware_smoothing": dist_aware_smoothing,
                  "bin_image": bin_image,
                  "black_edges": lambda im, intsty: black_edges(im, intsty, zero_val),
                  }
    for n, intsty in zip(noises, intensities):
        im = noise_dict[n](im, intsty)
    return im


TRANSFORM_DICT = {"speckle": speckle,
                  "guassian_blur": guassian_blur,
                  "dist_aware_smoothing": dist_aware_smoothing,
                  "distort_edges": lambda im, intsty: distort_edges_from_distortion_map(im, intsty, DISTORTION_MAPS),
                  "bin_image": bin_image,
                  "black_edges": black_edges,
                  "compose_noises": compose_noises
                  }

if __name__ == "__main__":
    for i in tqdm(range(1, 7)):
        dist_map_dir = DISTORTION_MAPS.replace("7", str(i))
        os.makedirs(dist_map_dir, exist_ok=True)
        create_distrotion_maps((256, 256), dist_map_dir, i)
