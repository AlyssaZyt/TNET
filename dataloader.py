import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class HumanDataset(Dataset):
    """
    human dataset
    """

    def __init__(self, dataname, transforms=None):

        items = []
        img_path = './data/' + dataname + '_img.json'
        trimap_path = './data/' + dataname + '_trimap.json'
        alpha_path = './data/' + dataname + '_alpha.json'

        with open(img_path, 'r') as j:
            imglist = json.load(j)
        with open(trimap_path, 'r') as j:
            trimaplist = json.load(j)
        with open(alpha_path, 'r') as j:
            alphalist = json.load(j)

        for i in range(len(imglist)):
            items.append((imglist[i], trimaplist[i], alphalist[i]))

        self.items = items
        self.transforms = transforms

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_name, trimap_name, alpha_name = self.items[index]
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        trimap = cv2.imread(trimap_name, cv2.IMREAD_GRAYSCALE)
        alpha = cv2.imread(alpha_name, cv2.IMREAD_GRAYSCALE)

        if self.transforms is not None:
            for transform in self.transforms:
                image, trimap, alpha = transform(image, trimap, alpha)

        return image, trimap, alpha


class RandomPatch(object):
    """
    random patch
    """

    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, image, trimap, alpha):
        image = cv2.resize(image, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
        trimap = cv2.resize(trimap, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
        alpha = cv2.resize(alpha, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)

        return image, trimap, alpha


class Normalize(object):
    """
    normalize
    """

    def __call__(self, image, trimap, alpha):
        image = (image.astype(np.float32) - (114., 121., 134.,)) / 255.0
        trimap[trimap == 0] = 0
        trimap[trimap == 128] = 1
        trimap[trimap == 255] = 2
        alpha = alpha.astype(np.float32) / 255.0
        return image, trimap, alpha


class NumpyToTensor(object):
    """
    numpy to tensor
    """

    def __call__(self, image, trimap, alpha):
        h, w, c = image.shape
        image = torch.from_numpy(image.transpose((2, 0, 1))).view(c, h, w).float()
        trimap = torch.from_numpy(trimap).view(-1, h, w).long()
        alpha = torch.from_numpy(alpha).view(1, h, w).float()
        return image, trimap, alpha


class TrimapToCategorical(object):
    """
    trimap to categorical
    """

    def __call__(self, image, trimap, alpha):
        trimap = np.array(trimap, dtype=np.int)
        input_shape = trimap.shape
        trimap = trimap.ravel()
        n = trimap.shape[0]
        categorical = np.zeros((3, n), dtype=np.long)
        categorical[trimap, np.arange(n)] = 1
        output_shape = (3,) + input_shape
        categorical = np.reshape(categorical, output_shape)
        return image, categorical, alpha