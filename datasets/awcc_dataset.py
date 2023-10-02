from PIL import Image
import torch.utils.data as data
import os
from glob import glob
from torchvision import transforms
import numpy as np

MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet


class Crowd(data.Dataset):
    def __init__(self, root_path, crop_size, downsample_ratio, method='train'):
        if method not in ['train', 'val']:
            raise NotImplementedError
        self.method = method

        self.root_path = root_path
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))

        self.c_size = crop_size
        assert self.c_size % downsample_ratio == 0

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*MEAN_STD)
        ])
        

    def __len__(self):
        return len(self.im_list)
        
    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('jpg', 'npy')
        img = Image.open(img_path).convert('RGB')


        if self.method == 'train':
            keypoints = np.load(gd_path)
            # training data
            images, points, targets, st_sizes = self.train_transform(img, keypoints)
            return images, points, targets, st_sizes

        if self.method == 'val':
            keypoints = np.load(gd_path)
            img = self.trans(img)
            return img, len(keypoints)


   