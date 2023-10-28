import torch
from PIL import Image
import os.path as osp
import numpy as np
import random
import torch.nn.functional as F

def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if type(image_list[0]) is tuple:
          return image_list

      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def pil_loader(root, path):
    with open(osp.join(root, path), 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageList(object):
    def __init__(self, image_list, labels=None, root='data', transform=None, target_transform=None, rand_transform=None):
        samples = make_dataset(image_list, labels)
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform
        self.rand_transform = rand_transform
        self.rand_num = 0
        self.loader = pil_loader
        self.root = root

    def __getitem__(self, index):

        path, target = self.samples[index]
        target = int(target)

        sample_ = self.loader(self.root, path)

        if self.transform is not None:
            sample = self.transform(sample_)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.rand_transform is not None:
            rand_sample = []
            for i in range(self.rand_num):
                rand_sample.append(self.rand_transform(sample_))

            return sample, target, index, *rand_sample
        else:
            return sample, target, index

    def __len__(self):
        return len(self.samples)

    def add_item(self, addition):
        # self.samples = np.concatenate((self.samples, addition), axis=0)
        self.samples.extend(addition)
        return self.samples

    def remove_item(self, reduced):
        self.samples = np.delete(self.samples, reduced, axis=0)
        return self.samples


class ImageList_mixup(object):
    def __init__(self, image_list, labels=None, root='data', epoch=0,
                 transform=None, target_transform=None, rand_transform=None):
        samples = make_dataset(image_list, labels)
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform
        self.rand_transform = rand_transform
        self.rand_num= 0
        self.loader, self.root = pil_loader, root

    def __getitem__(self, index):
        path, target = self.samples[index]
        lbl = F.one_hot(torch.tensor([target]), num_classes=5).squeeze(0)
        img_ = self.loader(self.root, path)
        img = self.transform(img_)

        if self.rand_transform is not None:
            rand_sample = []
            for i in range(self.rand_num):
                rand_sample.append(self.rand_transform(img_))

            return img, lbl, index, *rand_sample
        else:
            return img, lbl, index

    def __len__(self):
        return len(self.samples)

    def add_item(self, addition):
        self.samples.extend(addition)
        return self.samples

    def remove_item(self, idx):
        for id in idx:
            self.samples.pop(id)
        return self.samples