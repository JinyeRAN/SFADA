import numpy as np
import torch
from torch.utils.data.sampler import Sampler
import copy

def ActualSequentialLoader(subsetRandomLoader, indices=None, transform=None, batch_size=None):
    indices = indices if indices is not None else subsetRandomLoader.sampler.indices
    train_sampler = ActualSequentialSampler(indices)
    dataset = copy.deepcopy(subsetRandomLoader.dataset)
    if transform is not None:
        dataset.transform = transform

    batch_size = batch_size if batch_size is not None else subsetRandomLoader.batch_size
    actualSequentialLoader = torch.utils.data.DataLoader(dataset, sampler=train_sampler,
                                                         num_workers=subsetRandomLoader.num_workers,
                                                         batch_size=batch_size, drop_last=False)
    return actualSequentialLoader

class ActualSequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source)

    def __len__(self):
        return len(self.data_source)
