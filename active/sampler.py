import os, copy, random, logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from torch.utils.data import DataLoader

import utils.utils as utils
from solver import get_solver
from dataset.image_list import ImageList
from .budget import BudgetAllocator
from .utils import ActualSequentialSampler
from utils.ema import ModelEMA
from utils.tool import Prototype, MemoryBank

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

al_dict = {}

def register_strategy(name):
    def decorator(cls):
        al_dict[name] = cls
        return cls

    return decorator

def get_strategy(sample, *args):
    if sample not in al_dict: raise NotImplementedError
    return al_dict[sample](*args)


class SamplingStrategy:
    """
    Sampling Strategy wrapper class
    """
    def __init__(self, src_dset, tgt_dset, source_model, trained_generator, device, num_classes, cfg):
        self.src_dset = src_dset
        self.tgt_dset = tgt_dset
        self.num_classes = num_classes
        self.model = copy.deepcopy(source_model) # initialized with source model
        self.generator = trained_generator
        self.ema = ModelEMA(device, self.model, 0.99)

        self.device = device
        self.cfg = cfg
        self.idxs_lb = np.zeros(len(self.tgt_dset.train_idx), dtype=bool)
        self.solver = None
        self.lr_scheduler = None

        self.query_dset = tgt_dset.get_dsets()[1] # change to query dataset

        self.src_prototype = Prototype(self.cfg.DATASET.NUM_CLASS, self.cfg.MODEL.BOTTEN_NECK, self.device)
        self.tgt_memory_bank = MemoryBank(len(self.tgt_dset.train_idx), 256, 0.9, self.device)


    def query(self, n, epoch):
        pass

    def update(self, idxs, data, target, epoch):
        self.idxs_lb[idxs] = True
        if epoch==0:
            self.tgt_label = target
            self.tgt_data = data

    def pred(self, idxs=None, with_emb=False):
        if idxs is None:
            idxs = np.arange(len(self.tgt_dset.train_idx))[~self.idxs_lb]

        train_sampler = ActualSequentialSampler(self.tgt_dset.train_idx[idxs])
        data_loader = torch.utils.data.DataLoader(self.query_dset, sampler=train_sampler, num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                                                  batch_size=self.cfg.DATALOADER.BATCH_SIZE, drop_last=False)
        self.model.eval()
        all_log_probs, all_scores, all_embs = [], [], []
        all_targets, all_data = torch.zeros(len(self.tgt_dset.train_idx)).long(), torch.zeros(len(self.tgt_dset.train_idx),3,224,224)
        with torch.no_grad():
            for batch_idx, (data, target, idx, *_) in enumerate(data_loader):
                all_targets[idx] = target
                all_data[idx] = data
                data, target = data.to(self.device), target.to(self.device)
                if with_emb:
                   scores, embs = self.model(data, with_emb=True)
                   all_embs.append(embs.cpu())
                else:
                   scores = self.model(data, with_emb=False)
                log_probs = nn.LogSoftmax(dim=1)(scores)
                all_log_probs.append(log_probs)
                all_scores.append(scores)

        all_log_probs = torch.cat(all_log_probs)
        all_probs = torch.exp(all_log_probs)
        all_scores = torch.cat(all_scores)
        if with_emb:
            all_embs = torch.cat(all_embs)
            return idxs, all_probs, all_log_probs, all_scores, all_embs, all_data, all_targets
        else:
            return idxs, all_probs, all_log_probs, all_scores, all_data, all_targets

    def build_loaders(self):
        src_loader = self.src_dset.get_loaders()[0]
        tgt_loader = self.tgt_dset.get_loaders()[0]

        target_train_dset_lb = self.tgt_dset.get_dsets()[0]
        target_train_dset_ulb = self.tgt_dset.get_dsets()[-1]
        train_sampler = SubsetRandomSampler(self.tgt_dset.train_idx[self.idxs_lb])
        tgt_sup_loader = DataLoader(
            target_train_dset_lb,
            sampler=train_sampler,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            batch_size=self.cfg.DATALOADER.BATCH_SIZE,
            drop_last=False
        )
        train_sampler = SubsetRandomSampler(self.tgt_dset.train_idx[~self.idxs_lb])
        tgt_unsup_loader = DataLoader(
            target_train_dset_ulb,
            sampler=train_sampler,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            batch_size=self.cfg.DATALOADER.BATCH_SIZE,
            drop_last=False
        )

        # create joint src_tgt_sup loader as commonly used
        joint_list = [self.src_dset.train_dataset.samples[_] for _ in self.src_dset.train_idx] + \
                        [self.tgt_dset.train_dataset.samples[_] for _ in self.tgt_dset.train_idx[self.idxs_lb]]

        # use source train transform
        join_transform = self.src_dset.get_dsets()[0].transform
        joint_train_ds = ImageList(joint_list, root=self.cfg.DATASET.ROOT, transform=join_transform)
        joint_sup_loader = DataLoader(
            joint_train_ds,
            batch_size=self.cfg.DATALOADER.BATCH_SIZE,
            shuffle=True,
            drop_last=False,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS
        )

        return src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader

    def train(self, epoch):
        src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader = self.build_loaders()

        solver = get_solver(
            self.cfg.ADA.DA, self.model, self.ema, self.generator,
            self.src_prototype, self.tgt_memory_bank, src_loader,
            tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader,
            self.opt_net_tgt, True, self.device, self.cfg)

        train_sampler = ActualSequentialSampler(self.tgt_dset.train_idx)
        seq_query_loader = torch.utils.data.DataLoader(
            self.query_dset,
            sampler=train_sampler,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            batch_size=self.cfg.DATALOADER.BATCH_SIZE,
            drop_last=False
        )
        solver.solve(epoch, seq_query_loader, self.tgt_data, self.tgt_label, self.tgt_dset)

        return self.ema.ema

    def init_idx(self, tgt_dset):
        self.idxs_lb = np.zeros(len(tgt_dset.train_idx), dtype=bool)
        budget = np.round(len(tgt_dset.train_idx) * self.cfg.ADA.BUDGET) if self.cfg.ADA.BUDGET <= 1.0 else np.round(
            self.cfg.ADA.BUDGET)
        self.budget_allocator = BudgetAllocator(budget=budget, cfg=self.cfg)

    def acquire_budget(self, epoch):
        curr_budget, used_budget = self.budget_allocator.get_budget(epoch)
        return curr_budget, used_budget

    def test(self, test_loader):
        self.ema.ema.eval()
        correct, all_emb = 0, []
        with torch.no_grad():
            for data, target, idx in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, emb = self.ema.ema(data, True)
                all_emb.append(emb.detach().float().cpu())

                output = output.softmax(dim=1)
                pred = output.argmax(dim=1, keepdim=True)

                corr = pred.eq(target.view_as(pred)).sum().item()
                correct += corr

        self.all_emb = F.normalize(torch.cat(all_emb), dim=1)
        test_acc = 100. * correct / len(test_loader.sampler)
        return test_acc

    def save_checkpoint(self, epoch, save_dir):
        checkpoint = {
            "net": self.ema.ema.state_dict(),
            'optimizer': self.opt_net_tgt.state_dict(),
            "epoch": epoch,
            "selected_idxs": self.idxs_lb,
            "tgt_data":self.tgt_data,
            "tgt_label":self.tgt_label,
            "tgt_memory_bank":self.tgt_memory_bank,
        }
        torch.save(checkpoint, save_dir)

    def resume_checkpoint(self, load_dir, tgt_dset):
        if load_dir:
            checkpoint = torch.load(load_dir)
            self.model.load_state_dict(checkpoint['net'])
            self.ema = ModelEMA(self.device, self.model, 0.99)
            self.opt_net_tgt.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            self.idxs_lb = checkpoint['selected_idxs']
            self.tgt_label = checkpoint["tgt_label"]
            self.tgt_data = checkpoint["tgt_data"]
            self.tgt_memory_bank = checkpoint["tgt_memory_bank"]
        else:
            start_epoch = 0
        return start_epoch