import torch
import torch.nn as nn
import torch.nn.functional as F

import ot
import numpy as np


class Prototype:
    def __init__(self, C, dim, device, m=0.9):
        self.mo_pro = torch.zeros(C, dim).to(device)
        self.m = m

    @torch.no_grad()
    def update(self, feats, lbls, i_iter, norm=False):
        if i_iter == 0:
            momentum = 0
        else:
            momentum = self.m

        feats = F.normalize(feats)
        for i_cls in torch.unique(lbls):
            feats_i = feats[lbls == i_cls, :]
            feats_i_center = feats_i.mean(dim=0, keepdim=True)
            self.mo_pro[i_cls, :] = self.mo_pro[i_cls, :] * momentum + feats_i_center * (1 - momentum)
        if norm:
            self.mo_pro = F.normalize(self.mo_pro)

class MemoryBank():
    def __init__(self, len, dim, m, device):
        self.bank = torch.zeros(len, dim)
        self.m = m

    def init(self, bank):
        self.bank = bank

    def update(self, i_iter, feats, idx, norm):
        if i_iter == 0:
            momentum = 0
        else:
            momentum = self.m

        feats = F.normalize(feats)
        self.bank[idx.cpu(), :] = self.bank[idx.cpu(), :] * momentum + feats * (1 - momentum)

        if norm:
            self.bank = F.normalize(self.bank, dim=1)

def reg_intra(p1, p2):
    N, C = p1.shape
    cov = p1.t() @ p2

    loss = (torch.sum(cov) - torch.trace(cov)) / C
    return loss

def reg_inter(proto_s, feat_tu_w, score_s, device):
    with torch.no_grad():
        a = F.normalize(proto_s.mo_pro, dim=1)
        b = F.normalize(feat_tu_w, dim=1)
        mat = a @ b.t()
        M_st_weak = 1 - mat  # postive distance

    M = M_st_weak.data.cpu().numpy().astype(np.float64)
    ns, nt = M.shape
    a, b = np.ones((ns,)) / ns, np.ones((nt,)) / nt
    gamma = ot.unbalanced.sinkhorn_stabilized_unbalanced(a, b, M, 1, 1)
    gamma_st_weak = torch.from_numpy(gamma).to(device)

    pred_ot = gamma_st_weak.t()
    Lm = torch.nn.L1Loss()(pred_ot, score_s)
    return Lm


def acquire_data(loader, dataset, device, mode, score=None, high=None, low=None, gmm_output=None):
    try:
        if mode=='common':
            data, label, idx = next(loader)
        elif mode=='enhance':
            data, label, idx = next(loader)
            data_w, data_s = data[0], data[1]
        elif mode=='rand':
            data, label, idx, *data_r = next(loader)
        else:
            raise NotImplementedError
    except:
        if mode == 'common':
            loader = iter(dataset)
            data, label, idx = next(loader)
        elif mode == 'enhance':
            loader = iter(dataset)
            data, label, idx = next(loader)
            data_w, data_s = data[0], data[1]
        elif mode == 'rand':
            loader = iter(dataset)
            data, label, idx, *data_r = next(loader)
        else:
            raise NotImplementedError

    if mode=='common':
        return loader, data.to(device), label.to(device), idx.to(device)
    elif mode=='enhance':
        return loader, data_w.to(device), data_s.to(device), label.to(device), idx.to(device)
    elif mode == 'rand':
        return loader, data.to(device), data_r[0].to(device), label.to(device), idx.to(device)
    else:
        raise NotImplementedError


def rand_dataAug(data, data_rand, device):
    mask = torch.FloatTensor(np.random.beta(0.2, 0.2, size=(data.shape[0], 1, 1, 1))).to(device)
    data_aug = (data * mask) + (data_rand * (1 - mask))
    return data_aug



