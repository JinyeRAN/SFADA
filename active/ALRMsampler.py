import torch
import numpy as np
import torch.nn.functional as F

from .utils import ActualSequentialSampler
from .sampler import register_strategy, SamplingStrategy

import warnings
warnings.filterwarnings('ignore')

def default_gamma(X:torch.Tensor):
    gamma = 1.0 / X.shape[1]
    return gamma


def rbf_kernel(X:torch.Tensor, gamma:float=None):
    assert len(X.shape) == 2
    if gamma is None:
        gamma = default_gamma(X)
    K = torch.cdist(X, X)
    K.fill_diagonal_(0).pow_(2).mul_(-gamma).exp_()
    return K


def select_sample(K :torch.Tensor, num :int, plbl_idxs :torch.Tensor):
    sample_indices = torch.arange(0, K.shape[0])
    num_samples = sample_indices.shape[0]

    colsum = 2 * K.sum(0) / num_samples
    is_selected, is_plblselected = torch.zeros_like(sample_indices), torch.zeros_like(sample_indices)
    selected, plblselected = sample_indices[is_selected > 0], sample_indices[is_plblselected > 0]
    is_anyselected = is_selected + is_plblselected
    anyselected = sample_indices[is_anyselected > 0]

    idx = 0
    while 0 < num:
        candidate_indices = sample_indices[is_anyselected == 0]
        s1 = colsum[candidate_indices]

        if anyselected.shape[0] == 0:
            s1 -= K.diagonal()[candidate_indices].abs()
        else:
            temp = K[anyselected, :][:, candidate_indices]
            s2 = temp.sum(0) * 2 + K.diagonal()[candidate_indices]
            s2 /= (anyselected.shape[0] + 1)
            s1 -= s2

        best_sample_index = candidate_indices[s1.argmax()]

        if best_sample_index in plbl_idxs:
            is_plblselected[best_sample_index] = idx + 1
        else:
            is_selected[best_sample_index] = idx + 1
            selected = sample_indices[is_selected > 0]
            num -= 1

        idx += 1
        is_anyselected = is_selected + is_plblselected # combined indicater for selection
        anyselected = sample_indices[is_anyselected > 0]

    selected_in_order = selected[is_selected[is_selected > 0].argsort()]
    return selected_in_order


@register_strategy('ALRM')
class ALRMSampling(SamplingStrategy):
    def __init__(self, src_dset, tgt_dset, model, trained_generator, device, num_classes, cfg):
        super(ALRMSampling, self).__init__(src_dset, tgt_dset, model, trained_generator, device, num_classes, cfg)

    def query(self, select_num, epoch):
        idxs_all = np.arange(len(self.tgt_dset.train_idx))
        train_sampler = ActualSequentialSampler(self.tgt_dset.train_idx[idxs_all])
        data_loader = torch.utils.data.DataLoader(
            self.query_dset, sampler=train_sampler, num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            batch_size=self.cfg.DATALOADER.BATCH_SIZE, drop_last=False
        )

        self.model.eval()
        all_probs, all_embs, all_targets, all_data = [], [], torch.zeros(len(self.tgt_dset.train_idx)).long(), \
            torch.zeros(len(self.tgt_dset.train_idx),3,224,224)
        with torch.no_grad():
            for batch_idx, (data, target, idx, *_) in enumerate(data_loader):
                all_data[idx] = data
                scores, embs = self.model(data.to(self.device), with_emb=True)


                all_embs.append(embs.cpu())
                probs = F.softmax(scores, dim=-1)
                all_probs.append(probs.cpu())
                all_targets[idx]=target

        probs_info, feats_info = torch.cat(all_probs), F.normalize(torch.cat(all_embs), dim=-1)
        class_proto1 = F.normalize(list(self.model.classifier.parameters())[0].detach(), dim=-1).cpu()
        class_proto2 = F.normalize(list(self.model.classifier.parameters())[4].detach(), dim=-1).cpu()
        segment_info = torch.einsum('ij,jk,km->im', probs_info, class_proto2, class_proto1)
        embs_rich = feats_info * segment_info + feats_info

        similarity_v = torch.mm(embs_rich, embs_rich.t())
        c1 = torch.sqrt(torch.sum(embs_rich * embs_rich, dim=1, keepdim=True))
        c2 = torch.sqrt(torch.sum(embs_rich * embs_rich, dim=1, keepdim=True))
        similarity_d = torch.mm(c1, c2.t())
        sim = similarity_v / similarity_d


        sim_topk, topk = torch.topk(sim, k=self.cfg.LPDA.S_K + 1, dim=1)
        embs_local_descri = embs_rich[topk].sum(-2) / (self.cfg.LPDA.S_K + 1)  #.view(embs_rich.shape[0], -1)

        feat_rbfk = rbf_kernel(embs_local_descri)
        selected_idx = select_sample(K=feat_rbfk, num=select_num, plbl_idxs=self.idxs_lb)

        return selected_idx.cpu().numpy(), all_data, all_targets