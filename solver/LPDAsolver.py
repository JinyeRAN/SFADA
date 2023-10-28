import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import copy, time, logging, random
import numpy as np
from collections import Counter

from dataset.transform import return_rndn_transform
from dataset.image_list import ImageList, ImageList_mixup
from .solver import BaseSolver, register_solver
from utils.tool import acquire_data, reg_inter, reg_intra, Prototype
from utils.loss_functions import InterAlignLoss


def mixup(tgt_data_selected, tgt_label_selected, step_data, step_label, selected_num, device, selected:bool):
    step_idx= list(torch.arange(step_label.shape[0]).numpy())
    step_label, step_data = copy.deepcopy(step_label.float()), copy.deepcopy(step_data)
    tgt_label_selected = tgt_label_selected.to(device)
    if selected:
        selected_id_ori = random.sample(step_idx, selected_num)
        for selected_sample_id_ori in selected_id_ori:
            selected_sample_id_mix = torch.nonzero(tgt_label_selected!=step_label.argmax(dim=1)[selected_sample_id_ori])
            selected_sample_id_mix = random.choice(selected_sample_id_mix.squeeze(dim=1))
            select_ori_lbl = step_label[selected_sample_id_ori:selected_sample_id_ori+1, ]
            select_mix_lbl = F.one_hot(torch.tensor([tgt_label_selected[selected_sample_id_mix]]), num_classes=5).to(device)
            select_ori_data = step_data[selected_sample_id_ori:selected_sample_id_ori+1,]
            select_mix_data = tgt_data_selected[selected_sample_id_mix:selected_sample_id_mix+1,].to(device)

            alpha = 1.0
            lam = np.random.beta(alpha, alpha)
            step_data[selected_sample_id_ori:selected_sample_id_ori+1,] =  select_ori_data * lam + select_mix_data * (1 - lam)
            step_label[selected_sample_id_ori:selected_sample_id_ori+1,] = select_ori_lbl * lam + select_mix_lbl * (1 - lam)
    return step_data, step_label


@register_solver('LPDA')
class LAASolver(BaseSolver):
    def __init__(self, net, ema, trained_generator, src_proto, tgt_proto,
                 src_loader, tgt_loader, tgt_sup_loader,
                 tgt_unsup_loader, joint_sup_loader, tgt_opt, ada_stage, device, cfg, **kwargs):
        super(LAASolver, self).__init__(
            net, ema, trained_generator, src_proto, tgt_proto,
            src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader,
            joint_sup_loader, tgt_opt, ada_stage, device, cfg, **kwargs
        )
        self.CE_Loss = nn.CrossEntropyLoss()
        self.InterAlignLoss = InterAlignLoss(self.cfg.DATASET.NUM_CLASS, temp=0.5, device=self.device)

    def solve(self, epoch, seq_query_loader, tgt_data, tgt_label_all, tgt_dset):
        resume = 'Resuming |' if self.cfg.TRAINER.RESUME is not None else None
        tgtl_sup_dataset = self.tgt_sup_loader.dataset
        tgtl_sup_samples = [tgtl_sup_dataset.samples[i] for i in self.tgt_sup_loader.sampler.indices]
        tgtl_dataset = ImageList_mixup(tgtl_sup_samples, epoch=epoch, root=tgtl_sup_dataset.root, transform=tgtl_sup_dataset.transform)
        tgtl_dataset.rand_transform, tgtl_dataset.rand_num = return_rndn_transform(self.cfg), self.cfg.LPDA.A_RAND_NUM
        tgtl_idxs = self.tgt_sup_loader.sampler.indices.tolist()
        tgtl_idxs_fixed = torch.tensor(np.array(copy.deepcopy(tgtl_idxs)))
        tgtl_labels = [tgtl_dataset.samples[i][1] for i in range(len(tgtl_dataset))]
        counts = Counter(tgtl_labels)
        logging.info('{} Epoch {}: active sample: {}, active label: '.format(resume, epoch, len(tgtl_labels)) +str(counts))

        tgtl_loader = torch.utils.data.DataLoader(tgtl_dataset, shuffle=True, batch_size=self.tgt_sup_loader.batch_size, num_workers=self.tgt_sup_loader.num_workers)
        tgtl_iter = iter(tgtl_loader)
        tgt_iter = iter(self.tgt_unsup_loader)

        epoch_idxs_bool = torch.zeros(len(tgt_dset.train_idx)).bool()
        epoch_idxs_bool[tgtl_idxs] = True
        tgt_idxs_unlabelled, tgt_idxs_labelled = torch.nonzero(~epoch_idxs_bool).squeeze(1), torch.nonzero(epoch_idxs_bool).squeeze(1)
        tgt_data_unselected, tgt_data_selected = tgt_data[~epoch_idxs_bool], tgt_data[epoch_idxs_bool]
        tgt_label_unselected, tgt_label_selected = tgt_label_all[~epoch_idxs_bool], tgt_label_all[epoch_idxs_bool]

        self.net.train()
        gt_all, delete_tmp = [], []
        gt_all.extend(tgtl_labels)
        for idx in range(len(self.tgt_unsup_loader)):
            print(idx)
            batch_idx = epoch * len(self.tgt_unsup_loader) + idx
            with torch.no_grad():
                z = torch.rand(self.cfg.TRAINER.SF_GENERATOR_BZ, self.cfg.TRAINER.SF_GENERATOR_DIM).to(self.device)
                src_label = torch.randint(0, self.cfg.DATASET.NUM_CLASS, (self.cfg.TRAINER.SF_GENERATOR_BZ,)).to(self.device)
                src_emb = self.generator(z, src_label)
                self.src_prototype.update(src_emb.detach(), src_label, batch_idx, norm=True)
            tgt_iter, tgt_data_w, tgt_data_s, tgt_label, tgt_idx = acquire_data(tgt_iter, self.tgt_unsup_loader,self.device, mode='enhance')

            data_aug = torch.cat([tgt_data_w, tgt_data_s], dim=0).to(self.device)
            score_aug, emb_aug = self.net(data_aug, True)
            score_w, score_s = torch.chunk(score_aug, 2, dim=0)
            emb_w, emb_s = torch.chunk(emb_aug, 2, dim=0)
            loss = 0.0
            loss += reg_inter(self.src_prototype, emb_w, score_s, self.device)

            prob_tu_w, prob_tu_s = torch.softmax(score_w, dim=1), torch.softmax(score_s, dim=1)

            loss += reg_intra(prob_tu_w, prob_tu_s)

            try:
                data_ts, label_ts, idx_ts, *data_rand_ts = next(tgtl_iter)
            except:
                delete_tmp.sort(reverse=True)
                tgtl_dataset.remove_item(delete_tmp)
                tgtl_iter, delete_tmp = iter(tgtl_loader), []
                data_ts, label_ts, idx_ts, *data_rand_ts = next(tgtl_iter)

            if epoch > 9:
                real_idx, fake_idx, current_idx = torch.tensor(np.array(tgtl_idxs))[idx_ts], [], []
                for num in range(len(real_idx)):
                    if not real_idx[num].item() in tgtl_idxs_fixed:
                        fake_idx.append(idx_ts[num].item())
                        current_idx.append(num)

                if len(fake_idx) > 0:
                    lbl_pl_batch, img_pl_batch = label_ts.argmax(dim=1)[current_idx], data_ts[current_idx]
                    mixup_idx, factual_idx, dataset_idx = [], [], []
                    for iidx in range(len(fake_idx)):
                        label_i = lbl_pl_batch[iidx].item()
                        candidate_idx = torch.nonzero(tgt_label_selected == label_i).squeeze(1)
                        if len(candidate_idx)>0:
                            id_pl_select = random.choice(candidate_idx)
                            mixup_idx.append(id_pl_select.item())
                            factual_idx.append(iidx)
                            dataset_idx.append(fake_idx[iidx])
                    dataset_idx = torch.tensor(np.array((dataset_idx)))

                    if len(dataset_idx)>0:
                        data_factual = img_pl_batch[factual_idx].to(self.device)
                        data_mixup = tgt_data_selected[mixup_idx].to(self.device)
                        mixup_regular_data = 0.5 * data_factual + 0.5 + data_mixup
                        pl_dataset = lbl_pl_batch[factual_idx]
                        with torch.no_grad():
                            output_pl_regular = self.ema.ema(mixup_regular_data)
                            prob_pl_regular = torch.softmax(output_pl_regular, dim=-1)
                            prob_pl_regular, pesudo_pl_regular = prob_pl_regular.max(dim=1)
                        delete_current_id = torch.nonzero(pesudo_pl_regular.detach().cpu() != pl_dataset)
                        if len(delete_current_id)>0:delete_tmp.extend([i.item() for i in dataset_idx[delete_current_id.squeeze(1)]])

            data_ts, label_ts = data_ts.to(self.device), label_ts.to(self.device)
            output_ts, emb_ts = self.net(data_ts, with_emb=True)
            loss += self.CE_Loss(output_ts, label_ts.argmax(dim=1))
            loss += self.InterAlignLoss(self.src_prototype, emb_ts, label_ts.argmax(dim=1))

            num = label_ts.shape[0]
            data_ts_, label_ts_ = mixup(tgt_data_selected, tgt_label_selected, data_ts, label_ts, num, self.device, True)
            data_ts_, label_ts_ = data_ts_.to(self.device), label_ts_.to(self.device)
            output_ts_, emb_ts_ = self.net(data_ts_, with_emb=True)
            loss -= (F.log_softmax(output_ts_, dim=1) * label_ts_).sum() / data_ts_.shape[0]

            cls_proto1 = F.normalize(list(self.ema.ema.classifier.parameters())[0].detach(), dim=-1).cpu()
            cls_proto2 = F.normalize(list(self.ema.ema.classifier.parameters())[4].detach(), dim=-1).cpu()
            segment = torch.einsum('ij,jk,km->im', F.softmax(score_w, dim=1).detach().cpu(), cls_proto2, cls_proto1)
            emb_w_norm = F.normalize(emb_w, dim=-1).detach().cpu()
            embs_rich = emb_w_norm * segment + emb_w_norm
            self.tgt_memorybank.update(epoch, embs_rich, tgt_idx, norm=True)

            self.tgt_opt.zero_grad()
            loss.backward()
            self.tgt_opt.step()
            self.ema.update(self.net)

            if epoch>9:
                output_query,emb_query=F.softmax(output_ts, dim=1).detach().cpu(),F.normalize(emb_ts, dim=-1).detach().cpu()
                segment = torch.einsum('ij,jk,km->im', output_query, cls_proto2, cls_proto1)
                embs_rich = emb_query * segment + emb_query

                tgt_memorybank_selected = self.tgt_memorybank.bank[~epoch_idxs_bool]
                similarity_v = torch.mm(embs_rich, tgt_memorybank_selected.t())
                c1 = torch.sqrt(torch.sum(embs_rich * embs_rich, dim=1, keepdim=True))
                c2 = torch.sqrt(torch.sum(tgt_memorybank_selected * tgt_memorybank_selected, dim=1, keepdim=True))
                similarity_d = torch.mm(c1, c2.t())
                sim = similarity_v / similarity_d
                sim_topk, topk = torch.topk(-1 * sim, k=5, dim=1)

                rand_nn = torch.randint(0, topk.shape[1], (topk.shape[0], 1))
                nn_idxs = torch.gather(topk, dim=-1, index=rand_nn).squeeze(1)
                data_nn_origin = tgt_data_unselected[nn_idxs].to(self.device)
                idxs_nn_origin = tgt_idxs_unlabelled[nn_idxs]
                gts_nn_origin = tgt_label_unselected[nn_idxs]
                with torch.no_grad():
                    output_nn, emb_nn = self.ema.ema(data_nn_origin, with_emb=True)
                    prob_nn = torch.softmax(output_nn, dim=-1)
                    prob, pesudo = prob_nn.max(dim=1)
                    pl_threshold = pesudo[prob>self.cfg.LPDA.A_TH].detach().cpu().numpy()
                    id_threshold = idxs_nn_origin[prob.cpu()>self.cfg.LPDA.A_TH]
                    gt_threshold = gts_nn_origin[prob.cpu()>self.cfg.LPDA.A_TH]

                if len(id_threshold)>0:
                    mixup_idx, factual_idx = [], []
                    factual_data_ = data_nn_origin[prob > self.cfg.LPDA.A_TH]
                    for iid in range(len(pl_threshold)):
                        pesudo_threshold_per = pl_threshold[iid]
                        candidate = torch.nonzero(tgt_label_selected==pesudo_threshold_per).squeeze(1)
                        if len(candidate)>0:
                            id_select = random.choice(candidate)
                            mixup_idx.append(id_select.item())
                            factual_idx.append(iid)
                    if len(mixup_idx)>0:
                        mixup_data = tgt_data_selected[mixup_idx].to(self.device)
                        factual_data = factual_data_[factual_idx]
                        id_threshold = id_threshold[factual_idx]
                        gt_threshold = gt_threshold[factual_idx]
                        mixup_regular_data = factual_data * 0.5 + mixup_data * 0.5
                        with torch.no_grad():
                            output_regular = self.ema.ema(mixup_regular_data)
                            prob_regular = torch.softmax(output_regular, dim=-1)
                            prob_regular, pesudo_regular = prob_regular.max(dim=1)
                            pl_threshold_regular = pesudo_regular[prob_regular>self.cfg.LPDA.S_TH].detach().cpu()
                            id_threshold_regular = id_threshold[prob_regular.cpu()>self.cfg.LPDA.S_TH]
                            gt_threshold_regular = gt_threshold[prob_regular.cpu()>self.cfg.LPDA.S_TH]

                        if len(id_threshold_regular)>0:
                            conf_samples, conf_idx, conf_pl, conf_gt = [], [], [], []
                            dist = np.eye(5)[np.array(tgtl_labels)].sum(0) + 1
                            dist = dist / dist.max()
                            sp = 1 - dist / dist.max() + dist.min() / dist.max()

                            for i in range(pl_threshold_regular.shape[0]):
                                idx = id_threshold_regular[i].item()
                                pl_i = pl_threshold_regular[i].item()
                                gt_i = gt_threshold_regular[i]
                                if np.random.random() <= sp[pl_i]  and idx not in tgtl_idxs:
                                    conf_samples.append((self.tgt_loader.dataset.samples[idx][0], pl_i))
                                    conf_idx.append(idx)
                                    conf_pl.append(pl_i)
                                    conf_gt.append(gt_i)

                            if len(conf_idx)>0:
                                tgtl_dataset.add_item(conf_samples)
                                tgtl_idxs.extend(conf_idx)
                                tgtl_labels.extend(conf_pl)
                                gt_all.extend(conf_gt)
        pl_acc = (np.array(tgtl_labels) == np.array(gt_all)).sum() / len(tgtl_labels)
        extra_pl_num = len(tgtl_labels) - len(tgt_label_selected)
        logging.info('{} Epoch {}: pseudo label acc: {} || extra pseudo label num: {}'.format(resume, epoch, pl_acc,extra_pl_num))
        counts = Counter(tgtl_labels)
        logging.info('{} Epoch {}: active sample: {}, active label: '.format(resume, epoch, len(tgtl_labels)) + str(counts))