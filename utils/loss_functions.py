import torch
import torch.nn as nn
import torch.nn.functional as F

import ot
import random
import numpy as np

from copy import deepcopy


class ChainLoss(nn.Module):
    def __init__(self, same_weight=0.5,num_cls=5,gamma=1.0):
        super().__init__()
        self.idx = torch.arange(0, num_cls)
        self.name = ['cls'+str(num) for num in range(num_cls)]
        self.same_weight = same_weight
        self.other_weight = [0.2, 0.0, -0.2, -0.4]
        self.gamma = gamma

    def forward(self, labels, images_feas, device):
        for i, id_name in zip(self.idx, self.name):
            exec(id_name + '= torch.nonzero(labels==i).squeeze().cpu().numpy().tolist()')
            exec('if not type(' + id_name + ')==list: ' + id_name + ' = [' + id_name + ']')

        contrastive_label = torch.tensor([0]).to(device)
        total_contrastive_loss = torch.tensor(0.).to(device)

        for local_id, (id_label, id_feat) in enumerate(zip(labels, images_feas)):
            id_feat = id_feat.unsqueeze(0)
            name = ['cls0', 'cls1', 'cls2', 'cls3', 'cls4']
            id_name = 'cls' + str(id_label.cpu().item())
            name.remove(id_name)

            exec(id_name + '_ = deepcopy(' + id_name + ')')
            # print(eval(id_name + '_'))
            if eval('len(' + id_name + '_)>1'):
                # print('remove'+ str(local_id))
                exec(id_name + '_.remove(' + str(local_id) + ')')
            postive_id = eval('random.choice(' + id_name + '_)')
            postive_feats = images_feas[postive_id].unsqueeze(0)

            weight_list, other_before = [self.same_weight], None
            negative_feats = torch.tensor([]).to(device)
            max_0, max_id = 0, 0
            for id, o in enumerate(name):
                if max((eval('len(' + o + ')'), max_0)) > max_0:
                    max_0 = max((eval('len(' + o + ')'), max_0))
                    max_id = id
            name[0], name[max_id] = name[max_id], name[0]

            for other in name:
                id_other_weight = int(np.abs(id_label.item() - int(other.split('cls')[-1]))) - 1
                other_weight = self.other_weight[id_other_weight]

                if not eval('len(' + other + ')==0'):
                    weight_list.append(other_weight)
                    negative_id = eval('random.choice(' + other + ')')
                    negative_feat = images_feas[negative_id].unsqueeze(0)
                    negative_feats = torch.cat([negative_feats, negative_feat], dim=0)
                else:
                    weight_list.append(weight_list[-1])
                    if eval('len(' + str(other_before) + ')>1'):
                        exec(str(other_before) + '.remove(' + str(negative_id_before) + ')')
                    negative_id = eval('random.choice(' + str(other_before) + ')')
                    negative_feat = images_feas[negative_id].unsqueeze(0)
                    negative_feats = torch.cat([negative_feats, negative_feat], dim=0)
                other_before, negative_id_before = deepcopy(eval(other)), negative_id

            weight_torch = torch.from_numpy(np.array(weight_list))
            pairs = torch.cat([postive_feats, negative_feats], dim=0)

            feature = F.normalize(id_feat)
            pairs = F.normalize(pairs)
            similarity = feature.mm(pairs.t())
            similarity = similarity - weight_torch.unsqueeze(0).to(device)

            numerator = torch.exp((similarity[0][0]) / self.gamma)
            denominator = numerator + torch.sum(torch.exp((similarity / self.gamma)[0][1:]))
            result = torch.log(numerator / denominator).unsqueeze(0).unsqueeze(0)
            contrastive_loss = nn.NLLLoss()(result, contrastive_label)

            total_contrastive_loss = total_contrastive_loss + contrastive_loss
        loss_cl = total_contrastive_loss / images_feas.size(0)
        return loss_cl


class InterAlignLoss(nn.Module):
    def __init__(self, num_cls, temp, device):
        super().__init__()
        self.device = device
        self.temp = temp
        self.num_cls = num_cls
        self.src_one_hot = F.one_hot(torch.arange(0, num_cls)).float().to(device)

    def forward(self, src_prototype, tgtl_feats, tgtl_label):
        src_proto = F.normalize(src_prototype.mo_pro, dim=1).detach()
        # src_proto = F.normalize(src_prototype, dim=1)
        tgtl_feats = F.normalize(tgtl_feats, dim=1)
        tgtl_one_hot = F.one_hot(tgtl_label, num_classes=self.num_cls).float().to(self.device)
        similarity = torch.exp(torch.mm(tgtl_feats, src_proto.t()) / self.temp)
        pos_mask = torch.mm(tgtl_one_hot, self.src_one_hot.t())
        neg_mask = (~(pos_mask.bool())).float()

        pos = torch.sum(similarity * pos_mask, 1)
        neg = torch.sum(similarity * neg_mask, 1)

        loss = -(torch.mean(torch.log(pos / (pos + neg))))
        return loss