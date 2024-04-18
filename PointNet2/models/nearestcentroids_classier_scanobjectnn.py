import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import torch.nn as nn
import torch
import numpy as np
import time 
from torch_scatter import scatter_mean, scatter_max
from .clustering_methods import AGD_gpu 


class NCC_LVQ21(nn.Module):
    def __init__(self, device, num_classes, k=15, mu=0.999, mode = 'train'):
        super(NCC_LVQ21, self).__init__()

        self.mode = mode

        # Sinkhorn-Knopp stuff
        self.K = k # number of subclusters
        self.dev = device
        self.dtype = torch.float64

        # MEM stuff
        self.num_classes = num_classes
        self.dim = 64
        self.mu = mu

        # cluster_center
        self.register_buffer("cluster_center", torch.randn((self.num_classes, self.K, self.dim),requires_grad=False).to(self.dev))
        self.cluster_center = nn.functional.normalize(self.cluster_center, p=2, dim=2).detach_()

        self.cosine = nn.CosineSimilarity(dim=3)

        # LVQ stuff
        self.n_updates = 0
        self.step = 1e-4
        self.n_updates_to_stepdrop = 475 * 200 + 10
        self.minstep = 1e-6
        self.epsilon = 0.4
        self.slowdown_rate = 0.4

        self.register_buffer("point_queue", torch.randn((self.num_classes, 200, self.dim),requires_grad=False).to(self.dev))
        self.point_queue = nn.functional.normalize(self.point_queue, p=2, dim=2)
        self.point_queue_ptr = torch.zeros(self.num_classes, dtype=torch.long,requires_grad=False).to(self.dev)

    @property
    def training_step(self):
        if self.n_updates_to_stepdrop is None:
            return self.step

        updates_ratio = (1 - self.n_updates / self.n_updates_to_stepdrop)
        return self.minstep + (self.step - self.minstep) * updates_ratio


    def batch_training_update(self, feats, cosine_sim, targets):
        batch_size = feats.shape[0]
        targets = targets.contiguous().view(batch_size, -1).detach()
        cosine_sim = cosine_sim.detach()
        feats = feats.squeeze(1).detach()

        max_logits, max_indices = torch.max(cosine_sim, dim=2)
        m_values, m_indices = torch.topk(max_logits, 2, dim=1, largest=True, sorted=True) 

        top1_class = m_indices[:, 0:1]
        top1_subclass = torch.gather(max_indices, 1, top1_class)[:,0]
        top2_class = m_indices[:, 1:2]
        top2_subclass = torch.gather(max_indices, 1, top2_class)[:,0]

        closest_dists  = m_values[:, 0:1]
        runner_up_dists = m_values[:, 1:2]

        top1_weight_update = feats - self.cluster_center[top1_class.squeeze(1), top1_subclass]
        top2_weight_update = feats - self.cluster_center[top2_class.squeeze(1), top2_subclass]

        is_correct_prediction = (top1_class == targets)

        double_update_condition_satisfied = (
            (
                ((top1_class == targets) & (top2_class != targets)) |
                ((top1_class != targets) & (top2_class == targets))
            ) &
            (
                (closest_dists > ((1 - self.epsilon) * runner_up_dists)) &
                (runner_up_dists < ((1 + self.epsilon) * closest_dists))
            )
        )

        step = self.training_step
        assert step > 0

        flag1 = double_update_condition_satisfied & is_correct_prediction
        self.cluster_center[top1_class.squeeze(1), top1_subclass] += step * top1_weight_update * flag1.float()
        self.cluster_center[top2_class.squeeze(1), top2_subclass] -= step * top2_weight_update * flag1.float()

        flag2 = double_update_condition_satisfied & (~ is_correct_prediction)
        self.cluster_center[top1_class.squeeze(1), top1_subclass] -= step * top1_weight_update * flag2.float()
        self.cluster_center[top2_class.squeeze(1), top2_subclass] += step * top2_weight_update * flag2.float()

        flag3 = (~ double_update_condition_satisfied) & is_correct_prediction
        self.cluster_center[top2_class.squeeze(1), top2_subclass] += step * top1_weight_update * flag3.float()

        flag4 = (~ double_update_condition_satisfied) & (~ is_correct_prediction)
        self.cluster_center[top1_class.squeeze(1), top1_subclass] -= step * top1_weight_update * flag4.float()
        self.cluster_center = nn.functional.normalize(self.cluster_center, p=2, dim=2).detach_()

        self.n_updates += 1
        return 

    def _batch_update_subclass_centers(self, feats, labels):
        feats = feats.squeeze(1)

        this_classes = torch.unique(labels)
        for cls_id in this_classes:
            class_indices = (labels.squeeze(1) == cls_id).nonzero(as_tuple=False)
            indices = class_indices.squeeze(1)
            xc = feats[indices]   
            pc = self.cluster_center[cls_id]
            self.PS = torch.mm(xc, pc.t())  # N * K N是样本数量，k是子类数量 
            self.PS = nn.functional.normalize(self.PS, p=2, dim=1)
            self.L = AGD_gpu(self.PS)

            step = self.training_step / 10
            new_cluster_center = pc*(1 - step) + step *scatter_mean(xc, self.L, dim=0, dim_size=self.K)
            self.cluster_center[cls_id] = nn.functional.normalize(new_cluster_center, p=2, dim=1).detach_()

        return


    def forward(self, feats, labels=None,epoch=None):
        labels = labels.long()
        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])
        feats = nn.functional.normalize(feats, p=2, dim=2)

        if feats.requires_grad:
            self._batch_update_subclass_centers(feats.detach(), labels)

        cosine_sim = self.cosine(feats.unsqueeze(1).expand(batch_size, self.num_classes, self.K, self.dim), 
                            self.cluster_center.unsqueeze(0).expand(batch_size, self.num_classes, self.K, self.dim))

        return feats, cosine_sim
