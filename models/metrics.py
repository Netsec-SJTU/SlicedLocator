# @Time: 2022.6.4 1:01
# @Author: Bolun Wu

import numpy as np
import torch
import torchmetrics as tm
import torchmetrics.functional as tmf


class Metric(object):
    def __init__(self, name, value, prog_bar=False):
        self.name = name
        self.value = value
        self.prog_bar = prog_bar


class BaseMetric(object):
    def __init__(self, num_classes=None):
        self.num_classes = num_classes
    
    def forward(self):
        raise NotImplementedError
    
    def __call__(self, *args, **kw):
        return self.forward(*args, **kw)
    
    
class CoarsegrainedMetric(BaseMetric):
    def forward(self, g_out, g_y):
        g_acc = tmf.accuracy(g_out, g_y)
        g_recall = tmf.recall(g_out, g_y, average=None, multiclass=True, num_classes=self.num_classes)[1]
        g_fpr = 1 - tmf.specificity(g_out, g_y, average=None, multiclass=True, num_classes=self.num_classes)[1]
        g_f1 = tmf.f1_score(g_out, g_y, average='macro', multiclass=True, num_classes=self.num_classes)
        g_auc = tmf.auroc(g_out, g_y, num_classes=self.num_classes)
        return [Metric('g_auc', g_auc, False),
                Metric('g_f1', g_f1, True),
                Metric('g_acc', g_acc, False), 
                Metric('g_recall', g_recall, False),
                Metric('g_fpr', g_fpr, False)]


class FinegrainedMetric(BaseMetric):
    def forward(self, n_out, n_y, g_y, batch):
        # * only calculate for black samples
        ## get black graph id from g_y
        vul_graph_idx = (g_y==1).nonzero().flatten()
        
        ## get node idx for all black graphs
        vul_n_out = torch.tensor([], device=n_out.device, dtype=torch.float)
        vul_n_y = torch.tensor([], device=n_out.device, dtype=torch.long)
        
        for id in vul_graph_idx:
            node_idx_for_gid = (batch == id).nonzero().flatten()
            local_n_out = torch.index_select(n_out, 0, node_idx_for_gid)
            local_n_y = torch.index_select(n_y, 0, node_idx_for_gid)
            
            vul_n_out = torch.cat([vul_n_out, local_n_out], dim=0)
            vul_n_y = torch.cat([vul_n_y, local_n_y], dim=0)
        
        ## node metric
        if vul_n_y.size(0) == 0:
            n_acc, n_recall, n_fpr, n_f1, n_auc = 0., 0., 0., 0., 0.
        else:
            n_acc = tmf.accuracy(vul_n_out, vul_n_y)
            n_recall = tmf.recall(vul_n_out, vul_n_y, average=None, multiclass=True, num_classes=self.num_classes)[1]
            n_fpr = 1 - tmf.specificity(vul_n_out, vul_n_y, average=None, multiclass=True, num_classes=self.num_classes)[1]
            n_f1 = tmf.f1_score(vul_n_out, vul_n_y, average='macro', multiclass=True, num_classes=self.num_classes)
            n_auc = tmf.auroc(vul_n_out, vul_n_y, num_classes=self.num_classes)
        return vul_n_y.size(0), [Metric('n_auc', n_auc, False),
                                 Metric('n_f1', n_f1, True),
                                 Metric('n_acc', n_acc, False),
                                 Metric('n_recall', n_recall, True),
                                 Metric('n_fpr', n_fpr, True)]


class RankedMetric(BaseMetric):
    def __init__(self, num_classes=None):
        super(RankedMetric, self).__init__(num_classes)
        self.ndgc = tm.RetrievalNormalizedDCG(k=5)
        self.mrr = tm.RetrievalMRR() # average 1/first-rank
    
    def forward(self, n_out, n_y, g_y, batch):
        # * only calculate for black samples
        vul_graph_idx = (g_y==1).nonzero().flatten()
        
        n_out = torch.softmax(n_out, dim=-1)
        
        ## get node idx for all black graphs
        vul_n_out = torch.tensor([], device=n_out.device, dtype=torch.float)
        vul_n_y = torch.tensor([], device=n_out.device, dtype=torch.long)
        indexes = torch.tensor([], device=n_out.device, dtype=torch.long)
        
        for i, id in enumerate(vul_graph_idx):
            node_idx_for_gid = (batch == id).nonzero().flatten()
            local_n_out = torch.index_select(n_out, 0, node_idx_for_gid)[:, 1]
            local_n_y = torch.index_select(n_y, 0, node_idx_for_gid)
            local_indexes = torch.ones(len(local_n_y), device=n_out.device, dtype=torch.long) * i
            
            vul_n_out = torch.cat([vul_n_out, local_n_out], dim=0)
            vul_n_y = torch.cat([vul_n_y, local_n_y], dim=0)
            indexes = torch.cat([indexes, local_indexes], dim=0)
            
        if len(vul_graph_idx) == 0:
            ndgc, mrr = 0., 0.
        else:
            ndgc = self.ndgc(vul_n_out, vul_n_y, indexes=indexes)
            mrr = self.mrr(vul_n_out, vul_n_y, indexes=indexes)
        
        return len(vul_graph_idx), [Metric('ndgc', ndgc, True),
                                    Metric('mrr', mrr, False)]


def precision_at_k_cal(r, k):
    """precision@k"""
    assert k >= 1
    r = r[:k]
    return np.mean(r)


def recall_at_k_cal(r, num_gt, k):
    """recall@k"""
    assert k >= 1
    r = r[:k]
    return np.sum(r) / num_gt


def average_precision_at_k_cal(r, num_gt, k):
    """AP@k"""
    assert k >= 1
    r = r[:k]        
    out = [precision_at_k_cal(r, i+1) for i in range(len(r)) if r[i]]
    return np.sum(out) / num_gt


def first_ranking_cal(r):
    """FR"""
    for i, value in enumerate(r):
        if value != 0:
            return i + 1
    return np.nan


def average_ranking_cal(r):
    """AR"""
    rankings = []
    for i, value in enumerate(r):
        if value != 0:
            rankings.append(i + 1)
    if len(rankings) == 0:
        return np.nan
    return np.mean(rankings)
    

def dcg_at_k_cal(r, k):
    """DCG@k"""
    r = r[:k]
    return np.sum(r / np.log2(np.arange(2, len(r)+2)))


def ndcg_at_k_cal(r, k):
    """nDCG@k"""
    r = r[:k]
    ideal_dcg = dcg_at_k_cal(sorted(r, reverse=True), k) # IDCG
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k_cal(r, k) / ideal_dcg
