# @Time: 2022.6.10 16:44
# @Author: Bolun Wu

# * Finegrained metrics are from information retrieval
# * ref: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)
    
import argparse
import json
import os
import sys

import numpy as np

from utils import *

sys.path.append(os.path.join(root_dir(), 'models'))

import torch
import tqdm
from pytorch_lightning import seed_everything
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch_geometric.loader import DataLoader

from dataset.sdg_graph import SDGGraphDatasetFullFeature
from models.gnn import vuldnl_model_selector
from models.metrics import *


def cal_coarse_metric(predict_result):
    threshold = 0.5
    coarse_labels = [x['gt_coarse'] for _, x in predict_result.items()]
    coarse_probs = [x['coarse'] for _, x in predict_result.items()]
    coarse_preds = [int(x > threshold) for x in coarse_probs]
    
    acc = accuracy_score(coarse_labels, coarse_preds)
    f1 = f1_score(coarse_labels, coarse_preds, average='macro')
    auc = roc_auc_score(coarse_labels, coarse_probs)
    precision = precision_score(coarse_labels, coarse_preds, average=None)[1]
    recall = recall_score(coarse_labels, coarse_preds, average=None)[1]
    cmatrix = confusion_matrix(coarse_labels, coarse_preds)
    fpr = cmatrix[0][1] / (cmatrix[0][1] + cmatrix[0][0])
    
    gather = {'accuracy': acc, 'precision': precision, 
              'recall': recall, 'f1': f1, 'auc': auc, 'fpr': fpr}
    return gather, coarse_labels, coarse_probs


def cal_fine_metric(predict_result, mode='all_vul'):
    assert mode in ('all_vul', 'gt_vul')
    threshold = 0.5
    ranks = [1, 5, 10, 15, 20, 25, 30]
    recall_at_k = {k: 0.0 for k in ranks} # * recall@k
    precision_at_k = {k: 0.0 for k in ranks} # * precision@k
    map_at_k = {k: 0.0 for k in ranks} # * mean average precision@k
    ndcg_at_k = {k: 0.0 for k in ranks} # * ndcg@k
    mfr, mar = 0.0, 0.0 # * mean first ranking, mean average ranking
    total_0, total_1 = 0, 0
    
    for _, result in predict_result.items():
        if mode == 'all_vul' and (result['coarse'] <= threshold or result['gt_coarse'] == 0):
            continue
        elif mode == 'gt_vul' and (len(result['fine']) == 0 or result['gt_coarse'] == 0):
            continue
        gt_fine = result['gt_fine']
        ranked_fine = sorted(result['fine'], key=result['fine'].get, reverse=True)
        ranked_score = [int(l in gt_fine) for l in ranked_fine]
        num_gt = len(gt_fine)
        
        ## * calculate @k metrics (recall@k, precision@k, map@k, ndcg@k)
        for k in ranks:
            precision_at_k[k] += precision_at_k_cal(ranked_score, k)
            recall_at_k[k] += recall_at_k_cal(ranked_score, num_gt, k)
            map_at_k[k] += average_precision_at_k_cal(ranked_score, num_gt, k)
            ndcg_at_k[k] += ndcg_at_k_cal(ranked_score, k)
        
        ## * calculate mfr and mar
        _fr = first_ranking_cal(ranked_score)
        if not np.isnan(_fr):
            mfr += _fr
            mar += average_ranking_cal(ranked_score)
            total_1 += 1
        total_0 += 1
            
    # * average
    for k in recall_at_k.keys(): recall_at_k[k] /= total_0
    for k in precision_at_k.keys(): precision_at_k[k] /= total_0
    for k in map_at_k.keys(): map_at_k[k] /= total_0
    for k in ndcg_at_k.keys(): ndcg_at_k[k] /= total_0
    mfr /= total_1
    mar /= total_1
            
    def __gather_fine(gather, name, metric):
        for k in ranks: gather[f'{name}@{k}'] = metric[k]

    gather = {}
    names = ['recall', 'precision', 'map', 'ndcg']
    for name, metric in zip(names, [recall_at_k, precision_at_k, map_at_k, ndcg_at_k]):
        __gather_fine(gather, name, metric)

    gather['mfr'] = mfr
    gather['mar'] = mar
    return gather
    

def inference_dual(model, test_loader, device, filepath_to_label):
    # * inference for dual model, generate `predict_result`, containing both coarse and fine labels
    predict_result = {}
    fullpaths = test_loader.dataset.get_sdg_val_fps()
    
    i = 0
    for batch in tqdm.tqdm(test_loader):
        batch = batch.to(device)
        n_out, g_out = model(batch) # forward
        n_out, g_out = n_out.cpu(), g_out.cpu()
        
        # convert to probabilities
        n_out = torch.softmax(n_out, dim=-1)
        g_out = torch.softmax(g_out, dim=-1)
        batch = batch.cpu()
        
        # for each graph in the batch
        for j in range(g_out.shape[0]):
            cur_g_out = g_out[j]
            cur_n_out = n_out[batch.batch == j]
            cur_n_line = batch.n_line[batch.batch == j]
            cur_fp = fullpaths[i]
            
            if cur_fp not in predict_result:
                predict_result[cur_fp] = {'coarse': 0.0, 'fine': {}}
            
            coarse_vul_prob = cur_g_out[1].item()
            if coarse_vul_prob > predict_result[cur_fp]['coarse']:
                predict_result[cur_fp]['coarse'] = coarse_vul_prob
            
            for n in range(cur_n_out.shape[0]):
                line = cur_n_line[n].item()
                fine_vul_prob = cur_n_out[n, 1].item()
                if line not in predict_result[cur_fp]['fine']:
                    predict_result[cur_fp]['fine'][line] = fine_vul_prob
                elif fine_vul_prob > predict_result[cur_fp]['fine'][line]:
                    predict_result[cur_fp]['fine'][line] = fine_vul_prob
            i += 1
    
    # complement samples that are not covered by code repr.
    for filepath in list(set(filepath_to_label.keys()) - set(predict_result.keys())):
        predict_result[filepath] = {'coarse': 0.0, 'fine': {}}
    
    # add ground truth
    for filepath in filepath_to_label.keys():
        gt_coarse = filepath_to_label[filepath]['gt_coarse']
        gt_fine = filepath_to_label[filepath]['gt_fine']
        gt_fine.sort()
        
        predict_result[filepath]['gt_coarse'] = gt_coarse
        predict_result[filepath]['gt_fine'] = gt_fine

    return predict_result


if __name__ == '__main__':
    seed_everything(42)
    torch.set_grad_enabled(False)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', type=str, help='path to result dir')
    parser.add_argument('--k', type=int, default=-1, help='only test the k-th fold')
    args = parser.parse_args()
    args.save_dir = os.path.expanduser(args.save_dir)
        
    # load configure file
    with open(os.path.join(args.save_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    if config['cv']: ks = list(range(5))
    else: ks = [0]
    
    gather_list = []
    c_probs, c_labels = [], []
    for k in ks:
        if args.k != -1 and k != args.k: continue
        
        # * test set
        test_set = SDGGraphDatasetFullFeature(
            json_path=config['raw_json_path'],
            vocab_path=config['vocab_path'],
            anno_path=os.path.join(config['annotation_dir'], f'fold_{k}.json'),
            test=True, fold=k
        )
        config['vocab_size'] = len(test_set.vocab_meta['vocab'])
        config['token_type_size'] = len(test_set.vocab_meta['token_type_set'])
        config['max_code_size'] = test_set.vocab_meta['max_code_size']
            
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=config['num_workers'])
        
        # * load trained model
        ckpt_dir = os.path.join(args.save_dir, f'fold_{k}', 'checkpoints')
        model_path = os.path.join(ckpt_dir, os.listdir(ckpt_dir)[0])
        model = vuldnl_model_selector(config)
        model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
        model.eval()
        model = model.to(device)

        # test
        data_root = os.path.dirname(os.path.dirname(config['raw_json_path']))
        label_path = os.path.join(data_root, 'mix_label.json')
        with open(label_path, 'r') as f: fp_to_label_orig = json.load(f)
        
        with open(os.path.join(config['annotation_dir'], f'fold_{k}.json'), 'r') as f:
            annotation = json.load(f)
            test_fps = annotation['val']
        
        filepath_to_label = {fp: label for fp, label in fp_to_label_orig.items() if fp in test_fps}
        # * inference and compute metrics for different types (coarse, fine, dual)
        predict_result = inference_dual(model, test_loader, device, filepath_to_label)
        coarse_metrics, coarse_labels, coarse_probs = cal_coarse_metric(predict_result)
        fine_metrics = cal_fine_metric(predict_result, mode='all_vul')
        gather = {'coarse': coarse_metrics, 'fine': fine_metrics}

        gather_list.append(gather)
        c_probs.append(coarse_probs)
        c_labels.append(coarse_labels)
        
        with open(f'real_world/{config["model_name"]}_test_{k}.json', 'w') as f:
            json.dump(predict_result, f, indent=1)
        
        del test_set, test_loader
        del model
    
    if args.k != -1:
        sys.exit(0)
        
    gather_dict, test_result = {}, {}
    for k in list(gather_list[0].keys()):
        gather_dict[k], test_result[k] = {}, {}
        for _k in gather_list[0][k].keys():
            gather_dict[k][_k] = []
            test_result[k][_k] = []
    for gather in gather_list:
        for k, v in gather.items():
            for _k, _v in v.items():
                gather_dict[k][_k].append(_v)
                test_result[k][_k].append(_v)

    for k, v in gather_dict.items():
        for _k, _v in v.items():
            _mean, _std = float(np.mean(_v)), float(np.std(_v))
            test_result[k][_k] = [_mean, _std]
    
    with open(os.path.join(args.save_dir, 'test_result.json'), 'w') as f:
        json.dump(test_result, f, indent=1)
    
    with open(os.path.join(args.save_dir, 'c_labels.json'), 'w') as f:
        json.dump(c_labels, f, indent=1)
    
    with open(os.path.join(args.save_dir, 'c_probs.json'), 'w') as f:
        json.dump(c_probs, f, indent=1)
        
