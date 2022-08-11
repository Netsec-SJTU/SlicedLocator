# @Time: 2022.7.28 22:45
# @Author: Bolun Wu
# * this script contains SlicedLocator's whole pipeline (inference from a program)

import argparse
import copy
import json
import os
import sys

import torch
from torch_geometric.data import Batch, Data

root_dir = os.path.dirname(__file__)
helpers_dir = os.path.join(root_dir, 'helpers')
models_dir = os.path.join(root_dir, 'models')
sys.path.append(helpers_dir)
sys.path.append(models_dir)

from helpers.joern import generate_pdgc, plot_graph, run_joern
from helpers.sdg import generate_sdg
from helpers.tree_sitter_parser import TreeSitterParser
from models.gnn import VulDnlDual


if __name__ == '__main__':
    torch.set_grad_enabled(False) # diable gradient computation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # * arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True, help='path to the target program')
    parser.add_argument('-m', '--model_dir', type=str, required=True, help='path to the trained model')
    parser.add_argument('--gpu', action='store_true', help='use gpu')
    args = parser.parse_args()
    
    args.path = os.path.expanduser(args.path)
    args.model_dir = os.path.expanduser(args.model_dir)
    if args.gpu: device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else: device = torch.device('cpu')
    
    filename = os.path.basename(args.path)
    dirname = os.path.dirname(args.path)
    
    # * config
    config_path = os.path.join(args.model_dir, '..', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # * model
    print('Loading trained models...')
    ckpt_dir = os.path.join(args.model_dir, 'checkpoints')
    model_path = os.path.join(ckpt_dir, os.listdir(ckpt_dir)[0])
    model = VulDnlDual(
        embedding_mode=config['embedding_mode'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        conv_layers=config['conv_layers'],
        lstm_layers=config['lstm_layers'],
        vocab_size=config['vocab_size'],
        token_type_size=config['token_type_size'],
        max_code_size=config['max_code_size'],
        pool_type=config['pool_type'],
        kernel_type=config['kernel_type']
    )
    model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
    model = model.to(device)
    model.eval()
    
    # * data
    ## * joern
    print('Running Joern...')
    run_joern(args.path)
    
    ## * sdg
    print('Generating SDGs...')
    sdgs = generate_sdg(args.path, save_csv=False)
    with open(config['vocab_path'], 'r') as f:
        vocab_meta = json.load(f)
        
    ## * plot pdgc (unsliced)
    pdgc, _ = generate_pdgc(args.path, save_csv=False)
    plot_graph(pdgc, name=os.path.join(dirname, f'{filename}_pdgc.html'))
    
    vocab = vocab_meta['vocab']
    max_code_size = vocab_meta['max_code_size']
    vocab_dict = {word: i for i, word in enumerate(vocab)}
    token_type = vocab_meta['token_type_set']
    token_type_dict = {word: i for i, word in enumerate(token_type)}
    
    token_type_no_pad = copy.deepcopy(token_type)
    token_type_no_pad.remove('[PAD]')
    token_type_no_pad_dict = {word: i for i, word in enumerate(token_type_no_pad)}
    
    data_list = []
    tree_sitter_parser = TreeSitterParser(lib_path=os.path.join(helpers_dir, 'c.so'))
    for sdg in sdgs:
        x, x_type, x_type_statistic, mask = [], [], [], []
        for line in sdg['node_line_sym']:
            ## * get token ids and token types
            ## * and statistic on token types
            line_ids, line_types = [], []
            line_types_statistic = [0 for _ in range(len(token_type_no_pad))]
            tree_sitter_parser.parse(line)
            tokens, types = tree_sitter_parser.get_clean_tokens_types()
            for token, type in zip(tokens, types):
                if token not in vocab_dict: line_ids.append(vocab_dict['[UNK]'])
                else: line_ids.append(vocab_dict[token])
                line_types.append(token_type_dict[type])
                line_types_statistic[token_type_no_pad_dict[type]] += 1

            ## * padding
            padding_mask = [1] * len(line_ids)
            for _ in range(max_code_size - len(line_ids)):
                line_ids.append(vocab_dict['[PAD]'])
                line_types.append(token_type_dict['[PAD]'])
                padding_mask.append(0)
            
            x.append(line_ids)
            x_type.append(line_types)
            x_type_statistic.append(line_types_statistic)
            mask.append(padding_mask)
        
        edge_index = sdg['edge']
        edge_attr = sdg['edge_type']
        
        ## * contruct data
        x = torch.tensor(x, dtype=torch.long)
        x_type = torch.tensor(x_type, dtype=torch.long)
        x_type_statistic = torch.tensor(x_type_statistic, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        n_line = torch.tensor(sdg['node_line_no'], dtype=torch.long)
        length = torch.tensor(len(sdg['node_line_sym']), dtype=torch.long)
        
        data = Data(x=x, x_type=x_type, x_type_statistic=x_type_statistic,
                    mask=mask, edge_index=edge_index, edge_attr=edge_attr,
                    n_line=n_line, length=length)
        data_list.append(data)
    
    # * inference
    print('Generating predictions...')
    batch = Batch.from_data_list(data_list)
    batch = batch.to(device)
    n_out, g_out = model(batch) # forward
    n_out, g_out = n_out.cpu(), g_out.cpu()
    batch = batch.cpu()
    
    # * map the sdg result to sample result
    ## * to probabilities
    n_out, g_out = torch.softmax(n_out, dim=-1), torch.softmax(g_out, dim=-1)

    ## * for each graph in the batch
    _predict_result = {'coarse': 0.0, 'fine': {}}
    for j in range(g_out.size(0)):
        cur_g_out = g_out[j]
        cur_n_out = n_out[batch.batch == j]
        cur_n_line = batch.n_line[batch.batch == j]
        
        ## * coarse prediction
        coarse_vul_prob = cur_g_out[1].item()
        if coarse_vul_prob > _predict_result['coarse']:
            _predict_result['coarse'] = coarse_vul_prob
        
        fine_probs = []
        for n in range(cur_n_out.shape[0]):
            line = cur_n_line[n].item()
            fine_vul_prob = cur_n_out[n, 1].item()
            fine_probs.append(fine_vul_prob)
            if line not in _predict_result['fine']:
                _predict_result['fine'][line] = fine_vul_prob
            elif fine_vul_prob > _predict_result['fine'][line]:
                _predict_result['fine'][line] = fine_vul_prob
        sdgs[j]['fine_prediction'] = fine_probs
        
    predict_result = {
        'coarse': _predict_result['coarse'],
        'fine': {},
        'fine_ranked': {}
    }
    
    line_prob_list = [[k, v] for k, v in _predict_result['fine'].items()]
    ranked_by_line = sorted(line_prob_list, key=lambda x: x[0])
    ranked_by_prob = sorted(line_prob_list, key=lambda x: x[1], reverse=True)
    
    for k, v in ranked_by_line: predict_result['fine'][k] = v
    for k, v in ranked_by_prob: predict_result['fine_ranked'][k] = v
    
    ## * prediction result on sample level
    with open(os.path.join(dirname, f'slicedlocator_pred_{filename}.json'), 'w') as f:
        json.dump(predict_result, f, indent=1)

    ## * prediction result on sdg level
    
    with open(os.path.join(dirname, f'{filename}_sdg.json'), 'w') as f:
        json.dump(sdgs, f, indent=1)

