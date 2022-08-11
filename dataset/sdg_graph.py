# @Time: 2022.6.4 10:34
# @Author: Bolun Wu

import copy
import json
import os
import sys

import torch

current_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(current_dir)
helper_dir = os.path.join(root_dir, 'helpers')
sys.path.append(root_dir)
import tqdm
from helpers.tree_sitter_parser import TreeSitterParser
from torch_geometric.data import Data, InMemoryDataset

seed = 42
torch.set_grad_enabled(False)


def batch(iterable, n=1):
    """split an iterable in batch
    Ref: https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


class SDGGraphDatasetFullFeature(InMemoryDataset):
    def __init__(self, json_path:str, vocab_path: str, anno_path=None, test=False, fold=0):
        self.json_path = json_path
        self.vocab_path = vocab_path
        self.anno_path = anno_path
        self.fold = fold
        self.test = test
        with open(self.vocab_path, 'r') as f:
            self.vocab_meta = json.load(f)
            
        super(SDGGraphDatasetFullFeature, self).__init__()
    
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_dir(self) -> str:
        json_dir = os.path.dirname(self.json_path)
        return os.path.join(json_dir, f'pyg_e2e_fullfeature')

    @property
    def processed_file_names(self):
        json_name = os.path.basename(self.json_path).split('.')[0]
        if self.test:
            json_name = f'{json_name}_test_fold_{self.fold}'
        return [f'{json_name}.pt']
    
    @property
    def processed_paths(self):
        return [os.path.join(self.processed_dir, fname) for fname in self.processed_file_names]

    def get_annotation(self):
        with open(self.anno_path, 'r') as f:
            annotation = json.load(f)
        
        train_fps, val_fps = annotation['train'], annotation['val']
        train_idx, val_idx = [], []
        
        sdg_data = self.get_sdgs()
        for i, data in enumerate(sdg_data):
            if data['filepath'] in train_fps: train_idx.append(i)
            elif data['filepath'] in val_fps: val_idx.append(i)
            else: raise Exception('filepath not in train_fps or val_fps')
        return train_idx, val_idx 
        
    def get_sdgs(self):
        with open(self.json_path, 'r') as f:
            sdg_data = json.load(f)
        if self.test:
            with open(self.anno_path, 'r') as f:
                annotation = json.load(f)
            sdg_data = [sdg for sdg in sdg_data if sdg['filepath'] in annotation['val']]
        return sdg_data

    def get_sdg_val_fps(self):
        sdg_data = self.get_sdgs()
        with open(self.anno_path, 'r') as f:
            annotation = json.load(f)
        sdg_val_fps = [sdg['filepath'] for sdg in sdg_data if sdg['filepath'] in annotation['val']]
        return sdg_val_fps
    
    def process(self):
        sdg_data = self.get_sdgs()
        
        # * load some vocab meta
        max_code_size = self.vocab_meta['max_code_size']
        
        vocab = self.vocab_meta['vocab']
        vocab_dict = {word: i for i, word in enumerate(vocab)}
        
        token_type = self.vocab_meta['token_type_set']
        token_type_dict = {word: i for i, word in enumerate(token_type)}
        
        token_type_no_pad = copy.deepcopy(token_type)
        token_type_no_pad.remove('[PAD]')
        token_type_no_pad_dict = {word: i for i, word in enumerate(token_type_no_pad)}

        data_list = []
        tree_sitter_parser = TreeSitterParser(lib_path=os.path.join(helper_dir, 'c.so'))
        for sdg in tqdm.tqdm(sdg_data):
            ## * get x, x_type, x_type_statistic, mask
            ## * x contains lines of symbolized code, each token is a integer (id in vocab)
            ## * x_type contains lines of token type, each token type is a integer (id in token_type)
            ## * x_type_statistic is the statistic of token types for each line of code
            ## * mask is the mask of x, 1 for token, 0 for padding (used by subsequent attention mechanism)
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
            
            g_y = sdg['label']
            if g_y == 0:
                n_y = [-100 for _ in range(len(sdg['node_line_sym']))]
            else:
                n_y = []
                for line_no in sdg['node_line_no']:
                    if line_no in sdg['vul_line_no']: n_y.append(1)
                    else: n_y.append(0)
            
            # construct data
            x = torch.tensor(x, dtype=torch.long)
            x_type = torch.tensor(x_type, dtype=torch.long)
            x_type_statistic = torch.tensor(x_type_statistic, dtype=torch.float)
            mask = torch.tensor(mask, dtype=torch.long)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)
            n_y = torch.tensor(n_y, dtype=torch.long)
            g_y = torch.tensor(g_y)
            n_line = torch.tensor(sdg['node_line_no'], dtype=torch.long)
            length = torch.tensor(len(sdg['node_line_sym']), dtype=torch.long)
            
            data = Data(x=x, x_type=x_type, x_type_statistic=x_type_statistic,
                        mask=mask, edge_index=edge_index, edge_attr=edge_attr,
                        n_y=n_y, g_y=g_y, n_line=n_line, length=length)
        
            data_list.append(data)
            
        torch.save(self.collate(data_list), self.processed_paths[0])
        
