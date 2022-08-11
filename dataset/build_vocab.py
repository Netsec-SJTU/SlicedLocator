# @Time: 2022.6.29 18:12
# @Author: Bolun Wu

import argparse
import json
import os
import sys
from collections import Counter

import tqdm

current_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(current_dir)
helper_dir = os.path.join(root_dir, 'helpers')
sys.path.append(root_dir)
from helpers.tree_sitter_parser import TreeSitterParser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=('sdg', 'funcpdg', 'pdgc'), default='sdg')
    args = parser.parse_args()
    
    tree_sitter_parser = TreeSitterParser(lib_path=os.path.join(helper_dir, 'c.so'))
    json_path = 'data/mix_sdg_clean.json'

    with open(json_path, 'r') as f:
        sdg_data = json.load(f)
    
    token_min_freq = 3
    token_count = Counter()
    max_code_size, max_slice_size = 0, 0
    token_type_set = set()
    
    for data in tqdm.tqdm(sdg_data):
        if len(data['node_line_sym']) > max_code_size:
            max_slice_size = len(data['node_line_sym'])
        for line in data['node_line_sym']:
            if line == '':
                print(data)
                exit(0)
            tree_sitter_parser.parse(line)
            tokens, types = tree_sitter_parser.get_clean_tokens_types()
            if len(tokens) == 0:
                print(line)
            token_count.update(tokens)
            for type in types: token_type_set.add(type)
            if len(tokens) > max_code_size: max_code_size = len(tokens)
        
    prev, tail = ['[PAD]', '[UNK]'], []
    for k, v in token_count.items():
        if v >= token_min_freq:
            tail.append(k)
    vocab = prev + tail
    
    token_type_set = list(token_type_set)
    token_type_set.sort()
    token_type_set = ['[PAD]'] + token_type_set

    with open(os.path.join(current_dir, f'{args.type}_vocab.json'), 'w') as f:
        json.dump({'max_slice_size': max_slice_size,
                   'max_code_size': max_code_size,
                   'token_type_set': token_type_set,
                   'vocab': vocab}, 
                  f, indent=1)

    print(f'vocab size: {len(vocab)}')
