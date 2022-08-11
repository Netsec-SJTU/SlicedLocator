# @Time: 2022.6.24 19:21
# @Author: Bolun Wu

import json
import os

from sklearn.model_selection import StratifiedKFold

from cve import export_cve_func_code_label
from sard import generate_sard_label
from sdg import generate_sdg_for_dataset

seed = 42

def generate_mix_label(sard_c_dir, cve_db_path, save_dir):
    sard_fp_to_label = generate_sard_label(sard_c_dir, save_dir=save_dir)
    cve_fp_to_label = export_cve_func_code_label(cve_db_path, save_dir=save_dir, filters=True)
    os.remove(os.path.join(save_dir, 'sard_label.json'))
    os.remove(os.path.join(save_dir, 'cve_label.json'))
    
    fp_to_label = {}
    for k, v in sard_fp_to_label.items(): fp_to_label[k] = v
    for k, v in cve_fp_to_label.items(): fp_to_label[k] = v
    
    with open(os.path.join(save_dir, 'mix_label.json'), 'w') as f:
        json.dump(fp_to_label, f, indent=1)
    print(f'mix total samples: {len(fp_to_label)}')
    return fp_to_label


def generate_annotation(label_path):
    label_dir = os.path.dirname(label_path)
    anno_dir = os.path.join(label_dir, 'annotation')
    os.makedirs(anno_dir, exist_ok=True)
    
    with open(label_path, 'r') as f:
        fp_to_label = json.load(f)

    print('generating mix annotation...')
    fp_list = list(fp_to_label.keys())
    id_list = list(range(len(fp_list)))
    label_list = [fp_to_label[fp]['gt_coarse'] for fp in fp_list]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    for k, (train_idx, val_idx) in enumerate(skf.split(id_list, label_list)):
        train_fp = [fp_list[i] for i in train_idx]
        val_fp = [fp_list[i] for i in val_idx]
        with open(os.path.join(anno_dir, f'fold_{k}.json'), 'w') as f:
            json.dump({'train': train_fp, 'val': val_fp}, f, indent=1)
        
        print(f'fold {k}, train {len(train_fp)}, val {len(val_fp)}.')


def generate_mix_sdg(label_path, num_proc=8):
    try: assert os.path.exists(label_path)
    except: generate_mix_label(
        sard_c_dir='/home/wubolun/data/codevul/SARD/c',
        cve_db_path='/home/wubolun/data/codevul/CVEfixes_v1.0.0/Data/CVEfixes.db',
        save_dir='/home/wubolun/data/codevul/sard+cve'
    )
    generate_sdg_for_dataset(label_path, 'mix', num_proc=num_proc)


if __name__ == '__main__':
    pass
    # generate_mix_label(sard_c_dir='/home/wubolun/data/codevul/SARD/c',
    #                    cve_db_path='/home/wubolun/data/codevul/CVEfixes_v1.0.0/Data/CVEfixes.db',
    #                    save_dir='/home/wubolun/data/codevul/sard+cve')
    
    # generate_annotation(label_path='/home/wubolun/data/codevul/sard+cve/mix_label.json')
    
    # generate_mix_sdg(
    #     label_path='/home/wubolun/data/codevul/sard+cve/mix_label.json',
    #     num_proc=8
    # )