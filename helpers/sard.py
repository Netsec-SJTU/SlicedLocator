# @Time: 2022.5.28 15:05
# @Author: Bolun Wu

import json
import os

import tqdm

from joern import root_dir
from normalizer import comment_remover

# constants
seed = 42

def generate_sard_label(c_dir, save_dir=None):
    """this is for new version of SARD
    Ref: 1. https://samate.nist.gov/SARD/
         2. https://samate.nist.gov/SARD/sard-schema.json

    Args:
        c_dir (str): directory containing testcases
    """
    with open(os.path.join(root_dir, 'coi/sard_cwe.txt'), 'r') as f:
        fix_cwes = f.readlines()
        fix_cwes = list(map(lambda x: x.strip(), fix_cwes))
        
    filepath_to_label = {}
    for testcase_id in tqdm.tqdm(os.listdir(c_dir)):
        dirpath = os.path.join(c_dir, testcase_id)
        if not os.path.isdir(dirpath): continue
        
        manifest_path = os.path.join(dirpath, 'manifest.sarif')
        if not os.path.exists(manifest_path): continue
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # filter deprecated
        status = manifest['runs'][0]['properties']['status']
        if status == 'deprecated': continue
        
        # filter not in fix cwes
        discard_cwe = True
        for record in manifest['runs'][0]['results']:
            cwe_type = record['ruleId']
            if cwe_type in fix_cwes:
                discard_cwe = False
                break
        if manifest['runs'][0]['properties']['state'] == 'good': discard_cwe = False
        if discard_cwe: continue
        
        # multiple vul lines
        if manifest['runs'][0]['properties']['state'] in ('bad', 'mixed'):
            for record in manifest['runs'][0]['results']:
                for loc in record['locations']:
                    file = loc['physicalLocation']['artifactLocation']['uri']
                    line = loc['physicalLocation']['region']['startLine']
                    filepath = os.path.join(dirpath, file)
                    
                    # * we discard the empty vulnerable line
                    # * e.g. defect like missing f.close()
                    with open(filepath, 'r', errors='ignore') as f:
                        code = f.readlines()
                        curr_line = comment_remover(code[line-1]).strip()
                        if curr_line == '' or curr_line == '}':
                            continue
                    
                    if filepath not in filepath_to_label:
                        info = {'testcase_id': testcase_id,
                                'gt_coarse': 1,
                                'gt_fine': [line]}
                        filepath_to_label[filepath] = info
                    else:
                        if line not in filepath_to_label[filepath]['gt_fine']:
                            filepath_to_label[filepath]['gt_fine'].append(line)

        elif manifest['runs'][0]['properties']['state'] == 'good':
            # print('good sample')
            for record in manifest['runs'][0]['results']:
                for loc in record['locations']:
                    file = loc['physicalLocation']['artifactLocation']['uri']
                    
                    filepath = os.path.join(dirpath, file)
                    if filepath not in filepath_to_label:
                        info = {'testcase_id': testcase_id,
                                'gt_coarse': 0,
                                'gt_fine': []}
                        filepath_to_label[filepath] = info 

    if not save_dir: save_dir = os.path.dirname(c_dir)
    print(len(filepath_to_label))
    with open(os.path.join(save_dir, 'sard_label.json'), 'w') as f:
        json.dump(filepath_to_label, f, indent=1)
        
    return filepath_to_label
