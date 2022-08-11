# @Time: 2022.6.6 00:47
# @Author: Bolun Wu

import ast
import copy
import hashlib
import json
import os
import sqlite3

import pandas as pd

from joern import root_dir
from sdg import comment_remover

# constants
seed = 42

def export_cve_func_code_label(db_path, save_dir, export_c=False, export_csv=False, filters=False):
    """export cve functions code and label JSON file
    Ref: https://github.com/secureIT-project/CVEfixes
    
    Args:
        db_path (str): path to SQLite database
        save_dir (str): directory to save .c .cpp files
        export_c (bool, optional): whether to export .c .cpp files. Defaults to False.
    """    
    
    c_dir = os.path.join(os.path.dirname(os.path.dirname(db_path)), 'c')
    os.makedirs(c_dir, exist_ok=True)
    
    try: conn = sqlite3.connect(db_path, timeout=10)
    except: raise Exception(f'Failed to connect to database: {db_path}')
    
    query = """
    SELECT m.signature, m.start_line, m.end_line, m.code, m.before_change, 
        f.old_path, f.diff_parsed, f.programming_language,
        fixes.repo_url as project,
        cwe_c.cve_id, cwe_c.cwe_id
    FROM method_change m, file_change f, cwe_classification cwe_c, fixes
    WHERE f.file_change_id=m.file_change_id and f.hash=fixes.hash and fixes.cve_id=cwe_c.cve_id
        and (f.programming_language='C' or f.programming_language='C++')
    """
    
    print('Querying SQLite database...')
    df = pd.read_sql_query(query, conn)
    
    print('Filtering data...')
    ## * consider only .c .cpp files
    df = df[(df['old_path'].str.endswith('.c'))|(df['old_path'].str.endswith('.cpp'))]
    ## * fix project name
    df['project'] = df['project'].apply(lambda x: x.split('/')[-1])
    ## * fix diff_parsed to JSON
    df['diff_parsed'] = df['diff_parsed'].apply(lambda x: ast.literal_eval(x))

    ## * consider only those with delete operation in diff
    def __contain_diff_delete(row):
        diff_parsed = row['diff_parsed']
        if len(diff_parsed['deleted']) == 0: return False
        else: return True
    m = df.apply(__contain_diff_delete, axis=1)
    df = df[m]

    if filters:
        ## * consider top 15 projects (contain the top number of CVEs)
        df_tmp = df.groupby(['project', 'cve_id'], as_index=False).count()[['project', 'cve_id']]
        top_projects = list(df_tmp['project'].value_counts()[:15].index)
        df = df[df['project'].isin(top_projects)]
        with open(os.path.join(root_dir, 'coi', 'cve_project.txt'), 'w') as f:
            for project in top_projects: f.write(f'{project}\n')

        ## * consider top 20 CWE types
        top_cwe_ids = list(df.cwe_id.value_counts()[:20].index)
        # top_cwe_ids.remove('NVD-CWE-Other')
        # top_cwe_ids.remove('NVD-CWE-noinfo')
        df = df[df['cwe_id'].isin(top_cwe_ids)]
        with open(os.path.join(root_dir, 'coi', 'cve_cwe.txt'), 'w') as f:
            for cwe in top_cwe_ids: f.write(f'{cwe}\n')

    ## * add sig
    def __calculate_sig(row):
        unique = f"{row['cve_id']}@{row['project']}@{row['old_path']}@{row['signature']}"
        return hashlib.md5(unique.encode()).hexdigest()
    df['sig'] = df.apply(__calculate_sig, axis=1)

    with open(os.path.join(root_dir, 'coi', 'cve_joern_error.txt'), 'r') as f:
        error_sigs = f.readlines()
        error_sigs = list(map(lambda x: x.strip(), error_sigs))
    
    m = df.apply(lambda x: x['sig'] not in error_sigs, axis=1)
    df = df[m]

    if export_csv:
        df.to_csv('cve_summary.csv', index=False)
        
    with open(os.path.join(root_dir, 'coi', 'cve_fine_error.txt'), 'r') as f:
        error_fnames = f.readlines()
        error_fnames = list(map(lambda x: x.strip(), error_fnames))
        
    print('Constructing methods dict...')
    cve_methods_dict = {}
    for _, row in df.iterrows():
        sig = row['sig']
        if sig not in cve_methods_dict:
            cve_methods_dict[sig] = {
                'before': '', 'after': '', 'vul_lines': [],
                'project': row['project'], 'cve_id': row['cve_id'], 'cwe_id': row['cwe_id'],
                'old_path': row['old_path'], 'method': row['signature'],
            }

        if row['before_change'] == 'True':
            vul_lines = [int(x[0]) for x in row['diff_parsed']['deleted'] if x[1].strip() != '']
            vul_lines = [x - int(row['start_line']) + 1 for x in vul_lines]
            max_line = int(row['end_line']) - int(row['start_line']) + 1
            vul_lines = list(filter(lambda x: 1 <= x <= max_line, vul_lines))
            
            cve_methods_dict[sig]['before'] = row['code']
            for vul_line in vul_lines:
                if vul_line not in cve_methods_dict[sig]['vul_lines']:
                    cve_methods_dict[sig]['vul_lines'].append(vul_line)
        else:
            cve_methods_dict[sig]['after'] = row['code']
    
    print('Constructing labels dict...')
    filepath_to_label = {} # store labels
    for sig, method_info in cve_methods_dict.items():
        old_path = method_info['old_path']
        postfix = old_path.split('.')[-1]        
        cve_id = method_info['cve_id']
        dirpath = os.path.join(c_dir, cve_id)
        os.makedirs(dirpath, exist_ok=True)
        label_info_template = {
            'cve_id': cve_id, 'project': method_info['project'], 'cwe_id': method_info['cwe_id'],
            'method': method_info['method'], 'old_path': old_path
        }
        
        if method_info['before'] != '':
            filename = f'{sig}_before.{postfix}'
            filepath = os.path.join(dirpath, filename)
            if export_c:
                with open(filepath, 'w') as f: f.write(method_info['before'])
            label_info = copy.deepcopy(label_info_template)
            label_info['gt_coarse'] = 1
            label_info['gt_fine'] = sorted(method_info['vul_lines'])

            # * assure that vulnerable lines are within code file
            with open(filepath, 'r') as f: code = f.readlines()
            
            # * remove invalid vul lines
            invalid_line = set()
            for line in label_info['gt_fine']:
                curr_line = comment_remover(code[line-1]).strip()
                if curr_line in ['', '}', '{', 'return;', '{}', '} else'] or curr_line.startswith('/*') or curr_line.startswith('*') or curr_line.startswith('#'):
                    invalid_line.add(line)
            invalid_line = list(invalid_line)
            for l in invalid_line:
                label_info['gt_fine'].remove(l)
                
            for line in label_info['gt_fine']:
                assert line <= len(code)
            
            if len(label_info['gt_fine']) > 0 and 5 <= len(code) <= 500 and filename not in error_fnames:
                filepath_to_label[filepath] = label_info
            
        if method_info['after'] != '':
            filename = f'{sig}_after.{postfix}'
            filepath = os.path.join(dirpath, filename)
            if export_c:
                with open(filepath, 'w') as f: f.write(method_info['after'])
            with open(filepath, 'r') as f: code = f.readlines()
            label_info = copy.deepcopy(label_info_template)
            label_info['gt_coarse'] = 0
            label_info['gt_fine'] = []
            if 5 <= len(code) <= 500:
                filepath_to_label[filepath] = label_info

    with open(os.path.join(save_dir, 'cve_label.json'), 'w') as f:
        json.dump(filepath_to_label, f, indent=1)
    
    print(len(filepath_to_label))
    return filepath_to_label

