# @Time: 2022.5.19 19:15
# @Author: Bolun Wu

import json
import logging
import os
import subprocess
import sys
from difflib import SequenceMatcher

sys.path.append(os.path.dirname(__file__))
import networkx as nx
import pandas as pd

from normalizer import comment_remover

# constants
helpers_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(helpers_dir)
logs_dir = os.path.join(root_dir, 'logs')
html_dir = os.path.join(root_dir, 'html')
os.makedirs(logs_dir, exist_ok=True)


logging.basicConfig(filename=os.path.join(logs_dir, 'joern.log'),
                    filemode='a',
                    level=logging.NOTSET,
                    format='%(asctime)s - %(levelname)s - %(process)d - %(funcName)s - %(message)s')


def run_joern(filepath: str, i=0):
    """Use Joern to generate a graph for a given program

    Args:
        filepath (str): path to a .c file
    """

    assert os.path.isfile(filepath)    
    scala_script = os.path.join(helpers_dir, 'gen_graph.scala')
    print(i, filepath)
    
    if not os.path.exists(filepath):
        logging.error(f'{filepath}: FAIL. filepath does not exist.')
        return False
    
    savedir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    
    if os.path.exists(os.path.join(savedir, f'{filename}_edges.json')) and \
       os.path.exists(os.path.join(savedir, f'{filename}_nodes.json')):
        logging.info(f'{filepath}: PASS. Already parsed by Joern.')
        return True

    command = f'joern --script {scala_script} --params filepath={filepath},saveprefix={filepath}'
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(os.path.join(savedir, f'{filename}_edges.json')) and \
       os.path.exists(os.path.join(savedir, f'{filename}_nodes.json')):
        logging.info(f'{filepath}: SUCCESS.')
        return True
    else:
        logging.error(f'{filepath}: FAIL. Joern error.')
        return False


def check_node_edge_file(filepath):
    save_dir = os.path.dirname(filepath)
    file_name = os.path.basename(filepath)
    
    nodefile = os.path.join(save_dir, f'{file_name}_nodes.json')
    edgefile = os.path.join(save_dir, f'{file_name}_edges.json')  
    return os.path.exists(nodefile) and os.path.exists(edgefile)


def get_node_edges(filepath: str, save=False):
    """Get nodes and edges given the dirpath of `nodes.json` and `edges.json`
    (must run after `run_joern`)

    Args:
        savedir (str): dirpath containing JSON files
    """
    with open(filepath, 'r', errors='ignore') as f:
        source_code = f.readlines()
        source_code = list(map(lambda x: x.strip(), source_code))
        source_code = list(map(comment_remover, source_code))
        source_code = list(map(lambda x: x.strip(), source_code))
        source_code = list(map(lambda x: x.replace(' ', ''), source_code))

    savedir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    
    nodefile = os.path.join(savedir, f'{filename}_nodes.json')
    edgefile = os.path.join(savedir, f'{filename}_edges.json')
    
    assert os.path.exists(nodefile) and os.path.exists(edgefile)
    
    with open(edgefile, 'r') as f:
        edges = json.load(f)
        edges = pd.DataFrame(edges, columns=['src_node', 'dst_node', 'etype', 'dataflow'])
        edges.fillna('', inplace=True)
        # edges.to_csv('edges.csv', index=False)
        
    with open(nodefile, 'r') as f:
        nodes = json.load(f)
        nodes = pd.DataFrame.from_records(nodes)
        if 'controlStructureType' not in nodes.columns: # IF ELSE GOTO WHILE ...
            nodes['controlStructureType'] = ''
        nodes.fillna('', inplace=True)
        # nodes.to_csv('nodes.csv', index=False)
        try:
            nodes = nodes[['id', '_label', 'name', 'code', 'lineNumber', 'isExternal', 'controlStructureType']]
        except:
            logging.error(f'{savedir}: FAIL. node error.')
            return None
        
    nodes.fillna('', inplace=True)
    
    # assign node name to node code if code is None
    nodes.code = nodes.apply(lambda x: "" if x.code == "<empty>" else x.code, axis=1)
    nodes.code = nodes.apply(lambda x: x.code if x.code != "" else x['name'],axis=1)
    
    # assign node label for printing in the graph
    nodes['node_label'] = nodes._label + '_' + nodes.lineNumber.astype(str) + ': ' + nodes.code

    def __filter_node(row):
        if row['lineNumber'] == '':
            return False
        line_num = int(row['lineNumber'])
        if line_num > len(source_code):
            return False
        
        if source_code[line_num-1].startswith('THE'):
            return False
        if source_code[line_num-1] == '':
            return False
        
        if row['isExternal'] == 'true':
            return False
         
        if row['code'] != '': # get common substrings to check validity
            if row['_label'] == 'LOCAL' and '[' in row['code'] and ']' in row['code']:
                pos_l_bracket = row['code'].index('[')
                pos_r_bracket = row['code'].index(']')
                code_joern = row['code'][:pos_l_bracket] + row['code'][pos_r_bracket+1:] + row['code'][pos_l_bracket:pos_r_bracket+1]
                code_joern = code_joern.replace(' ', '')
            else:
                code_joern = row['code'].replace(' ', '')

            code_file = source_code[line_num-1]
            ## * dealing with CWE-78 in SARD
            if 'EXECL(' in code_file or 'EXECLP(' in code_file:
                return True

            match_ratio = SequenceMatcher(None, code_file, code_joern).find_longest_match()[2] / len(code_joern)
            if match_ratio < 0.5:
                return False

        return True

    m = nodes.apply(__filter_node, axis=1)
    nodes = nodes[m]
    
    # filter node type
    nodes = nodes[nodes._label != 'COMMENT']
    nodes = nodes[nodes._label != 'FILE']
    nodes = nodes[nodes._label != 'MEMBER']
    
    # filter edge type
    edges = edges[edges.etype != 'CONTAINS']
    edges = edges[edges.etype != 'SOURCE_FILE']
    edges = edges[edges.etype != 'DOMINATE']
    edges = edges[edges.etype != 'POST_DOMINATE']
    
    # merge node features into edges dataframe -> `merge``
    merge = edges.merge(nodes[['id', '_label', 'name', 'code', 'lineNumber', 'controlStructureType']].rename(columns={
            '_label': 'src_label', 'name': 'src_name', 'code': 'src_code',
            'lineNumber': 'src_lineNumber', 'controlStructureType': 'src_controlStructureType'
        }), left_on='src_node', right_on='id')
    
    merge = merge.merge(nodes[['id', '_label', 'name', 'code', 'lineNumber', 'controlStructureType']].rename(columns={
            '_label': 'dst_label', 'name': 'dst_name', 'code': 'dst_code',
            'lineNumber': 'dst_lineNumber', 'controlStructureType': 'dst_controlStructureType'
        }), left_on='dst_node', right_on='id')
    # merge.to_csv('merge.csv', index=False)
    
    # remove those lineNumber is empty edges
    merge = merge[(merge.src_lineNumber != '') & (merge.dst_lineNumber != '')]
    
    # remove those lineNumber is empty nodes
    nodes = nodes[nodes.lineNumber != '']
    
    if save:
        nodes.to_csv(os.path.join(savedir, f'{filename}_nodes.csv'), index=False)
        merge.to_csv(os.path.join(savedir, f'{filename}_edges.csv'), index=False)
    return nodes, merge


def generate_pfg(filepath, save_csv=False):
    """generate PFG(Program Flow Graph) given filepath containing `nodes.json` and `edges.json`  
    # * We define program flow graph as ⬇:
    # * control flow + control dependence + data dependence + function call
    # * then we need to convert control flow and control dependence into a uniform control dependence
    # * so as to generate program dependency graph with calls (PDGc)
    
    Joern edge types:
    1. CDG: control dependency
    2. REACHING_DEF: data dependency
    3. REF: data dependency
    4. CFG: control flow (can be transformed into control dependency)
    5. CALL: function call
    
    Args:
        filepath (str): containing `run_joern` JSON results

    Returns:
        G: networkx.MultiDiGraph
    """
    
    filter_edges = ['CDG', 'CFG', 'REACHING_DEF', 'REF', 'CALL']
    node_df, edges = get_node_edges(filepath, save_csv)
    
    edges = edges[edges.etype.isin(filter_edges)]
    
    G = nx.MultiDiGraph()
    line_list = sorted(list(set(list(edges.src_lineNumber.unique()) + list(edges.dst_lineNumber.unique()))))
    line_list = list(map(lambda x: int(x), line_list))
    drawn_edges = []
    
    with open(filepath, 'r', errors='ignore') as f:
        raw_code = f.readlines()
        raw_code = list(map(lambda x: x.strip(), raw_code))
        
    for edge_type in filter_edges:
        edges_local = edges[edges.etype == edge_type]
        for _, row in edges_local.iterrows():
            # ignore edges between same line nodes
            if row.src_lineNumber == row.dst_lineNumber: continue
            
            # * For CFG, CDG, DDG, CALL. Joern generate inverse directions
            if edge_type in ['CDG', 'CFG', 'REACHING_DEF', 'CALL']:
                src, dst = int(row.dst_lineNumber), int(row.src_lineNumber)
                src_code, dst_code = raw_code[dst-1], raw_code[src-1]
                # src_code, dst_code = row.dst_code, row.src_code
            else: # REF
                src, dst = int(row.src_lineNumber), int(row.dst_lineNumber)
                src_code, dst_code = raw_code[src-1], raw_code[dst-1]
                # src_code, dst_code = row.src_code, row.dst_code
                
            if edge_type in ['CFG', 'CDG', 'REACHING_DEF'] and src > dst: continue
            if edge_type == 'REACHING_DEF' and row.dataflow == '': continue
            
            etype = 'REACHING_DEF' if edge_type == 'REF' else edge_type
            if (src, dst, etype) in drawn_edges: continue # already added
            
            G.add_edge(src, dst, etype=etype)
            G.nodes[src]['code'] = src_code
            G.nodes[dst]['code'] = dst_code
            
            drawn_edges.append((src, dst, etype))
    
    return G, node_df


def generate_cpg(filepath, save_csv=False):
    """generate CPG(Code Property Graph) given filepath containing `nodes.json` and `edges.json`
    
    Joern edge types:
    1. CDG
    2. REACHING_DEF
    3. REF
    4. CFG
    5. AST
    
    Args:
        filepath (str): containing `run_joern` JSON results 
        save_csv (bool, optional): whether to save node and edge csv. Defaults to False.
    
    Returns:
        G: networkx.MultiDiGraph
    """
    
    filter_edges = ['CDG', 'CFG', 'REACHING_DEF', 'REF', 'AST']
    node_df, edges = get_node_edges(filepath, save_csv)

    G = nx.MultiDiGraph()
    edges = edges[edges.etype.isin(filter_edges)]
    drawn_edges = []
    
    for _, row in edges.iterrows():
        edge_type = row.etype
        if edge_type in ['CDG', 'CFG', 'REACHING_DEF']:
            src, dst = row.dst_node, row.src_node
            src_code, dst_code = row.dst_code, row.src_code
        else:
            src, dst = row.src_node, row.dst_node
            src_code, dst_code = row.src_code, row.dst_code
            
        if edge_type == 'REACHING_DEF' and row.dataflow == '': continue
        etype = 'REACHING_DEF' if edge_type == 'REF' else edge_type
        if (src, dst, etype) in drawn_edges: continue
        
        G.add_edge(src, dst, etype=etype)
        G.nodes[src]['code'] = src_code
        G.nodes[dst]['code'] = dst_code
        
        drawn_edges.append((src, dst, etype))

    return G, node_df

def generate_pdgc(filepath, save_csv=False):
    """generate PDGc given filepath containing `nodes.json` and `edges.json`
    # * We define program dependency graph with function call (PDGc) as ⬇：
    # * control dependence (including user-defined function) + data dependence + function calls
    
    hint: generate pfg first, then convert Joern CF and CG into final control dependency edges
    Args:
        filepath (str): containing `run_joern` JSON results
    """
    
    pfg, node_df = generate_pfg(filepath, save_csv)
        
    # extract CFG only
    cfg, edge_dict = nx.DiGraph(), {}
    for e, datadict in pfg.edges.items():
        if datadict['etype'] == 'CFG':
            cfg.add_edge(e[0], e[1])
            if e[0] not in edge_dict:
                edge_dict[e[0]] = [e[1]]
            else:
                edge_dict[e[0]].append(e[1])
    
    # find root nodes and branch nodes
    branch_nodes = []
    for n in cfg.nodes:
        if len(cfg.in_edges(n)) == 0 or len(cfg.out_edges(n)) > 1:
            branch_nodes.append(n)
    
    # util function to find all nodes in the same branch
    def _get_all_edges(node, new_edges):
        for n in edge_dict[node]:
            new_edges.append(n)
            if n not in edge_dict:
                continue
            elif n in branch_nodes:
                continue
            else:
                _get_all_edges(n, new_edges)
    
    # convert cfg into control dependence graph
    converted_dict = {}
    for node in branch_nodes:
        new_edges = []
        _get_all_edges(node, new_edges)
        converted_dict[node] = new_edges
    
    # merge cfg->cdg into original pfg(without cfg)
    # treat both cfg->cdg and original cdg as control dependence
    pdgc, added_edges = nx.MultiDiGraph(), []    

    ## add original pfg first (CDG, DDG, CALL)
    for e, datadict in pfg.edges.items():
        if datadict['etype'] == 'CFG':
            continue
        if (e[0], e[1], datadict['etype']) in added_edges:
            continue
        pdgc.add_edge(e[0], e[1], etype=datadict['etype'])
        pdgc.nodes[e[0]]['code'] = pfg.nodes[e[0]]['code']
        pdgc.nodes[e[1]]['code'] = pfg.nodes[e[1]]['code']
        added_edges.append((e[0], e[1], datadict['etype']))
    
    ## then add cfg->cdg
    for src_node, dst_nodes in converted_dict.items():
        for dst_node in dst_nodes:
            if (src_node, dst_node, 'CDG') in added_edges:
                continue
            pdgc.add_edge(src_node, dst_node, etype='CDG')
            pdgc.nodes[src_node]['code'] = pfg.nodes[src_node]['code']
            pdgc.nodes[dst_node]['code'] = pfg.nodes[dst_node]['code']
            added_edges.append((src_node, dst_node, 'CDG'))

    ## * ATTENTION: theoretically, every node must has a CDG edge
    ## find those nodes without CDG edges, and connected to the former closest branch node
    for node in pdgc.nodes:
        only_ddg = True
        for e in pdgc.in_edges(node):
            edges = pdgc.get_edge_data(*e)
            for _, v in edges.items():
                if v['etype'] != 'REACHING_DEF':
                    only_ddg = False
        for e in pdgc.out_edges(node):
            edges = pdgc.get_edge_data(*e)
            for _, v in edges.items():
                if v['etype'] != 'REACHING_DEF':
                    only_ddg = False
        
        ## add a CDG edge to the node
        if only_ddg:
            closest_branch = None
            
            ## init closest
            for branch in branch_nodes:
                if branch < node:
                    closest_branch = branch
                    break
                
            if closest_branch is not None:
                for branch in branch_nodes:
                    if branch < node and branch > closest_branch:
                        closest_branch = branch
                
                pdgc.add_edge(closest_branch, node, etype='CDG')

    return pdgc, node_df  
    
    
if __name__ == '__main__':
    fp = 'test/main.c'
    run_joern(filepath=fp)
    pdgc, nodes = generate_pdgc(fp, save_csv=False)
