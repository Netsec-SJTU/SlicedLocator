# @Time: 2022/3/9 10:54
# @Author: Bolun Wu

import os
import subprocess

from tree_sitter import Language, Parser

current_dir = os.path.dirname(__file__)

class TreeSitterParser(object):
    """This class extracts tokens from a piece of C code
       based on Tree-sitter (https://tree-sitter.github.io/tree-sitter/)

    """
    def __init__(self, lib_path='./c.so'):
        self.enc = 'utf-8'
        self.languages = {}
        self.build_library(lib_path)
        
    def build_library(self, lib_path):
        """build Tree-sitter .so library

        Args:
            name (str): file name of the library (.so)
        """
        if not os.path.exists(lib_path):
            if not os.path.exists(os.path.join(current_dir, 'tree-sitter-c')):
                subprocess.run(['git', 'clone', 'git@github.com:tree-sitter/tree-sitter-c.git', os.path.join(current_dir, 'tree-sitter-c')])
            Language.build_library(lib_path, [os.path.join(current_dir, 'tree-sitter-c')])
        
        self.languages['c'] = Language(lib_path, 'c')
    
    def parse(self, source, mode='source'):
        """parser source code using Tree-sitter

        Args:
            source (_type_): _description_
        """
        # c path
        if mode != 'source' and os.path.exists(source):
            with open(source, 'r', errors='ignore') as f:
                self.code = f.read()
        # code
        elif isinstance(source, str):
            self.code = source
        else:
            raise Exception('source cannot be parsed.')
        
        self.parser = Parser()
        self.parser.set_language(self.languages['c'])
        self.tree = self.parser.parse(bytes(self.code, self.enc))
    
    # EXTERNAL
    def get_tokens(self):
        tokens, _ = self.__get_token_and_index_by_node(self.tree.root_node)
        return tokens

    def get_tokens_and_indices(self):
        tokens, indices = self.__get_token_and_index_by_node(self.tree.root_node)
        return tokens, indices

    def get_tokens_types_and_indices(self):
        tokens, types, indices = self.__get_token_type_index_by_node(self.tree.root_node)
        return tokens, types, indices

    def get_clean_tokens_types(self):
        _tokens, _types, _ = self.__get_token_type_index_by_node(self.tree.root_node)
        tokens, types = [], []
        for i in range(len(_tokens)):
            if _tokens[i] == '': continue
            tokens.append(_tokens[i])
            types.append(_types[i])
        return tokens, types
    
    # EXTERNAL
    def get_tokens_by_funcnames(self, func_names):
        if isinstance(func_names, str):
            func_names = [func_names]
        ret_dict = {}
        functions = self.__get_function_nodes()
        for func in functions:
            def_node, dec_node = func
            func_name = self.__index_to_token((dec_node.start_point, dec_node.end_point))
            func_name = func_name[:func_name.find('(')]
            if func_name in func_names:
                tokens, _ = self.__get_token_and_index_by_node(def_node)
                ret_dict[func_name] = tokens
        return ret_dict
    
    def get_tokens_and_indices_by_funcnames(self, func_names):
        if isinstance(func_names, str):
            func_names = [func_names]
        ret_dict = {}
        functions = self.__get_function_nodes()
        for func in functions:
            def_node, dec_node = func
            func_name = self.__index_to_token((dec_node.start_point, dec_node.end_point))
            func_name = func_name[:func_name.find('(')]
            if func_name in func_names:
                tokens, indices = self.__get_token_and_index_by_node(def_node)
                ret_dict[func_name] = [tokens, indices]
        return ret_dict
    
    def __get_index_traverse(self, root):
        if self.__is_leaf(root):
            return [(root.start_point, root.end_point)]
        else:
            indexes = []
            for child in root.children:
                indexes.extend(self.__get_index_traverse(child))
            return indexes
        
    def __get_type_traverse(self, root):
        if self.__is_leaf(root):
            return [root.type]
        else:
            indexes = []
            for child in root.children:
                indexes.extend(self.__get_type_traverse(child))
            return indexes
        
    def __get_token_and_index_by_node(self, root):
        tokens_index = self.__get_index_traverse(root)
        tokens = [self.__index_to_token(index) for index in tokens_index]
        return tokens, tokens_index                

    def __get_token_type_index_by_node(self, root):
        tokens_index = self.__get_index_traverse(root)
        tokens_type = self.__get_type_traverse(root)
        tokens = [self.__index_to_token(index) for index in tokens_index]
        return tokens, tokens_type, tokens_index   
    
    def __get_node_traverse(self, root, node_list: list):
        node_list.append(root)
        if self.__is_leaf(root):
            return
        else:
            for child in root.children:
                self.__get_node_traverse(child, node_list)
    
    # EXTERNAL
    def node_dft(self, root):
        node_list = []
        self.__get_node_traverse(root, node_list)
        return node_list
    
    def __get_function_nodes(self):
        funcs = []
        node_list = self.node_dft(self.tree.root_node)
        for i, node in enumerate(node_list):
            if node.type == 'function_definition':
                def_node, dec_node = node, None
                for child in def_node.children:
                    if child.type == 'function_declarator':
                        dec_node = child
                        break
                if dec_node:
                    funcs.append((def_node, dec_node))
        return funcs
        
    def __index_to_token(self, index):
        """get token at index(start_point, end_point)
        start_point and end_point are positions of a character
        position (x,y) means the char is at yth position in line x
        Args:
            index (tuple): (start_point, end_point)

        Returns:
            str: a code token
        """
        code = self.code.split('\n')
        start_point, end_point = index
        if start_point[0] == end_point[0]:
            token = code[start_point[0]][start_point[1]:end_point[1]]
        else:
            token = ''
            token += code[start_point[0]][start_point[1]:]
            for i in range(start_point[0]+1, end_point[0]):
                token += code[i]
            token += code[end_point[0]][:end_point[1]]
        return token

    def __is_leaf(self, node):
        # code = self._index_to_token((node.start_point, node.end_point))
        # print(f'type: {node.type}, child: {len(node.children)}, code: {code}')
        return (len(node.children) == 0 or node.type == 'string_literal') and node.type != 'comment'
    
    def print_tree(self):
        """AST tree visualization in command line by DFS
        """
        def _print_tree_traversal(root_node, indent=' '*2):
            node_type = root_node.type
            index = (root_node.start_point, root_node.end_point)
            # print(f'{indent}{node_type}:{self.__index_to_token(index)}')
            print(f'{indent}{root_node}')
            if self.__is_leaf(root_node):
                return
            for child in root_node.children:
                _print_tree_traversal(child, indent+' '*2)
        root_node = self.tree.root_node
        _print_tree_traversal(root_node)

