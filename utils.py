# @Time: 2022.6.2 22:32
# @Author: Bolun Wu

import os

def root_dir():
    return os.path.dirname(__file__)

def helpers_dir():
    return os.path.join(root_dir(), 'helpers')

def res_dir():
    return os.path.join(root_dir(), 'res')

os.makedirs(res_dir(), exist_ok=True)

def count_code_lines_of_this_dir():
    count = 0
    for root, dirs, files in os.walk(root_dir()):
        for file in files:
            if file.endswith('.py') or file.endswith('.scala') or file.endswith('.sh'):
                with open(os.path.join(root, file), 'r') as f:
                    count += len(f.readlines())
    return count


if __name__ == '__main__':
    total_lines = count_code_lines_of_this_dir()
    print('Total lines: {}'.format(total_lines))

    