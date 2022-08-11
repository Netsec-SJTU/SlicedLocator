# @Time: 2022.6.11 18:50
# @Author: Bolun Wu
# ref: https://github.com/DeepWukong/Dataset

import json
import logging
import os
import shutil
import ssl
import urllib.request
import zipfile

import requests
from tqdm.contrib.concurrent import process_map

helpers_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(helpers_dir)
logs_dir = os.path.join(root_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)


logging.basicConfig(filename=os.path.join(logs_dir, 'spider.log'),
                    filemode='a',
                    level=logging.NOTSET,
                    format='%(asctime)s - %(levelname)s - %(process)d - %(funcName)s - %(message)s')


def extract_zip(path: str, folder: str):
    r"""Extracts a zip archive to a specific folder.
    Ref: https://pytorch-geometric.readthedocs.io/en/latest/_modules/index.html
    
    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
    """
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)


def download_url(url, folder, filename):
    """Downloads the content of an URL to a specific folder.
    Ref: https://pytorch-geometric.readthedocs.io/en/latest/_modules/index.html
    
    Args:
        url (string): The url.
        folder (string): The folder.
    """
    os.makedirs(folder, exist_ok=True)
    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]

    path = os.path.join(folder, filename)
    if os.path.exists(path):
        return

    # print(f'Downloading {url}')
    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


def download_sard_testcase(merge_args):
    root, testcase_id = merge_args
    api = 'https://samate.nist.gov/SARD/downloads/versions/{}-v1.0.0.zip'
    api_v2 = 'https://samate.nist.gov/SARD/downloads/versions/{}-v2.0.0.zip'
    url = api.format(testcase_id)
    
    testcase_path = os.path.join(root, testcase_id)
    os.makedirs(testcase_path, exist_ok=True)

    # download
    try:
        # download
        download_url(url=url, folder=testcase_path, filename=f'{testcase_id}.zip')

        # unzip
        extract_zip(path=os.path.join(testcase_path, f'{testcase_id}.zip'), folder=testcase_path)

        # check status
        deprecated = False
        with open(os.path.join(testcase_path, 'manifest.sarif'), 'r') as f:
            manifest = json.load(f)
            if manifest['runs'][0]['properties']['status'] == 'deprecated':
                deprecated = True

        # if deprecated, download v2.0.0
        if deprecated:
            shutil.rmtree(testcase_path)
            os.makedirs(testcase_path, exist_ok=True)
            
            url = api_v2.format(testcase_id)
            download_url(url=url, folder=testcase_path, filename=f'{testcase_id}.zip')
            extract_zip(path=os.path.join(testcase_path, f'{testcase_id}.zip'), folder=testcase_path)
            logging.info(f'{url}: download v2.0.0 for deprecated')
        else:
            logging.info(f'{url}: success')
            
    except Exception as e:
        logging.info(f'{url}: fail {str(e)}')


def get_testcases(root='/home/wubolun/data/codevul/DeepWukong/Dataset',
                  save_dir='/home/wubolun/data/codevul/SARD'):
    test_cases = set()
    
    # * get more good samples
    good_records = []
    good_url = 'https://samate.nist.gov/SARD/api/test-cases/search?language%5B%5D=c&state%5B%5D=good&page={}&limit=500'
    res = requests.get(good_url.format(1))
    res = res.json()
    for i in range(1, res['pageCount']+1):
        res = requests.get(good_url.format(i))
        res = res.json()
        good_records.extend(res['testCases'])
    good_testcases = list(map(lambda x: x['download'].split('/')[-1].split('-')[0], good_records))
    for good_testcase in good_testcases:
        test_cases.add(good_testcase)
        
    # * get samples from DeepWukong
    for filename in os.listdir(root):
        filepath = os.path.join(root, filename)
        if not os.path.isdir(filepath):
            continue
        json_path = os.path.join(filepath, 'xfg-sym-unique/bigJson.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
            for item in data:
                test_cases.add(item['testcaseID'])

    test_cases = list(test_cases)
    
    # * remove deprecated samples
    # * e.g. https://samate.nist.gov/SARD/test-cases/2082/versions/1.0.0
    test_cases.remove('2082')
    test_cases.remove('149047')
    test_cases.remove('2081')
    
    print(len(test_cases))
    with open(os.path.join(save_dir, 'test_case.json'), 'w') as f:
        json.dump(test_cases, f, indent=1)


if __name__ == '__main__':
    
    # get_testcases()
    
    with open('/home/wubolun/data/codevul/SARD/test_case.json', 'r') as f:
        data = json.load(f)

    root = '/home/wubolun/data/codevul/SARD/c'
    merge_args = []

    for testcase_id in data:
        merge_args.append([root, testcase_id])

    process_map(download_sard_testcase, merge_args, max_workers=8)
    