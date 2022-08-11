# @Time: 2022.6.4 16:44
# @Author: Bolun Wu

import argparse
import json
import os
import shutil
import sys

from utils import *

sys.path.append(os.path.join(root_dir(), 'models'))
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader

from dataset.sdg_graph import SDGGraphDatasetFullFeature
from models.gnn import vuldnl_model_selector

seed = 42
pl.seed_everything(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='path to config file')
    args = parser.parse_args()
    
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    
    if config['cv']: ks = list(range(5))
    else: ks = [0]
    
    for k in ks:
        
        save_dir = os.path.join(res_dir(), config['data_name'], config['model_name'])
        if os.path.exists(os.path.join(save_dir, config['run_name'], f'fold_{k}', 'best_val_result.json')):
            continue

        # dataset
        print(f'Start training fold {k}...')     
        dataset = SDGGraphDatasetFullFeature(
            json_path=config['clean_json_path'],
            vocab_path=config['vocab_path'],
            anno_path=os.path.join(config['annotation_dir'], f'fold_{k}.json'),
            fold=k
        )
        config['vocab_size'] = len(dataset.vocab_meta['vocab'])
        config['token_type_size'] = len(dataset.vocab_meta['token_type_set'])
        config['max_code_size'] = dataset.vocab_meta['max_code_size']

        train_idx, val_idx = dataset.get_annotation()
        train_set, val_set = dataset[train_idx], dataset[val_idx]
        train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
        val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
        print(f'Train size: {len(train_set)}. Validation size: {len(val_set)}')
        
        # model save dir
        model = vuldnl_model_selector(config)
        checkpoint_callback = ModelCheckpoint(monitor=config['monitor'], mode='max',
                                            save_weights_only=True, filename='{epoch}_{step}')
        lr_monitor = LearningRateMonitor(logging_interval='step')
        logger = TensorBoardLogger(save_dir=save_dir, name=config['run_name'], version=f'fold_{k}')
        
        trainer = pl.Trainer(max_epochs=config['epoch'],
                             accelerator='gpu', gpus=[0],
                             log_every_n_steps=50, logger=logger,
                             callbacks=[checkpoint_callback, lr_monitor])
        trainer.fit(model, train_loader, val_loader)
        
        val_result = trainer.test(model, val_loader, ckpt_path='best')[0]
        with open(os.path.join(save_dir, config['run_name'], f'fold_{k}', 'best_val_result.json'), 'w') as f:
            json.dump(val_result, f, indent=1)
        
        del dataset
        del train_set, val_set
        del train_loader, val_loader
        del model, trainer
    
    shutil.copyfile(args.config_path, os.path.join(save_dir, config['run_name'], 'config.yaml'))
    with open(os.path.join(save_dir, config['run_name'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=1)

