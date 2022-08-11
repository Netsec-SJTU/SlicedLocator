# @Time: 2022.6.3 21:58
# @Author: Bolun Wu

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GlobalAttention, global_mean_pool
from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn.models import MLP
from torch_geometric.utils import to_dense_batch

from metrics import CoarsegrainedMetric, FinegrainedMetric


class BaseModule(pl.LightningModule):
    def __init__(self, lr=1e-4, milestones=None):
        super(BaseModule, self).__init__()
        self.lr = lr
        self.milestones = milestones
    
    def forward(self):
        raise NotImplementedError
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = []
        if self.milestones is not None:
            assert isinstance(self.milestones, list)
            lr_sche = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones)
            scheduler.append(lr_sche)
        return [optimizer], scheduler
    
    def log_epoch(self, name, value, prog_bar, batch_size):
        self.log(name, value, on_step=False, on_epoch=True, prog_bar=prog_bar, batch_size=batch_size)

    
class BaseDualModule(BaseModule):
    def __init__(self,
                 coarse_ce_weight=[0.5, 0.5],
                 fine_ce_weight=[0.05, 0.95],
                 loss_weight=[0.5, 0.5],
                 *args, **kw):
        super(BaseDualModule, self).__init__(*args, **kw)
        self.coarse_ce_weight = coarse_ce_weight
        self.fine_ce_weight = fine_ce_weight
        self.loss_weight = loss_weight
        self.coarse_loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(coarse_ce_weight))
        self.fine_loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(fine_ce_weight))
        self.coarse_metric_fn = CoarsegrainedMetric(num_classes=2)
        self.fine_metric_fn = FinegrainedMetric(num_classes=2)
    
    def training_step(self, batch, *args):
        n_out, g_out = self(batch)
        # loss
        coarse_loss = self.coarse_loss_fn(g_out, batch.g_y)
        fine_loss = self.fine_loss_fn(n_out, batch.n_y)
        loss = self.loss_weight[0] * coarse_loss + self.loss_weight[1] * fine_loss
        # metric
        coarse_metrics = self.coarse_metric_fn(g_out, batch.g_y)
        vul_n_size, fine_metrics = self.fine_metric_fn(n_out, batch.n_y, batch.g_y, batch.batch)
        # logger
        self.log_epoch('train/loss', loss, True, g_out.size(0))
        for m in coarse_metrics: self.log_epoch(f'train/{m.name}', m.value, False, g_out.size(0))
        for m in fine_metrics: self.log_epoch(f'train/{m.name}', m.value, False, vul_n_size)
        return {'loss': loss}
    
    def validation_step(self, batch, *args):
        n_out, g_out = self(batch)
        # loss
        coarse_loss = self.coarse_loss_fn(g_out, batch.g_y)
        fine_loss = self.fine_loss_fn(n_out, batch.n_y)
        loss = self.loss_weight[0] * coarse_loss + self.loss_weight[1] * fine_loss
        # metric
        coarse_metrics = self.coarse_metric_fn(g_out, batch.g_y)
        vul_n_size, fine_metrics = self.fine_metric_fn(n_out, batch.n_y, batch.g_y, batch.batch)
        # logger
        self.log_epoch('val/loss', loss, True, g_out.size(0))
        for m in coarse_metrics: self.log_epoch(f'val/{m.name}', m.value, m.prog_bar, g_out.size(0))
        for m in fine_metrics: self.log_epoch(f'val/{m.name}', m.value, m.prog_bar, vul_n_size)
    
    def test_step(self, batch, *args):
        n_out, g_out = self(batch)
        # loss
        coarse_loss = self.coarse_loss_fn(g_out, batch.g_y)
        fine_loss = self.fine_loss_fn(n_out, batch.n_y)
        loss = self.loss_weight[0] * coarse_loss + self.loss_weight[1] * fine_loss
        # metric
        coarse_metrics = self.coarse_metric_fn(g_out, batch.g_y)
        vul_n_size, fine_metrics = self.fine_metric_fn(n_out, batch.n_y, batch.g_y, batch.batch)
        # logger
        self.log_epoch('test/loss', loss, True, g_out.size(0))
        for m in coarse_metrics: self.log_epoch(f'test/{m.name}', m.value, True, g_out.size(0))
        for m in fine_metrics: self.log_epoch(f'test/{m.name}', m.value, True, vul_n_size)


class VulDnlDual(BaseDualModule):
    def __init__(self,
                 embedding_mode: str,
                 embedding_dim: int,
                 hidden_dim: int,
                 conv_layers: int,
                 lstm_layers: int,
                 vocab_size: int=0,
                 token_type_size: int=0,
                 max_code_size: int=0,
                 num_n_class: int=2,
                 num_g_class: int=2,
                 dropout: float=0.5,
                 pool_type: str='attention',
                 kernel_type: str='gin',
                 *args, **kw):
        super(VulDnlDual, self).__init__(*args, **kw)
        assert embedding_mode == 'pretrained' or 'token' in embedding_mode
        
        self.embedding_mode = embedding_mode
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.conv_layers = conv_layers
        self.lstm_layers = lstm_layers
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_code_size = max_code_size
        
        self.num_n_class = num_n_class
        self.num_g_class = num_g_class
        
        self.dropout = dropout
        self.pool_type = pool_type
        self.kernel_type = kernel_type
        
        if embedding_mode != 'pretrained':
            self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
            if 'pos' in self.embedding_mode:
                self.register_buffer('position_ids', torch.arange(max_code_size, dtype=torch.long).expand(1, -1))
                self.pos_encoding = nn.Embedding(max_code_size, embedding_dim)
            if 'type' in self.embedding_mode:
                self.token_type_embedding = nn.Embedding(token_type_size, embedding_dim)
            if 'sta' in self.embedding_mode:
                self.statistic_proj = nn.Linear(token_type_size-1, hidden_dim)
            self.layer_norm = nn.LayerNorm(embedding_dim)
            self.scoring_fc = nn.Linear(embedding_dim, 1)

        self.proj = nn.Linear(embedding_dim, hidden_dim)
        self.edge_embedding = nn.Embedding(3, hidden_dim)
        
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True, dropout=self.dropout)
        self.convs = nn.ModuleList()
        for _ in range(conv_layers):
            self.convs.append(self.init_conv(hidden_dim, hidden_dim))
        
        lin_in_channels = hidden_dim * 2

        if self.pool_type == 'attention':
            self.pooling = GlobalAttention(nn.Linear(lin_in_channels, 1))
                
        self.node_fc = nn.Linear(lin_in_channels * 2, num_n_class)
        self.graph_fc = nn.Linear(lin_in_channels, num_g_class)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        ## * node projection
        if self.embedding_mode != 'pretrained':
            mask = data.mask.bool().unsqueeze(-1)
            
            x_ = self.token_embedding(x)
            if 'pos' in self.embedding_mode:
                pos = self.position_ids.expand(x.size(0), -1) 
                x_ += self.pos_encoding(pos)
            if 'type' in self.embedding_mode:
                x_ += self.token_type_embedding(data.x_type)
            
            x_ = self.layer_norm(x_)
            x_ = F.dropout(x_, p=self.dropout, training=self.training)
            
            att = F.leaky_relu(self.scoring_fc(x_))
            att = att.masked_fill_(~mask, float('-inf'))
            att = torch.softmax(att, dim=1)
            
            x = torch.matmul(att.transpose(1, 2), x_).squeeze()
            
            if 'sta' in self.embedding_mode:
                x += self.statistic_proj(data.x_type_statistic)
                
            x = self.proj(x)       
        
        else:
            
            x = self.proj(x)

        ## * lstm
        x_dense, dense_mask = to_dense_batch(x, data.batch)
        x_dense, _ = self.lstm(x_dense)
        x_lstm = x_dense[dense_mask]
        
        ## * gnn
        edge_attr = self.edge_embedding(edge_attr)
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index, edge_attr) + x
        
        ## * concatenate lstm and gnn output
        x = torch.cat([x_lstm, x], dim=-1)
        
        ## * graph representation
        if self.pool_type == 'attention':
            g_x = self.pooling(x, data.batch)
        elif self.pool_type == 'mean':
            g_x = global_mean_pool(x, data.batch)
        
        ## * node representation
        n_x = torch.cat([x, g_x[data.batch]], dim=-1)
        
        ## * classification
        n_x = self.node_fc(n_x)
        g_x = self.graph_fc(g_x)
        
        return n_x, g_x
    
    def init_conv(self, in_channels: int, out_channels: int):
        if self.kernel_type == 'gin':
            mlp = MLP([in_channels, in_channels, out_channels],
                    batch_norm=True, dropout=self.dropout)
            return GINEConv(mlp)
        else:
            raise Exception('Unknown kernel type')
    
    def get_code_embeddings(self, x, mask, x_type, x_type_statistic):       
        ## * node projection
        if self.embedding_mode != 'pretrained':
            mask = mask.bool().unsqueeze(-1)
            
            x_ = self.token_embedding(x)
            if 'pos' in self.embedding_mode:
                pos = self.position_ids.expand(x.size(0), -1) 
                x_ += self.pos_encoding(pos)
            if 'type' in self.embedding_mode:
                x_ += self.token_type_embedding(x_type)
            
            x_ = self.layer_norm(x_)
            x_ = F.dropout(x_, p=self.dropout, training=self.training)
            
            att = F.leaky_relu(self.scoring_fc(x_))
            att = att.masked_fill_(~mask, float('-inf'))
            att = torch.softmax(att, dim=1)
            
            x = torch.matmul(att.transpose(1, 2), x_).squeeze()
            
            if 'sta' in self.embedding_mode:
                x += self.statistic_proj(x_type_statistic)
                
            x = self.proj(x)       
        
        else:
            
            x = self.proj(x)
        return x


def vuldnl_model_selector(config):
    if config['model_name'].lower() == 'vuldnldual':
        model = VulDnlDual(
            embedding_mode=config['embedding_mode'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            conv_layers=config['conv_layers'],
            lstm_layers=config['lstm_layers'],
            vocab_size=config['vocab_size'],
            token_type_size=config['token_type_size'],
            max_code_size=config['max_code_size'],
            pool_type=config['pool_type'],
            kernel_type=config['kernel_type'],
            lr=config['lr'], milestones=config['milestones'],
            fine_ce_weight=config['fine_ce_weight']
            )
    else:
        raise Exception('Model name not found')
    
    print(model)
    return model
