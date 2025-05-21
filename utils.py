import os
import yaml
import torch
import torch.nn as nn


def load_config(path='config.yaml'):
    config_dir = os.path.dirname(os.path.realpath(__file__)) 
    config_path = os.path.join(config_dir, path) 
    
    if not os.path.exists(config_path): 
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_subconfig(section, path='config.yaml'):
    config = load_config(path)
    return config.get(section, {})


def xavier_init(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight)
        

def kaiming_init(model):
    if isinstance(model, nn.Linear):
        nn.init.kaiming_uniform_(model.weight)  


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
