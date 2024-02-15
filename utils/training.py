from typing import Callable, Optional, List, Tuple, Union
import os
import json
from copy import deepcopy

import torch
from torch_geometric.loader import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch.nn as nn
from tqdm import tqdm
import wandb

from utils.custom_loss_functions import MaskedL2Eval, PowerImbalanceV2, MixedMSEPoweImbalanceV2, get_mask_from_bus_type


def append_to_json(log_path, run_id, result):
    log_entry = {str(run_id): result}

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.isfile(log_path)
    try:
        with open(log_path, "r") as json_file:
            exist_log = json.load(json_file)
    except FileNotFoundError:
        exist_log = {}
    with open(log_path, "w") as json_file:
        exist_log.update(log_entry)
        json.dump(exist_log, json_file, indent=4)


def train_epoch(
    accelerator,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    optimizer: Optimizer,
    device: torch.device,
    total_length: int = 100000,
    batch_size: int = 128,
    log_to_wandb: bool = False,
    epoch: int = 0,
    train_step: int = 0,
    inverse_transforms: dict[str, Callable]={'node':{}, 'edge':{}},
) -> float:
    """
    Trains a neural network model for one epoch using the specified data loader and optimizer.

    Args:
        model (nn.Module): The neural network model to be trained.
        loader (DataLoader): The PyTorch Geometric DataLoader containing the training data. * Or another DataLoader
        optimizer (torch.optim.Optimizer): The PyTorch optimizer used for training the model.
        device (str): The device used for training the model (default: 'cpu').

    Returns:
        float: The mean loss value over all the batches in the DataLoader.

    """
    model = model.to(device)
    _sum_weight = 0
    model.train()
    train_losses = {
        'MaskedL2': {},
        'MaskedL2Split': {},
        'PowerImbalance': {},
        'MSE': {},
    }
    masked_l2_eval = MaskedL2Eval(normalize=False, split_real_imag=False, pre_transforms=inverse_transforms)
    masked_l2_split_eval = MaskedL2Eval(normalize=False, split_real_imag=True, pre_transforms=inverse_transforms)
    with tqdm(initial=1, total=total_length+1) as pbar:
        for data in loader:
            data = data.to(device)
            _weight = data.x.shape[0] # actually = batch_size * num_nodes per sample. used as average weights over batches
            with accelerator.autocast():
                out = model(data)   # (N, 4)
                                    # data.y.shape == (N, 4)
                is_to_pred = get_mask_from_bus_type(data.bus_type) # 0, 1 mask of (N, 4). 1 is need to predict
                if isinstance(loss_fn, MixedMSEPoweImbalanceV2):
                    mixed_loss_terms = loss_fn(out, data.edge_index, data.edge_attr, data.y)
                    loss = mixed_loss_terms['loss']
                    # mini eval
                    #   mixed part, mse + phys
                    train_losses['MSE']['total'] = train_losses['MSE'].get('total', 0.) + mixed_loss_terms['mse'].mean().item() * _weight
                    train_losses['PowerImbalance']['total'] = train_losses['PowerImbalance'].get('total', 0.) + mixed_loss_terms['physical'].mean().item() * _weight
                    #   masked l2 part
                    with torch.no_grad():
                        masked_l2_loss_terms = masked_l2_eval(out, data.y, is_to_pred)
                        masked_l2_split_loss_terms = masked_l2_split_eval(out, data.y, is_to_pred)
                    for term_name, value in masked_l2_loss_terms.items():
                        train_losses['MaskedL2'][term_name] = train_losses['MaskedL2'].get(term_name, 0.) + value.mean().item() * _weight
                    for term_name, value in masked_l2_split_loss_terms.items():
                        train_losses['MaskedL2Split'][term_name] = train_losses['MaskedL2Split'].get(term_name, 0.) + value.mean().item() * _weight
                    # log mini eval results
                    if log_to_wandb:
                        wandb.log({
                            'Train': {
                                'Loss': loss.item(),
                                'MaskedL2': {k: v.mean().item() for k, v in masked_l2_loss_terms.items()},
                                'MaskedL2Split': {k: v.mean().item() for k, v in masked_l2_split_loss_terms.items()},
                                'MSE': mixed_loss_terms['mse'].mean().item(),
                                'PowerImbalance': mixed_loss_terms['physical'].mean().item(),
                            },
                            'Epoch': epoch,
                            'Train Step': train_step
                        }, step=train_step)
                else:
                    raise ValueError(f'you shouldn\'t use this loss function {type(loss_fn)}!')
            
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.)
            
            accelerator.wait_for_everyone()
            optimizer.step()
            optimizer.zero_grad()
            accelerator.wait_for_everyone()
                
            _sum_weight += _weight
            train_step += 1
            pbar.set_description(f'train loss: {loss.item():.4f}')
            pbar.update(1)
    
    for func_name, terms in train_losses.items():
        for term_name, term_value in terms.items():
            train_losses[func_name][term_name] = term_value / _sum_weight

    return train_losses, train_step


def main():
    log_path = 'logs/save_logs.json'
    run_id = 'arb_id_01'
    result = {
        'train_loss': 0.3,
        'val_loss': 0.2,
    }
    append_to_json(log_path, run_id, result)


if __name__ == '__main__':
    main()
