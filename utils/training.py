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

from utils.custom_loss_functions import Masked_L2_loss, PowerImbalanceV2, MixedMSEPoweImbalanceV2, get_mask_from_bus_type


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
    num_samples = 0
    model.train()
    train_losses = {
        'MaskedL2': {},
        'PowerImbalance': {},
        'MSE': {},
    }
    with tqdm(initial=1, total=785) as pbar:
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)   # (N, 4)
                                # data.y.shape == (N, 4)
            
            is_to_pred = get_mask_from_bus_type(data.bus_type) # 0, 1 mask of (N, 4). 1 is need to predict
            if isinstance(loss_fn, Masked_L2_loss):
                loss_terms = loss_fn(out, data.y, is_to_pred)
                for k, v in loss_terms.items():
                    train_losses['MaskedL2'][k] = train_losses['MaskedL2'].get(k, 0.) + v.mean().item() * batch_size
                loss = loss_terms['total']
                if log_to_wandb:
                    wandb.log({
                        'Train': {
                            'train_loss': loss.item(),
                            'MaskedL2': {
                                k: v.mean().item() for k, v in loss_terms.items()
                            }
                        },
                        'Epoch': epoch,
                        'Train Step': train_step
                    }, step=train_step)
            elif isinstance(loss_fn, PowerImbalanceV2):
                print('PowerImbalanceV2 deprecated. used mixed with w=0.')
                return 
                # masked_out = out * is_to_pred + data.x * (1 - is_to_pred) # (N, 4)
                # loss = loss_fn(masked_out, data.edge_index, data.edge_attr)
                # train_losses['PowerImbalance']['total'] = train_losses['PowerImbalance'].get('total', 0.) + loss.mean().item() * batch_size
            elif isinstance(loss_fn, MixedMSEPoweImbalanceV2):
                mixed_loss_terms = loss_fn(out, data.edge_index, data.edge_attr, data.y)
                loss = mixed_loss_terms['loss']
                with torch.no_grad():
                    masked_l2_loss_terms = Masked_L2_loss(normalize=False)(out, data.y, is_to_pred)
                for k, v in masked_l2_loss_terms.items():
                    train_losses['MaskedL2'][k] = train_losses['MaskedL2'].get(k, 0.) + v.mean().item() * batch_size
                train_losses['MSE']['total'] = train_losses['MSE'].get('total', 0.) + mixed_loss_terms['mse'].mean().item() * batch_size
                train_losses['PowerImbalance']['total'] = train_losses['PowerImbalance'].get('total', 0.) + mixed_loss_terms['physical'].mean().item() * batch_size
                if log_to_wandb:
                    wandb.log({
                        'Train': {
                            'train_loss': loss.item(),
                            'MaskedL2': {k: v.mean().item() for k, v in masked_l2_loss_terms.items()},
                            'MSE': mixed_loss_terms['mse'].mean().item(),
                            'PowerImbalance': mixed_loss_terms['physical'].mean().item(),
                        }
                    }, step=train_step)
            else:
                print('invalid loss function')
                return

            loss.backward()
            optimizer.step()
            num_samples += batch_size
            
            train_step += 1
            pbar.set_description(f'train loss: {loss.item():.4f}')
            pbar.update(1)
    
    for k, v in train_losses.items():
        for kk, vv in v.items():
            train_losses[k][kk] = vv / num_samples

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
