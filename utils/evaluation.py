"""This module provides functions for 
    - evaluation_epoch - evaluate performance over a whole epoch
    - other evaluation metrics function [NotImplemented]
"""
from typing import Callable, Optional, Union, Tuple
import os

import torch
from torch_geometric.loader import DataLoader
from torch.optim.optimizer import Optimizer
import torch.nn as nn
from tqdm import tqdm

from utils.custom_loss_functions import MaskedL2Eval, PowerImbalanceV2, MaskedL1Eval, MixedMSEPoweImbalanceV2, get_mask_from_bus_type

LOG_DIR = 'logs'
SAVE_DIR = 'models'


def load_model(
    model: nn.Module,
    run_id: str,
    device: Union[str, torch.device]
) -> Tuple[nn.Module, dict]:
    SAVE_MODEL_PATH = os.path.join(SAVE_DIR, 'model_'+run_id+'.pt')
    if type(device) == str:
        device = torch.device(device)

    try:
        saved = torch.load(SAVE_MODEL_PATH, map_location=device)
        model.load_state_dict(saved['model_state_dict'])
    except FileNotFoundError:
        print("File not found. Could not load saved model.")
        return -1

    return model, saved


def num_params(model: nn.Module) -> int:
    """
    Returns the number of trainable parameters in a neural network model.

    Args:
        model (nn.Module): The neural network model.

    Returns:
        int: The number of trainable parameters in the model.

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate_epoch(
        model: nn.Module,
        loader: DataLoader,
        eval_funcs: dict[str, Callable],
        device: str = 'cpu',
        total_length: int=100000,
        batch_size: int=128
) -> dict[str, dict[str, float]]:
    """
    Evaluates the performance of a trained neural network model on a dataset using the specified data loader.

    Args:
        model (nn.Module): The trained neural network model to be evaluated.
        loader (DataLoader): The PyTorch Geometric DataLoader containing the evaluation data.
        device (str): The device used for evaluating the model (default: 'cpu').

    Returns:
        float: The mean loss value over all the batches in the DataLoader.

    """
    model.eval()
    _sum_weight = 0
    eval_losses = {func_name: {} for func_name in eval_funcs.keys()}
    pbar = tqdm(loader, initial=1, total=total_length+1, desc='Evaluating:')
    for data in pbar:
        _weight = data.x.shape[0] # actually = batch_size * num_nodes per sample. used as average weights over batches
        data = data.to(device)
        out = model(data)

        is_to_pred = get_mask_from_bus_type(data.bus_type) # 0, 1 mask of (N, 4). 1 is need to predict
        for func_name, func in eval_funcs.items():
            if isinstance(func, MaskedL2Eval) or isinstance(func, MaskedL1Eval):
                # averaged per scenario per node
                loss_terms = func(out, data.y, is_to_pred)
                for term_name, value in loss_terms.items():
                    eval_losses[func_name][term_name] = eval_losses[func_name].get(term_name, 0.) + value.mean().item() * _weight
            elif isinstance(func, PowerImbalanceV2):
                # averaged per scenario
                loss = func(out, data.edge_index, data.edge_attr)
                eval_losses[func_name]['total'] = eval_losses[func_name].get('total', 0.) + loss.mean().item() * _weight
            else:
                print("you shouldn't use other eval functions.")
                pass
            
        _sum_weight += _weight
    _angle_pred = out[:, 1].detach()
    _angle_target = data.y[:, 1]
    print(f'Normalized Angle')
    for i in range(10):
        print(f"target/pred: {_angle_target[i]:.5f}\t| {_angle_pred[i]:.5f}")
    
    for func_name, terms in eval_losses.items():
        for term_name, term_value in terms.items():
            eval_losses[func_name][term_name] = term_value / _sum_weight
    return eval_losses
