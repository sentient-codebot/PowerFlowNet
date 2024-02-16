from datetime import datetime
import os
import random
import math
from functools import partial

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from accelerate import Accelerator

from tqdm import tqdm

from datasets.power_flow_data import PowerFlowDataset, create_pf_dp, create_batch_dp, create_dataloader
from networks.MPN import MPN, MPN_simplenet, SkipMPN, MaskEmbdMPN, MultiConvNet, MultiMPN, MaskEmbdMultiMPNV2, TypeSensitiveGCN
from utils.argument_parser import argument_parser
from utils.training import train_epoch, append_to_json
from utils.evaluation import evaluate_epoch
from utils.custom_loss_functions import MaskedL2Loss, PowerImbalanceV2, MixedMSEPoweImbalanceV2, MaskedL2Eval, EdgeWeightType, MaskedL1Eval

import wandb


def main():
    # Step 0: Parse Arguments and Setup
    args = argument_parser()
    run_id = datetime.now().strftime("%Y%m%d") + '-' + str(random.randint(0, 9999))
    LOG_DIR = 'logs'
    SAVE_DIR = 'models'
    TRAIN_LOG_PATH = os.path.join(LOG_DIR, 'train_log/train_log_'+run_id+'.pt')
    SAVE_LOG_PATH = os.path.join(LOG_DIR, 'save_logs.json')
    SAVE_MODEL_PATH = os.path.join(SAVE_DIR, 'model_'+run_id+'.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1234)
    np.random.seed(1234)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    models = {
        'MPN': MPN,
        'MPN_simplenet': MPN_simplenet,
        'SkipMPN': SkipMPN,
        'MaskEmbdMPN': MaskEmbdMPN,
        'MultiConvNet': MultiConvNet,
        'MultiMPN': MultiMPN,
        'MaskEmbdMultiMPN': MaskEmbdMultiMPNV2,
        'TypeSensitiveGCN': TypeSensitiveGCN 
    }

    # Training parameters
    data_dir = args.data_dir
    nomalize_data = not args.disable_normalize
    num_epochs = args.num_epochs
    
    lr = args.lr
    batch_size = args.batch_size
    grid_case = args.case
    
    alpha = args.alpha
    tau = args.tau
    
    # Network parameters
    nfeature_dim = args.nfeature_dim
    efeature_dim = args.efeature_dim
    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    n_gnn_layers = args.n_gnn_layers
    conv_K = args.K
    dropout_rate = args.dropout_rate
    model = models[args.model]

    log_to_wandb = args.wandb
    wandb_entity = args.wandb_entity
    if log_to_wandb:
        wandb.init(project="PowerFlowNet",
                   name=run_id,
                   config=vars(args))

    # Step 1: Load data
    train_dp = create_pf_dp(data_dir, grid_case, 'train', False, 50000, random_bus_type=args.random_bus_type)
    train_batch_dp, trans, inv_trans = create_batch_dp(train_dp, batch_size, normalize=nomalize_data)
    val_dp = create_pf_dp(data_dir, grid_case, 'val', False, 50000, transforms=list(trans['data'].values()))
    test_dp = create_pf_dp(data_dir, grid_case, 'test', False, 50000, transforms=list(trans['data'].values()))
    
    if len(train_dp) == 0 or len(val_dp) == 0 or len(test_dp) == 0:
        print("No enough data found for all three tasks. Please check the data directory and the case name.")
    
    print(f"#Samples: training {len(train_dp)}, validation {len(val_dp)}, test {len(test_dp)}")
    
    train_loader = create_dataloader(train_batch_dp, num_workers=1, shuffle=True)
    val_loader = create_dataloader(create_batch_dp(val_dp, batch_size)[0], num_workers=1, shuffle=False)
    test_loader = create_dataloader(create_batch_dp(test_dp, batch_size)[0], num_workers=1, shuffle=False)
    
    # Step 2: Create data-dependent loss function
    ##  train loss function
    LossFunc = partial(MixedMSEPoweImbalanceV2, normalize=True, split_real_imag=False, pre_transforms=inv_trans, edge_weight_type=EdgeWeightType.IMPEDANCE)
    if args.train_loss_fn == 'power_imbalance':
        train_loss_fn = LossFunc(alpha=0., tau=1.).to(device)
    elif args.train_loss_fn == 'mse':
        train_loss_fn = LossFunc(alpha=1., tau=0.).to(device)
    elif args.train_loss_fn == 'mixed_mse_power_imbalance' or args.train_loss_fn == 'mixed':
        train_loss_fn = LossFunc(alpha=alpha, tau=tau).to(device)
    else:
        train_loss_fn = LossFunc(alpha=1., tau=0.).to(device) # mse
    ##  eval loss function
    eval_funcs = {
        'MaskedL2': MaskedL2Eval(normalize=False, split_real_imag=False, pre_transforms=inv_trans),
        'MaskedL2Split': MaskedL2Eval(normalize=False, split_real_imag=True, pre_transforms=inv_trans),
        'MaskedL1': MaskedL1Eval(normalize=False, split_real_imag=False, pre_transforms=inv_trans),
        'MaskedL1Split': MaskedL1Eval(normalize=False, split_real_imag=True, pre_transform=inv_trans),
        'PowerImbalance': PowerImbalanceV2(pre_transforms=inv_trans)
    }
    
    # Step 2: Create model and optimizer (and scheduler)
    model = model(
        in_channels_node=nfeature_dim,
        in_channels_edge=efeature_dim,
        out_channels_node=output_dim,
        hidden_dim=hidden_dim,
        n_gnn_layers=n_gnn_layers,
        K=conv_K,
        dropout_rate=dropout_rate
    ).to(device) 

    #calculate model size
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: ", pytorch_total_params)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                        mode='min',
    #                                                        factor=0.5,
    #                                                        patience=5,
    #                                                        verbose=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=max(len(train_dp)//batch_size,1), epochs=num_epochs)

    # Step 3: Train model
    best_train_loss = 10000.
    best_val_loss = 10000.
    train_log = {
        'train': {
            'loss': []},
        'val': {
            'loss': []},
    }
    # pbar = tqdm(range(num_epochs), total=num_epochs, position=0, leave=True)
    # data = next(iter(create_batch_dp(train_dp, 1)))
    # losses = eval_loss_fn(data.y, data.edge_index, data.edge_attr, data.y)
    # exit()
    accelerator = Accelerator()
    model, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, scheduler
    )
    train_step = 0
    for epoch in range(num_epochs):
        print('Epoch:', epoch+1, '/', num_epochs)
        train_losses, train_step = train_epoch(
            accelerator,
            model, 
            train_loader, 
            train_loss_fn, 
            optimizer, 
            device, 
            total_length=math.ceil(len(train_dp)/batch_size), 
            batch_size=batch_size, 
            log_to_wandb=log_to_wandb, 
            epoch=epoch, 
            train_step=train_step,
            inverse_transforms=inv_trans
        )
        
        val_losses = evaluate_epoch(
            model, 
            val_loader, 
            eval_funcs, 
            device, 
            total_length=math.ceil(len(val_dp)/batch_size), 
            batch_size=batch_size,
        )
        
        train_loss = train_losses['PowerImbalance']['total'] if isinstance(train_loss_fn, PowerImbalanceV2) else train_losses['MaskedL2']['total']
        val_loss = val_losses['PowerImbalance']['total'] if isinstance(train_loss_fn, PowerImbalanceV2) else val_losses['MaskedL2']['total']
        accelerator.wait_for_everyone()
        scheduler.step()
        train_log['train']['loss'].append(train_loss)
        train_log['val']['loss'].append(val_loss)

        if log_to_wandb:
            wandb.log({'Epoch Train': train_losses,
                      'Epoch Validation': val_losses,
                      }, step = train_step)

        if train_loss < best_train_loss:
            best_train_loss = train_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if args.save:
                _to_save = {
                    'epoch': epoch,
                    'args': args,
                    'val_loss': best_val_loss,
                    'model_state_dict': model.state_dict(),
                    'transforms': trans,
                    'inverse_transforms': inv_trans,
                }
                os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)
                torch.save(_to_save, SAVE_MODEL_PATH)
                append_to_json(
                    SAVE_LOG_PATH,
                    run_id,
                    {
                        'val_loss': f"{best_val_loss: .4f}",
                        # 'test_loss': f"{test_loss: .4f}",
                        'train_log': TRAIN_LOG_PATH,
                        'saved_file': SAVE_MODEL_PATH,
                        'epoch': epoch,
                        'model': args.model,
                        'train_case': args.case,
                        'train_loss_fn': args.train_loss_fn,
                        'args': vars(args)
                    }
                )
                os.makedirs(os.path.dirname(TRAIN_LOG_PATH), exist_ok=True)
                torch.save(train_log, TRAIN_LOG_PATH)

        print(f"Epoch {epoch+1} / {num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, best_val_loss={best_val_loss:.4f}")

    
    
    print(f"Training Complete. Best validation loss: {best_val_loss:.4f}")
    
    # Step 4: Evaluate model
    if args.save:
        _to_load = torch.load(SAVE_MODEL_PATH)
        model.load_state_dict(_to_load['model_state_dict'])
        test_losses = evaluate_epoch(model, test_loader, eval_funcs, device, total_length=math.ceil(len(val_dp)/batch_size), batch_size=batch_size)
        print(f"Test loss: {test_losses}")
        if log_to_wandb:
            wandb.run.summary['Test Losses'] = test_losses

    # Step 5: Save results
    os.makedirs(os.path.join(LOG_DIR, 'train_log'), exist_ok=True)
    if args.save:
        torch.save(train_log, TRAIN_LOG_PATH)


if __name__ == '__main__':
    main()