from datetime import datetime
import os
import random

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from tqdm import tqdm

from datasets.power_flow_data import PowerFlowDataset, create_pf_dp, create_batch_dp, create_dataloader
from networks.MPN import MPN, MPN_simplenet, SkipMPN, MaskEmbdMPN, MultiConvNet, MultiMPN, MaskEmbdMultiMPNV2
from utils.argument_parser import argument_parser
from utils.training import train_epoch, append_to_json
from utils.evaluation import evaluate_epoch
from utils.custom_loss_functions import Masked_L2_loss, PowerImbalanceV2, MixedMSEPoweImbalanceV2

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
    models = {
        'MPN': MPN,
        'MPN_simplenet': MPN_simplenet,
        'SkipMPN': SkipMPN,
        'MaskEmbdMPN': MaskEmbdMPN,
        'MultiConvNet': MultiConvNet,
        'MultiMPN': MultiMPN,
        'MaskEmbdMultiMPN': MaskEmbdMultiMPNV2
    }

    # Training parameters
    data_dir = args.data_dir
    nomalize_data = not args.disable_normalize
    num_epochs = args.num_epochs
    loss_fn = Masked_L2_loss(regularize=args.regularize, regcoeff=args.regularization_coeff, normalize=True)
    eval_loss_fn = MixedMSEPoweImbalanceV2(alpha=0.9, tau=0.020, noramlize=False)
    lr = args.lr
    batch_size = args.batch_size
    grid_case = args.case
    
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1234)
    np.random.seed(1234)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Step 1: Load data
    # trainset = PowerFlowDataset(root=data_dir, case=grid_case, split=[.5, .2, .3], task='train', normalize=nomalize_data)
    # valset = PowerFlowDataset(root=data_dir, case=grid_case, split=[.5, .2, .3], task='val', normalize=nomalize_data)
    # testset = PowerFlowDataset(root=data_dir, case=grid_case, split=[.5, .2, .3], task='test', normalize=nomalize_data)
        
    # train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    train_dp = create_pf_dp(data_dir, grid_case, 'train', True)
    val_dp = create_pf_dp(data_dir, grid_case, 'val', False)
    test_dp = create_pf_dp(data_dir, grid_case, 'test', False)
    
    if len(train_dp) == 0 or len(val_dp) == 0 or len(test_dp) == 0:
        raise ValueError("No data found. Please check the data directory and the case name.")
    
    print(f"#Samples: training {len(train_dp)}, validation {len(val_dp)}, test {len(test_dp)}")
    
    train_loader = create_dataloader(create_batch_dp(train_dp, batch_size), num_workers=4, shuffle=True)
    val_loader = create_dataloader(create_batch_dp(val_dp, batch_size), num_workers=4, shuffle=False)
    test_loader = create_dataloader(create_batch_dp(test_dp, batch_size), num_workers=4, shuffle=False)
    
    ## [Optional] physics-informed loss function
    if args.train_loss_fn == 'power_imbalance':
        # overwrite the loss function
        loss_fn = PowerImbalanceV2().to(device)
    elif args.train_loss_fn == 'masked_l2':
        loss_fn = Masked_L2_loss(regularize=args.regularize, regcoeff=args.regularization_coeff)
    elif args.train_loss_fn == 'mixed_mse_power_imbalance':
        loss_fn = MixedMSEPoweImbalanceV2(alpha=0.9, tau=0.020, noramlize=True).to(device)
    else:
        loss_fn = torch.nn.MSELoss()
    
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
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_dp)//batch_size, epochs=num_epochs)

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
    for epoch in range(num_epochs):
        print('Epoch:', epoch+1, '/', num_epochs)
        train_losses = train_epoch(
            model, train_loader, loss_fn, optimizer, device, 
            total_length=len(train_dp)//batch_size, batch_size=batch_size)
        
        val_losses = evaluate_epoch(model, val_loader, eval_loss_fn, device, 
                                    total_length=len(val_dp)//batch_size, batch_size=batch_size)
        
        train_loss = train_losses['PowerImbalance']['total'] if isinstance(loss_fn, PowerImbalanceV2) else train_losses['MaskedL2']['total']
        val_loss = val_losses['PowerImbalance']['total'] if isinstance(loss_fn, PowerImbalanceV2) else val_losses['MaskedL2']['total']
        scheduler.step()
        train_log['train']['loss'].append(train_loss)
        train_log['val']['loss'].append(val_loss)

        if log_to_wandb:
            wandb.log({'train_loss': train_losses,
                      'val_loss': val_loss}, step=epoch)

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
        test_loss = evaluate_epoch(model, test_loader, eval_loss_fn, device)
        print(f"Test loss: {best_val_loss:.4f}")
        if log_to_wandb:
            wandb.log({'test_loss', test_loss})

    # Step 5: Save results
    os.makedirs(os.path.join(LOG_DIR, 'train_log'), exist_ok=True)
    if args.save:
        torch.save(train_log, TRAIN_LOG_PATH)


if __name__ == '__main__':
    main()