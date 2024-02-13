from datetime import datetime
import os
import random

import numpy as np
import torch
from torch_geometric.loader import DataLoader

import json
from tqdm import tqdm

from datasets.power_flow_data import PowerFlowDataset
from networks.MPN import MPN, MPN_simplenet, SkipMPN, MaskEmbdMPN, MultiConvNet, MultiMPN, MaskEmbdMultiMPN, MaskEmbdMultiMPN_NoMP
from utils.argument_parser import argument_parser
from utils.training import train_epoch, append_to_json
from utils.evaluation import evaluate_epoch
from utils.custom_loss_functions import MaskedL2Loss, PowerImbalance, MixedMSEPoweImbalance

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
        'MaskEmbdMultiMPN': MaskEmbdMultiMPN
    }
    cases = ['14', '118', '6470rte']

    # Training parameters
    data_dir = args.data_dir
    num_epochs = args.num_epochs
    num_epochs = 1
    eval_loss_fn = MaskedL2Loss(regularize=False)
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
    if log_to_wandb:
        wandb.init(project="PowerFlowNet",
                   entity="PowerFlowNet",
                   name=run_id,
                   config=args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1234)
    np.random.seed(1234)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Step 1: Load data
    trainsets = [PowerFlowDataset(
        root=data_dir, case=case, split=[.5, .2, .3], task='train') for case in cases]
    valsets = [PowerFlowDataset(
        root=data_dir, case=case, split=[.5, .2, .3], task='val') for case in cases]
    testsets = [PowerFlowDataset(
        root=data_dir, case=case, split=[.5, .2, .3], task='test') for case in cases]

    train_loaders = []
    val_loaders = []
    test_loaders = []
    for i in range(len(trainsets)):
        if i > 1:
            batch_size = 32
        elif i == 1:
            batch_size = 1024
        else:
            batch_size = 2048

        train_loaders.append(DataLoader(
            trainsets[i], batch_size=batch_size, shuffle=True))
        val_loaders.append(DataLoader(
            valsets[i], batch_size=batch_size, shuffle=False))
        test_loaders.append(DataLoader(
            testsets[i], batch_size=batch_size, shuffle=False))

    loss_fn = torch.nn.MSELoss()

    # Step 2: Create model and optimizer (and scheduler)
    node_in_dim, node_out_dim, edge_dim = trainsets[i].get_data_dimensions()
    assert node_in_dim == 16

    model_full = MaskEmbdMultiMPN(
        nfeature_dim=nfeature_dim,
        efeature_dim=efeature_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        n_gnn_layers=n_gnn_layers,
        K=conv_K,
        dropout_rate=dropout_rate
    ).to(device)

    model_No = MaskEmbdMultiMPN(
        nfeature_dim=nfeature_dim,
        efeature_dim=efeature_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        n_gnn_layers=1,
        K=conv_K,
        dropout_rate=dropout_rate
    ).to(device)

    model_None = MaskEmbdMultiMPN_NoMP(
        nfeature_dim=nfeature_dim,
        efeature_dim=efeature_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        n_gnn_layers=n_gnn_layers,
        K=conv_K,
        dropout_rate=dropout_rate
    ).to(device)

    model_One = MaskEmbdMultiMPN_NoMP(
        nfeature_dim=nfeature_dim,
        efeature_dim=efeature_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        n_gnn_layers=1,
        K=conv_K,
        dropout_rate=dropout_rate
    ).to(device)

    models = [model_full, model_No, model_One, model_None]

    model_names = ['model_full', 'model_1Conv',
                   'model_NoMP', 'model_1Conv_NoMP']

    results = {'model': [],
               'trained_on': [],
               'evaluated_on': [],
               'test_loss': [], }

    for i, model_to_load in enumerate(models):
        

        # train on case with model i
        for c, case in enumerate(cases):
            model = model_to_load

            if c > 1:
                num_epochs = 30
            else:
                num_epochs = 100
            num_epochs = 1

            pytorch_total_params = sum(p.numel() for p in model.parameters())
            print("\n\n===========================================")
            print(f'Case {case}, model {model_names[i]}:')
            print("Total number of parameters: ", pytorch_total_params)

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr, steps_per_epoch=len(train_loaders[c]), epochs=num_epochs)

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
                train_loss = train_epoch(
                    model, train_loaders[c], loss_fn, optimizer, device)
                val_loss = evaluate_epoch(
                    model, val_loaders[c], eval_loss_fn, device)
                scheduler.step()
                train_log['train']['loss'].append(train_loss)
                train_log['val']['loss'].append(val_loss)

                if log_to_wandb:
                    wandb.log({'train_loss': train_loss,
                               'val_loss': val_loss})

            # evaluate model i on all cases
            for cc, case_n in enumerate(cases):

                test_loss = evaluate_epoch(
                    model, test_loaders[cc], eval_loss_fn, device)
                print('--Results ', model_names[i], '| "Trained on: ', cases[c],
                      'Evaluated on: ', cases[cc], '| Test loss: ', test_loss)

                results['model'].append(model_names[i])
                results['trained_on'].append(cases[c])
                results['evaluated_on'].append(cases[cc])
                results['test_loss'].append(test_loss)

                if log_to_wandb:
                    wandb.log({'test_loss': test_loss})
            torch.cuda.empty_cache()
            print("\n\n\n===========================================")
            # print the contents of the results dictionary aligned using print formatting
            print("Results dictionary:")
            print("model           | trained_on  | evaluated_on | test_loss")
            print("--------------------------------------------------------------------")
            for j in range(len(results['model'])):
                print('{: <15} | {: <11} | {: <12} | {: <8}'.format(
                    results['model'][j], results['trained_on'][j], results['evaluated_on'][j], results['test_loss'][j]))

            print("--------------------------------------------------------------------")
            # save the results dictionary as json to root folder

            with open("generalization.json", "w") as outfile:
                json.dump(results, outfile)


if __name__ == '__main__':
    main()
