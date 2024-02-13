import os
import math

import torch
import torch_geometric

from datasets.power_flow_data import create_pf_dp, create_batch_dp, create_dataloader
from networks.MPN import MPN, MPN_simplenet, SkipMPN, MaskEmbdMPN, MultiConvNet, MultiMPN, MaskEmbdMultiMPNV2
from utils.evaluation import load_model

from torch_geometric.loader import DataLoader
from utils.evaluation import evaluate_epoch
from utils.argument_parser import argument_parser

from utils.custom_loss_functions import MaskedL2Eval, PowerImbalanceV2

LOG_DIR = 'logs'
SAVE_DIR = 'models'


@torch.no_grad()
def main():
    run_id = '20240208-1129'
    models = {
        'MPN': MPN,
        'MPN_simplenet': MPN_simplenet,
        'SkipMPN': SkipMPN,
        'MaskEmbdMPN': MaskEmbdMPN,
        'MultiConvNet': MultiConvNet,
        'MultiMPN': MultiMPN,
        'MaskEmbdMultiMPN': MaskEmbdMultiMPNV2,
    }

    args = argument_parser()
    batch_size = args.batch_size
    grid_case = args.case
    data_dir = args.data_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # testset = PowerFlowDataset(root=data_dir, case=grid_case,
    #                         split=[.5, .2, .3], task='test')
    # test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    test_dp = create_pf_dp(data_dir, grid_case, 'test', False, 50000)
    test_loader = create_dataloader(create_batch_dp(
        test_dp, batch_size
    ), num_workers=1, shuffle=False)
    
    pwr_imb_loss = PowerImbalanceV2().to(device)
    masked_l2_split = MaskedL2Eval(normalize=False, split_real_imag=True).to(device)
    masked_l2 = MaskedL2Eval(normalize=False, split_real_imag=False).to(device)
    eval_funcs = {
        'PowerImbalance': pwr_imb_loss,
        'MaskedL2Split': masked_l2_split,
        'MaskedL2': masked_l2,
    }
    
    # Network Parameters
    nfeature_dim = args.nfeature_dim
    efeature_dim = args.efeature_dim
    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    n_gnn_layers = args.n_gnn_layers
    conv_K = args.K
    dropout_rate = args.dropout_rate
    model = models[args.model]

    model = model(
        in_channels_node=nfeature_dim,
        in_channels_edge=efeature_dim,
        out_channels_node=output_dim,
        hidden_dim=hidden_dim,
        n_gnn_layers=n_gnn_layers,
        K=conv_K,
        dropout_rate=dropout_rate,
    ).to(device)  # 40k params
    model.eval()

    model, _ = load_model(model, run_id, device)
    
    print(f"Model: {args.model}")
    print(f"Case: {grid_case}")
    test_losses = evaluate_epoch(model, test_loader, eval_funcs, device, total_length=math.ceil(len(test_dp)/batch_size), batch_size=batch_size)
    print(test_losses)

if __name__ == "__main__":
    main()
