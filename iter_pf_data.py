"""unit test script for creating dp for power flow data"""
import torch
import numpy as np
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from tqdm import tqdm

from datasets.power_flow_data import create_pf_dp, create_batch_dp

def main():
    dp = create_pf_dp(
        root = 'data/power_flow_dataset/',
        case = '118',
        task = 'train',
        fill_noise = True,
    )
    batch_dp = create_batch_dp(dp, batch_size=2)
    
    rs = MultiProcessingReadingService(num_workers=4)
    dl = DataLoader2(batch_dp, reading_service=rs)
    it = iter(dl)
    num_batches = 0
    for batch in tqdm(it):
        print(type(batch))
    print(f'num_batches: {num_batches}')
    pass
    
if __name__ == '__main__':
    main()