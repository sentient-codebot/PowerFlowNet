"""unit test script for creating dp for power flow data"""
import torch
import numpy as np
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from tqdm import tqdm

from datasets.power_flow_data import create_pf_dp, create_batch_dp

def main():
    dp = create_pf_dp(
        root = 'data/power_flow_dataset/',
        case = '6470rte',
        task = 'train',
        fill_noise = True,
    )
    batch_dp = create_batch_dp(dp, batch_size=128)
    
    num_batch = 0
    for batch in tqdm(batch_dp):
        num_batch += 1
        
    print(f"Number of batches: {num_batch}")
    exit()
        
    
    rs = MultiProcessingReadingService(num_workers=2)
    dl = DataLoader2(batch_dp, reading_service=rs)
    it = iter(dl)
    batch = next(it)
    pass
    
if __name__ == '__main__':
    main()