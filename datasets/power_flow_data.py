"""
this file defines the class of PowerFlowData, which is used to load the data of Power Flow
"""
import os
from copy import deepcopy
from typing import Callable, Optional, List, Tuple, Union
from functools import partial

import numpy as np
import torch
from torch.utils.data import IterDataPipe
import torchdata.datapipes.iter as pipes
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService, adapter
import torch_geometric
from torch_geometric.data import Data, Batch, InMemoryDataset

from .datapipe_utils import get_filelist, get_existing_node_edge

ori_feature_names_x = [
    'index',                # - removed
    'type',                 # --- one-hot encoded,  0, 1, 2, 3
    'voltage magnitude',    # --- this matters,     4, 
    'voltage angle',        # --- this matters,     5,
    'Pd',                   # --- this matters, preprocessed as Pd-Pg   6,
    'Qd',                   # --- this matters                          7,
    'Gs',                   # - equivalent to Pd, Qd                    8,
    'Bs',                   # - equivalent to Pd, Qd                    9,
    'Pg'                    # - removed
]

feature_names_x = [
    'type_0',               # --- one-hot encoded,  0,
    'type_1',               # --- one-hot encoded,  1,
    'type_2',               # --- one-hot encoded,  2,
    'type_3',               # --- one-hot encoded,  3,
    'voltage magnitude',    # --- this matters,     4, 
    'voltage angle',        # --- this matters,     5,
    'Pd',                   # --- this matters, preprocessed as Pd-Pg   6,
    'Qd',                   # --- this matters                          7,
    'Gs',                   # - equivalent to Pd, Qd                    8,
    'Bs',                   # - equivalent to Pd, Qd                    9,
    'to_predict_voltage_magnitude', #               10,
    'to_predict_voltage_angle',     #               11,
    'to_predict_Pd',                #               12,
    'to_predict_Qd'                 #               13,
    'to_predict_Gs',                #               14,
    'to_predict_Bs'                 #               15,
]

ori_feature_names_y = [
    'index',                # - removed
    'type',                 # - removed
    'voltage magnitude',    # --- we care about this
    'voltage angle',        # --- we care about this
    'active power',         # --- we care about this
    'reactive power',       # --- we care about this
    'Gs',                   # -
    'Bs'                    # -
]

feature_names_y = [
    'voltage magnitude',    # --- we care about this
    'voltage angle',        # --- we care about this
    'active power',         # --- we care about this
    'reactive power',       # --- we care about this
    'Gs',                   # -
    'Bs'         
]

edge_feature_names = [
    'r',                    # --- this matters, resistance, pu
    'x',                    # --- this matters, reactance,  pu
    'b',
    'tau',
    'angle'
]
        

class PowerFlowDataset(InMemoryDataset):
    """PowerFlowData(InMemoryDataset)

    Parameters:
        root (str, optional) - Root directory where the dataset should be saved. (optional: None)
        pre_filter (callable)- A function 

    Comments:
        we actually do not need adjacency matrix, since we can use edge_index to represent the graph from `edge_features`

    Returns:
        class instance of PowerFlowData
    """
    partial_file_names = [
        "edge_features.npy",
        "node_features_x.npy",
        "node_features_y.npy"
    ]
    split_order = {
        "train": 0,
        "val": 1,
        "test": 2
    }
    mixed_cases = [
        '118v2',
        '14v2',
    ]

    def __init__(self, 
                root: str, 
                case: str = '14', 
                split: Optional[List[float]] = None, 
                task: str = "train", 
                transform: Optional[Callable] = None, 
                pre_transform: Optional[Callable] = None, 
                pre_filter: Optional[Callable] = None,
                normalize=True):

        assert len(split) == 3
        assert task in ["train", "val", "test"]
        self.normalize = normalize
        self.case = case  # THIS MUST BE EXECUTED BEFORE super().__init__() since it is used in raw_file_names and processed_file_names
        self.split = split
        self.task = task
        super().__init__(root, transform, pre_transform, pre_filter)
        self.mask = torch.tensor([])
        self.data, self.slices, self.mask = self._normalize_dataset(
            *torch.load(self.processed_paths[0]))  # necessary, do not forget!

    def get_data_dimensions(self):
        return self[0].x.shape[1], self[0].y.shape[1], self[0].edge_attr.shape[1]

    def get_data_means_stds(self):
        assert self.normalize == True
        return self.xymean[:1, :], self.xystd[:1, :], self.edgemean[:1, :], self.edgestd[:1, :]

    def _normalize_dataset(self, data, slices) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.normalize:
            # TODO an actual mask, perhaps never necessary though
            return data, slices, torch.tensor([])

        # selecting the right features
        # for x
        print(data.x.shape)
        print(data)
        # exit()
        data.x[:, 4] = data.x[:, 4] - data.x[:, 8]  # Pd = Pd - Pg
        # + 4 for the one-hot encoding for four node types, -2 because we remove the index and Pg
        template = torch.zeros((data.x.shape[0], data.x.shape[1] + 3 - 2))
        template[:, 0:4] = torch.nn.functional.one_hot(
            data.x[:, 1].type(torch.int64), num_classes=4)
        template[:, 4:10] = data.x[:, 2:8]
        data.x = template
        # for y
        data.y = data.y[:, 2:]

        # SHAPE NOW: torch.Size([14, 10]) torch.Size([14, 6]) for x and y

        # normalizing
        # for node attributes
        xy = torch.concat([data.x[:, 4:], data.y], dim=0)
        # 6 for:
        mean = torch.mean(xy, dim=0).unsqueeze(
            dim=0).expand(data.x.shape[0], 6)
        std = torch.std(xy, dim=0).unsqueeze(dim=0).expand(
            data.x.shape[0], 6)  # Vm, Va, Pd, Qd, Gs, Bs
        self.xymean, self.xystd = mean, std
        # + 0.0000001 to avoid NaN's because of division by zero
        data.x[:, 4:] = (data.x[:, 4:] - mean) / (std + 0.0000001)
        data.y = (data.y - mean) / (std + 0.0000001)
        # for edge attributes
        mean = torch.mean(data.edge_attr, dim=0).unsqueeze(dim=0).expand(
            data.edge_attr.shape[0], data.edge_attr.shape[1])
        std = torch.std(data.edge_attr, dim=0).unsqueeze(dim=0).expand(
            data.edge_attr.shape[0], data.edge_attr.shape[1])
        self.edgemean, self.edgestd = mean, std
        data.edge_attr = (data.edge_attr - mean) / (std + 0.0000001)

        # adding the mask
        # where x and y are unequal, the network must predict
        # 1 where value changed, 0 where it did not change
        unequal = (data.x[:, 4:] != data.y).float()
        data.prediction_mask = unequal
        data.x = torch.concat([data.x, unequal], dim=1)

        return data, slices, unequal

    @property
    def raw_file_names(self) -> List[str]:
        if self.case != 'mixed':
            return ["case"+f"{self.case}"+"_"+name for name in self.partial_file_names]
        else:
            return ["case"+f"{case}"+"_"+name for case in self.mixed_cases for name in self.partial_file_names]

    @property
    def processed_file_names(self) -> List[str]:
        return ["case"+f"{self.case}"+"_processed_data.pt"]

    def len(self):
        return self.slices['x'].shape[0]-1

    # def get(self, idx: int) -> Data: # override
    #     return self.data[idx]

    def process(self):
        # then use from_scipy_sparse_matrix()
        assert len(self.raw_paths) % 3 == 0
        raw_paths_per_case = [[self.raw_paths[i], self.raw_paths[i+1], self.raw_paths[i+2],] for i in range(0, len(self.raw_paths), 3)]
        data_list = []
        for case, raw_paths in enumerate(raw_paths_per_case):
            # adj_mat = dense_to_sparse(torch.from_numpy(np.load(raw_paths[0])))
            edge_features = torch.from_numpy(np.load(raw_paths[0])).float()
            node_features_x = torch.from_numpy(np.load(raw_paths[1])).float()
            node_features_y = torch.from_numpy(np.load(raw_paths[2])).float()

            if self.split is not None:
                split_len = [int(len(node_features_x) * i) for i in self.split]
                edge_features = torch.split(edge_features, split_len, dim=0)[
                    self.split_order[self.task]]
                node_features_x = torch.split(node_features_x, split_len, dim=0)[
                    self.split_order[self.task]]
                node_features_y = torch.split(node_features_y, split_len, dim=0)[
                    self.split_order[self.task]]

            per_case_data_list = [
                Data(
                    x=node_features_x[i],
                    y=node_features_y[i],
                    edge_index=edge_features[i, :, 0:2].T.to(torch.long)-1,
                    edge_attr=edge_features[i, :, 2:],
                ) for i in range(len(node_features_x))
            ]
            
            data_list.extend(per_case_data_list)

        if self.pre_filter is not None:  # filter out some data
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def create_pf_dp(
    root: str,
    case: str,
    task: str,
    fill_noise: bool,
    random_bus_type: bool = False,
    total_samples: int = 50000,
    transforms: list[Callable] = [],
) -> IterDataPipe:
    """
    process: 
        - two datapipes in parallel for node and edge
        - list csv in sorted order
        - filter out the ones that are not in the split (first 50% train, then 20% val, then 30% test)
        - read csv as np.array
        - merge two dp into one that creates Data with node and edges
    """
    # dataset specific constants
    total_samples = total_samples
    split = [0.5, 0.2, 0.3]
    split_indices = {
        'train': list(range(int(total_samples*split[0]))),
        'val': list(range(int(total_samples*split[0]), int(total_samples*sum(split[:2])))),
        'test': list(range(int(total_samples*sum(split[:2])), total_samples)),
    }
    if task == 'validation':
        task = 'val'
    
    # node and edge file list
    node_files = get_filelist(os.path.join(root, 'raw'), 'node_features', case, split_indices[task])
    edge_files = get_filelist(os.path.join(root, 'raw'), 'edge_features', case, split_indices[task])
    sn_mva_files = get_filelist(os.path.join(root, 'raw'), 'sn_mva', case, split_indices[task])
    node_files, edge_files = get_existing_node_edge(node_files, edge_files)
        
    # load node and edge together
    dp = pipes.IterableWrapper(
        zip(node_files, edge_files, sn_mva_files),
    )
    dp = dp.read_pf_data(length=len(node_files)) # (node_array [N, 6], edge_array [E, 4])
    dp = dp.create_geometric_data(fill_noise=fill_noise, random_bus_type=random_bus_type) 
    
    if 'data' in transforms:
        transforms = transforms['data']
    for transform in transforms:
        dp = dp.map(transform)
    
    return dp

def create_batch_dp(dp: IterDataPipe, batch_size: int, normalize: bool = False) -> IterDataPipe:
    """
    Create a datapipe to shuffle and batch the data, collate the batched data, and yield. 
    """
    transforms = {'data': {}, 'node': {}, 'edge': {}}
    inverse_transforms = {'data': {}, 'node': {}, 'edge': {}}
    collate_fn = partial(torch_geometric.data.Batch.from_data_list)
    dp = dp.shuffle(buffer_size=500)
    dp = dp.batch(batch_size)
    dp = dp.collate(collate_fn=collate_fn) # collate_fn is a function that takes a list of samples from dp and returns a batch
    if normalize:
        dp = dp.instance_normalize()
        transforms['data']['normalize'] = dp.normalize
        transforms['node']['normalize'] = dp.normalize_node
        transforms['edge']['normalize'] = dp.normalize_edge
        inverse_transforms['data']['normalize'] = dp.denormalize
        inverse_transforms['node']['normalize'] = dp.denormalize_node
        inverse_transforms['edge']['normalize'] = dp.denormalize_edge
        
    dp = dp.sharding_filter() # allows duplicating dp for multiple workers. skipping this will cause issues
    
    return dp, transforms, inverse_transforms

def create_dataloader(dp: IterDataPipe, num_workers: int, shuffle: bool = True) -> DataLoader2:
    """
    Create a DataLoader2 with MultiProcessingReadingService
    """
    rs = MultiProcessingReadingService(num_workers=num_workers)
    dl = DataLoader2(dp, datapipe_adapter_fn=adapter.Shuffle(shuffle), reading_service=rs)
    return dl

def main():
    try:
        # shape = (N, n_edges, 7)       (from, to, ...)
        edge_features = np.load("data/raw/case14_edge_features.npy")
        # shape = (N, n_nodes, n_nodes)
        adj_matrix = np.load("data/raw/case14_adjacency_matrix.npy")
        # shape = (N, n_nodes, 9)
        node_features_x = np.load("data/raw/case14_node_features_x.npy")
        # shape = (N, n_nodes, 8)
        node_features_y = np.load("data/raw/case14_node_features_y.npy")
    except FileNotFoundError:
        print("File not found.")

    print(f"edge_features.shape = {edge_features.shape}")
    print(f"adj_matrix.shape = {adj_matrix.shape}")
    print(f"node_features_x.shape = {node_features_x.shape}")
    print(f"node_features_y.shape = {node_features_y.shape}")

    trainset = PowerFlowDataset(root="data", case=14,
                             split=[.5, .2, .3], task="train")
    train_loader = torch_geometric.loader.DataLoader(
        trainset, batch_size=12, shuffle=True)
    print(len(trainset))
    print(trainset[0])
    print(next(iter(train_loader)))
    pass


if __name__ == "__main__":
    main()