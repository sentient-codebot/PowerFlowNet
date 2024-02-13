"""provide torchdata iter datepipes for creating torch geometric graph datasets
Author: Nan Lin
Date: 2024-02-06
"""
import os

import torch
import numpy as np
from torch.utils.data import IterDataPipe, functional_datapipe

from torch_geometric.data import Data

def get_filelist(raw_dir, feature, case, indices):
    "raw_dir/{feature}/case{case}_{feature}_{index}.csv"
    _get_path = lambda index: os.path.join(
        raw_dir,
        'case'+case,
        feature,
        f'case{case}_{feature}_{index}.csv'
    )
    return [_get_path(index) for index in indices]

def filter_csv(filename):
    return filename.endswith('.csv') and filename.startswith('case')

def filter_indices(filename, indices):
    " .../case118_..._{index}.csv"
    filename = os.path.basename(filename)
    filename = filename.split('.')[0]
    index = filename.split('_')[-1]
    index = int(index)
    return index in indices

def get_existing_node_edge(node_files, edge_files):
    "get only indices where both node and edge files are present"
    assert len(node_files) == len(edge_files)
    idx = 0
    while idx < len(node_files):
        if (not os.path.exists(node_files[idx])) or (not os.path.exists(edge_files[idx])):
            node_files.pop(idx)
            edge_files.pop(idx)
        else:
            idx += 1        
    
    return node_files, edge_files

@functional_datapipe('read_node_features')
class ReadNodeFeatures(IterDataPipe):
    " source_dp: a datapipe that yields csv paths"
    def __init__(self, source_dp):
        super().__init__()
        self.dp = source_dp
        # index, type, vm, va, p, q
        self.dtype = np.dtype([
            ('index', np.int32),
            ('type', np.int32),
            ('vm', np.float32),
            ('va', np.float32),
            ('p', np.float32),
            ('q', np.float32),
        ])

    def __iter__(self):
        for path in self.dp:
            with open(path, 'r') as f:
                array = np.loadtxt(f, delimiter=',', dtype=self.dtype, comments=None, skiprows=1)
            yield array
        
@functional_datapipe('read_edge_features')
class ReadEdgeFeatures(IterDataPipe):
    " source_dp: a datapipe that yields csv paths"
    def __init__(self, source_dp):
        super().__init__()
        self.dp = source_dp
        # from, to, r, x
        self.dtype = np.dtype([
            ('from', np.int32),
            ('to', np.int32),
            ('r', np.float32),
            ('x', np.float32),
        ])
        
    def __iter__(self):
        for path in self.dp:
            with open(path, 'r') as f:
                array = np.loadtxt(f, delimiter=',', dtype=self.dtype, comments=None, skiprows=1)
            yield array
        
@functional_datapipe('read_pf_data')
class ReadPFData(IterDataPipe):
    " source_dp: a datapipe that yields a 2-tuple [node_path, edge_path] of csv paths. NOTE: PQ ARE IN MW/MVAR"
    def __init__(self, source_dp, length=25000):
        super().__init__()
        self.dp = source_dp
        self.node_dtype = np.dtype([
            ('index', np.int32),
            ('type', np.int32),
            ('vm', np.float32),
            ('va', np.float32),
            ('p', np.float32),
            ('q', np.float32),
        ])
        self.edge_dtype = np.dtype([
            ('from', np.int32),
            ('to', np.int32),
            ('r', np.float32),
            ('x', np.float32),
        ])
        self.sn_mva_dtype = float
        self.length = length
        
    def __len__(self):
        return self.length
        
    def __iter__(self):
        for node_path, edge_path, sn_mva_path in self.dp:
            with open(sn_mva_path, 'r') as f:
                sn_mva = np.loadtxt(f, delimiter=',', dtype=self.sn_mva_dtype, comments=None, skiprows=1)
            with open(node_path, 'r') as f:
                node_array = np.loadtxt(f, delimiter=',', dtype=self.node_dtype, comments=None, skiprows=1)
            with open(edge_path, 'r') as f:
                edge_array = np.loadtxt(f, delimiter=',', dtype=self.edge_dtype, comments=None, skiprows=1)
            yield node_array, edge_array, sn_mva
            
@functional_datapipe('create_geometric_data')
class CreateGeometricData(IterDataPipe):
    """ create a torch.geometric.data.Data from node and edge arrays
    :param fill_noise: fill the missing data with noise
    """
    def __init__(self, source_dp, fill_noise:bool=True, random_bus_type:bool=False):
        super().__init__()
        self.dp = source_dp
        self.fill_noise = fill_noise
        self.random_bus_type = random_bus_type
        self.node_mask = {
            'slack': torch.tensor([[1., 1., 0., 0.,]]), # slack
            'pv': torch.tensor([[1., 0., 1., 0.,]]), # pv
            'pq': torch.tensor([[0., 0., 1., 1.,]]), # pq
        } # vm, va, p, q
        
    def apply_mask(self, input, bus_type_name:str):
        if self.fill_noise:
            out = input*self.node_mask[bus_type_name] \
                + torch.randn_like(input) * (1 - self.node_mask[bus_type_name])
        else:
            out = input*self.node_mask[bus_type_name]
            
        return out
    
    def __len__(self):
        return len(self.dp)
        
    def __iter__(self):
        for node_array, edge_array, sn_mva in self.dp:
            if not self.random_bus_type:
                bus_type = torch.from_numpy(node_array['type'].astype(np.int64)).view(-1, 1) # shape: (N, 1)
            else:
                bus_type = torch.randint(3, (node_array.shape[0], 1))
            y = torch.from_numpy(
                np.stack([
                    node_array['vm'].astype(np.float32),
                    node_array['va'].astype(np.float32),
                    node_array['p'].astype(np.float32) / sn_mva,
                    node_array['q'].astype(np.float32) / sn_mva,
                ], axis=1)
            ) # shape: (N, 4)
            is_slack = (bus_type == 0)
            is_pv = (bus_type == 1).view(-1, 1)
            is_pq = (bus_type == 2).view(-1, 1)
            x = torch.where(is_slack, self.apply_mask(y, 'slack'), y) # slack
            x = torch.where(is_pv, self.apply_mask(x, 'pv'), x) # pv
            x = torch.where(is_pq, self.apply_mask(x, 'pq'), x) # pq
            edge_index = torch.from_numpy(
                np.stack([
                    edge_array['from'].astype(np.int64),
                    edge_array['to'].astype(np.int64),
                ], axis=0)
            ) # shape: (2, E)
            edge_attr = torch.from_numpy(
                np.stack([
                    edge_array['r'].astype(np.float32),
                    edge_array['x'].astype(np.float32),
                ], axis=1)
            ) # shape: (E, 2)
            data = Data(x=x, y=y, bus_type=bus_type, edge_index=edge_index, edge_attr=edge_attr)
            yield data
            
@functional_datapipe('instance_normalize')
class InstanceNormalize(IterDataPipe):
    "normalize the node features per feature. "
    def __init__(self, source_dp):
        super().__init__()
        self.dp = source_dp
        _data = next(iter(self.dp))
        self.node_mean = _data.y.mean(dim=0, keepdim=True)
        self.node_std = _data.y.std(dim=0, keepdim=True) + 1e-6
        self.edge_mean = _data.edge_attr.mean(dim=0, keepdim=True)
        self.edge_std = _data.edge_attr.std(dim=0, keepdim=True) + 1e-6
        
    def __len__(self):
        return len(self.dp)
    
    def normalize_node(self, x):
        if self.node_mean is None:
            self.node_mean = x.mean(dim=0, keepdim=True)
            self.node_std = x.std(dim=0, keepdim=True) + 1e-6
        return (x - self.node_mean.to(x.device)) / self.node_std.to(x.device)
    
    def normalize_edge(self, edge_attr):
        if self.edge_mean is None:
            self.edge_mean = edge_attr.mean(dim=0, keepdim=True)
            self.edge_std = edge_attr.std(dim=0, keepdim=True) + 1e-6
        return (edge_attr - self.edge_mean.to(edge_attr.device)) / self.edge_std.to(edge_attr.device)
    
    def denormalize_node(self, x):
        if self.node_mean is None:
            print('denorm node: mean and std not found. return original data')
            return x
        return x * self.node_std.to(x.device) + self.node_mean.to(x.device)
    
    def denormalize_edge(self, edge_attr):
        if self.edge_mean is None:
            print('denorm edge: mean and std not found. return original data')
            return edge_attr
        return edge_attr * self.edge_std.to(edge_attr.device) + self.edge_mean.to(edge_attr.device)
    
    def normalize(self, data):
        if self.node_mean is None:
            print('normalize: mean and std not found. return original data')
            return data
        data.x = self.normalize_node(data.x)
        data.y = self.normalize_node(data.y)
        data.edge_attr = self.normalize_edge(data.edge_attr)
        return data
    
    def denormalize(self, data):
        if self.node_mean is None:
            print('denormalize: mean and std not found. return original data')
            return data
        data.x = self.denormalize_node(data.x)
        data.y = self.denormalize_node(data.y)
        data.edge_attr = self.denormalize_edge(data.edge_attr)
        return data
        
    def __iter__(self):
        for data in self.dp:
            if self.node_mean is None:
                print('******************cal mean and std******************')
                self.node_mean = data.y.mean(dim=0, keepdim=True)
                self.node_std = data.y.std(dim=0, keepdim=True) + 1e-6
                self.edge_mean = data.edge_attr.mean(dim=0, keepdim=True)
                self.edge_std = data.edge_attr.std(dim=0, keepdim=True) + 1e-6
            data = self.normalize(data)
            yield data