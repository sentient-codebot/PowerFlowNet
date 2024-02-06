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

def get_only_matched_node_edge(node_files, edge_files):
    "get only indices where both node and edge files are present"
    assert len(node_files) == len(edge_files)
    for idx, (node_file, edge_file) in enumerate(zip(node_files, edge_files)):
        if (not os.path.exists(node_file)) or (not os.path.exists(edge_file)):
            node_files.pop(idx)
            edge_files.pop(idx)
            
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
    " source_dp: a datapipe that yields a 2-tuple [node_path, edge_path] of csv paths"
    def __init__(self, source_dp):
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
        
    def __iter__(self):
        for node_path, edge_path in self.dp:
            with open(node_path, 'r') as f:
                node_array = np.loadtxt(f, delimiter=',', dtype=self.node_dtype, comments=None, skiprows=1)
            with open(edge_path, 'r') as f:
                edge_array = np.loadtxt(f, delimiter=',', dtype=self.edge_dtype, comments=None, skiprows=1)
            yield node_array, edge_array
            
@functional_datapipe('create_geometric_data')
class CreateGeometricData(IterDataPipe):
    """ create a torch.geometric.data.Data from node and edge arrays
    :param fill_noise: fill the missing data with noise
    """
    def __init__(self, source_dp, fill_noise:bool=True):
        super().__init__()
        self.dp = source_dp
        self.fill_noise = fill_noise
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
        
    def __iter__(self):
        for node_array, edge_array in self.dp:
            bus_type = torch.from_numpy(node_array['type'].astype(np.int64)).view(-1, 1) # shape: (N, 1)
            y = torch.from_numpy(
                np.stack([
                    node_array['vm'].astype(np.float32),
                    node_array['va'].astype(np.float32),
                    node_array['p'].astype(np.float32),
                    node_array['q'].astype(np.float32),
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