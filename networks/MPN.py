from typing import Annotated as Float
from typing import Annotated as Int

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing, TAGConv, GCNConv, ChebConv
from torch_geometric.utils import degree

def copy_one_way_edges(edge_index, edge_attr):
    edge_index_reverse = torch.stack(
        [edge_index[1,:], edge_index[0,:]],
        dim = 0
    )   # (2, E)
    edge_index_reverse = edge_index_reverse[:, edge_index[0,:] != edge_index[1,:]] # remove self-loops, (2, E-N)
    edge_attr_reverse = edge_attr[edge_index[0,:] != edge_index[1,:]] # (E-N, 2)
    modified_edge_index = torch.cat(
        [edge_index, edge_index_reverse],
        dim = 1
    ) # (2, 2*E-N)
    modified_edge_attr = torch.cat(
        [edge_attr, edge_attr_reverse],
        dim = 0
    ) # (2*E-N, 2)
    return modified_edge_index, modified_edge_attr

class EdgeAggregation(MessagePassing):
    """MessagePassing for aggregating edge features

    """
    def __init__(self, nfeature_dim, efeature_dim, hidden_dim, output_dim):
        super().__init__(aggr='add')
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim

        # self.linear = nn.Linear(nfeature_dim, output_dim) 
        self.edge_aggr = nn.Sequential(
            nn.Linear(nfeature_dim*2 + efeature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def message(self, x_i, x_j, edge_attr):
        '''
        x_j:        shape (N, nfeature_dim,)
        edge_attr:  shape (N, efeature_dim,)
        '''
        return self.edge_aggr(torch.cat([x_i, x_j, edge_attr], dim=-1)) # PNAConv style
    
    def forward(self, x, edge_index, edge_attr):
        '''
        input:
            x:          shape (N, num_nodes, nfeature_dim,)
            edge_attr:  shape (N, num_edges, efeature_dim,)
            
        output:
            out:        shape (N, num_nodes, output_dim,)
        '''
        # Step 1: Add edges in reverse direction to allow two-way message passing
        edge_index, edge_attr = copy_one_way_edges(edge_index, edge_attr)
        
        # Step 2: Calculate the degree of each node.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] 
        
        # Step 3: Feature transformation. 
        # x = self.linear(x) # no feature transformation
        
        # Step 4: Propagation
        out = self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr, norm=norm)
        #   no bias here
        
        return out

class MPN(nn.Module):
    """Wrapped Message Passing Network
        - One-time Message Passing to aggregate edge features into node features
        - Multiple Conv layers
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)
        self.convs = nn.ModuleList()

        if n_gnn_layers == 1:
            self.convs.append(TAGConv(hidden_dim, output_dim, K=K))
        else:
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))

        for l in range(n_gnn_layers-2):
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
            
        self.convs.append(TAGConv(hidden_dim, output_dim, K=K))

    def is_directed(self, edge_index):
        'determine if a graph id directed by reading only one edge'
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack(
                [edge_index[1,:], edge_index[0,:]],
                dim = 0
            )   # (2, E)
            edge_index = torch.cat(
                [edge_index, edge_index_dup],
                dim = 1
            )   # (2, 2*E)
            edge_attr = torch.cat(
                [edge_attr, edge_attr],
                dim = 0
            )   # (2*E, fe)
            
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    
    def forward(self, data):
        assert data.x.shape[-1] == self.nfeature_dim * 2 + 4 # features and their mask + one-hot node type embedding
        x = data.x[:, 4:4+self.nfeature_dim] # first four features: node type. not elegant at all this way. just saying. 
        input_x = x # problem if there is inplace operation on x, so pay attention
        mask = data.x[:, -self.nfeature_dim:]# last few dimensions: mask.
        edge_index = data.edge_index
        edge_features = data.edge_attr
        
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)
        
        x = self.edge_aggr(x, edge_index, edge_features)
        for i in range(len(self.convs)-1):
            # x = self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_attr)
            x = self.convs[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        # x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
        x = self.convs[-1](x=x, edge_index=edge_index)
        
        return x
    
class SkipMPN(nn.Module):
    """Wrapped Message Passing Network
        - * Added skip connection
        - One-time Message Passing to aggregate edge features into node features
        - Multiple Conv layers
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)
        self.convs = nn.ModuleList()

        if n_gnn_layers == 1:
            self.convs.append(TAGConv(hidden_dim, output_dim, K=K))
        else:
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))

        for l in range(n_gnn_layers-2):
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
            
        self.convs.append(TAGConv(hidden_dim, output_dim, K=K))

    def is_directed(self, edge_index):
        'determine if a graph id directed by reading only one edge'
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack(
                [edge_index[1,:], edge_index[0,:]],
                dim = 0
            )   # (2, E)
            edge_index = torch.cat(
                [edge_index, edge_index_dup],
                dim = 1
            )   # (2, 2*E)
            edge_attr = torch.cat(
                [edge_attr, edge_attr],
                dim = 0
            )   # (2*E, fe)
            
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    
    def forward(self, data):
        assert data.x.shape[-1] == self.nfeature_dim * 2 + 4 # features and their mask + one-hot node type embedding
        x = data.x[:, 4:4+self.nfeature_dim] # first four features: node type. not elegant at all this way. just saying. 
        input_x = x # problem if there is inplace operation on x, so pay attention
        mask = data.x[:, -self.nfeature_dim:]# last few dimensions: mask.
        edge_index = data.edge_index
        edge_features = data.edge_attr
        
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)
        
        x = self.edge_aggr(x, edge_index, edge_features)
        for i in range(len(self.convs)-1):
            # x = self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_attr)
            x = self.convs[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        # x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
        x = self.convs[-1](x=x, edge_index=edge_index)
        
        # skip connection
        x = input_x + x
        
        return x
    
class MaskEmbdMPN(nn.Module):
    """Wrapped Message Passing Network
        - * Added embedding for mask
        - One-time Message Passing to aggregate edge features into node features
        - Multiple Conv layers
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)
        self.convs = nn.ModuleList()

        if n_gnn_layers == 1:
            self.convs.append(TAGConv(hidden_dim, output_dim, K=K))
        else:
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))

        for l in range(n_gnn_layers-2):
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
            
        self.convs.append(TAGConv(hidden_dim, output_dim, K=K))
        
        self.mask_embd = nn.Sequential(
            nn.Linear(nfeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nfeature_dim)
        )

    def is_directed(self, edge_index):
        'determine if a graph id directed by reading only one edge'
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack(
                [edge_index[1,:], edge_index[0,:]],
                dim = 0
            )   # (2, E)
            edge_index = torch.cat(
                [edge_index, edge_index_dup],
                dim = 1
            )   # (2, 2*E)
            edge_attr = torch.cat(
                [edge_attr, edge_attr],
                dim = 0
            )   # (2*E, fe)
            
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    
    def forward(self, data):
        assert data.x.shape[-1] == self.nfeature_dim * 2 + 4 # features and their mask + one-hot node type embedding
        x = data.x[:, 4:4+self.nfeature_dim] # first four features: node type. not elegant at all this way. just saying. 
        input_x = x # problem if there is inplace operation on x, so pay attention
        mask = data.x[:, -self.nfeature_dim:]# last few dimensions: mask.
        
        x = self.mask_embd(mask) + x
        
        edge_index = data.edge_index
        edge_features = data.edge_attr
        
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)
        
        x = self.edge_aggr(x, edge_index, edge_features)
        for i in range(len(self.convs)-1):
            # x = self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_attr)
            x = self.convs[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        # x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
        x = self.convs[-1](x=x, edge_index=edge_index)
        
        return x


class MultiMPN(nn.Module):
    """Wrapped Message Passing Network
        - Multi-step mixed MP+Conv
        - No convolution layers
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        # self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)
        # self.convs = nn.ModuleList()
        self.layers = nn.ModuleList()

        if n_gnn_layers == 1:
            self.layers.append(EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, output_dim, K=K))
        else:
            self.layers.append(EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, hidden_dim, K=K))

        for l in range(n_gnn_layers-2):
            self.layers.append(EdgeAggregation(hidden_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, hidden_dim, K=K))
            
        # self.layers.append(TAGConv(hidden_dim, output_dim, K=K))
        self.layers.append(EdgeAggregation(hidden_dim, efeature_dim, hidden_dim, output_dim))

    def is_directed(self, edge_index):
        'determine if a graph id directed by reading only one edge'
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack(
                [edge_index[1,:], edge_index[0,:]],
                dim = 0
            )   # (2, E)
            edge_index = torch.cat(
                [edge_index, edge_index_dup],
                dim = 1
            )   # (2, 2*E)
            edge_attr = torch.cat(
                [edge_attr, edge_attr],
                dim = 0
            )   # (2*E, fe)
            
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    
    def forward(self, data):
        assert data.x.shape[-1] == self.nfeature_dim * 2 + 4 # features and their mask + one-hot node type embedding
        x = data.x[:, 4:4+self.nfeature_dim] # first four features: node type. not elegant at all this way. just saying. 
        input_x = x # problem if there is inplace operation on x, so pay attention
        mask = data.x[:, -self.nfeature_dim:]# last few dimensions: mask.
        edge_index = data.edge_index
        edge_features = data.edge_attr
        
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)
        
        for i in range(len(self.layers)-1):
            if isinstance(self.layers[i], EdgeAggregation):
                x = self.layers[i](x=x, edge_index=edge_index, edge_attr=edge_features)
            else:
                x = self.layers[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        # x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
        if isinstance(self.layers[-1], EdgeAggregation):
            x = self.layers[-1](x=x, edge_index=edge_index, edge_attr=edge_features)
        else:
            x = self.layers[-1](x=x, edge_index=edge_index)
        
        return x


class MaskEmbdMultiMPN(nn.Module):
    """Wrapped Message Passing Network
        - Mask Embedding
        - Multi-step mixed MP+Conv
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        # self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)
        # self.convs = nn.ModuleList()
        self.layers = nn.ModuleList()

        if n_gnn_layers == 1:
            self.layers.append(EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, output_dim, K=K))
        else:
            self.layers.append(EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, hidden_dim, K=K))

        for l in range(n_gnn_layers-2):
            self.layers.append(EdgeAggregation(hidden_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, hidden_dim, K=K))
            
        # self.layers.append(TAGConv(hidden_dim, output_dim, K=K))
        self.layers.append(EdgeAggregation(hidden_dim, efeature_dim, hidden_dim, output_dim))
        
        self.mask_embd = nn.Sequential(
            nn.Linear(nfeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nfeature_dim)
        )

    def is_directed(self, edge_index):
        'determine if a graph id directed by reading only one edge'
        if edge_index.shape[1] == 0:
            # no edge at all, only single nodes. automatically undirected
            return False
        # next line: if there is the reverse of the first edge does not exist, then directed. 
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack(
                [edge_index[1,:], edge_index[0,:]],
                dim = 0
            )   # (2, E)
            edge_index = torch.cat(
                [edge_index, edge_index_dup],
                dim = 1
            )   # (2, 2*E)
            edge_attr = torch.cat(
                [edge_attr, edge_attr],
                dim = 0
            )   # (2*E, fe)
            
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    
    def forward(self, data):
        assert data.x.shape[-1] == self.nfeature_dim * 2 + 4 # features and their mask + one-hot node type embedding
        x = data.x[:, 4:4+self.nfeature_dim] # first four features: node type. not elegant at all this way. just saying. 
        input_x = x # problem if there is inplace operation on x, so pay attention
        mask = data.x[:, -self.nfeature_dim:]# last few dimensions: mask.
        edge_index = data.edge_index
        edge_features = data.edge_attr
                
        x = self.mask_embd(mask) + x
        
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)

        for i in range(len(self.layers)-1):
            if isinstance(self.layers[i], EdgeAggregation):
                x = self.layers[i](x=x, edge_index=edge_index, edge_attr=edge_features)
            else:
                x = self.layers[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        # x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
        if isinstance(self.layers[-1], EdgeAggregation):
            x = self.layers[-1](x=x, edge_index=edge_index, edge_attr=edge_features)
        else:
            x = self.layers[-1](x=x, edge_index=edge_index)
        
        return x
    
    
class MaskEmbdMultiMPN_NoMP(nn.Module):
    """Wrapped Message Passing Network
        - Mask Embedding
        - Multi-step mixed MP+Conv
        - No convolution layers
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        # self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)
        # self.convs = nn.ModuleList()
        self.layers = nn.ModuleList()

        if n_gnn_layers == 1:
            # self.layers.append(EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, output_dim, K=K))
        else:
            # self.layers.append(EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, hidden_dim, K=K))

        for l in range(n_gnn_layers-2):
            # self.layers.append(EdgeAggregation(hidden_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, hidden_dim, K=K))
            
        # self.layers.append(TAGConv(hidden_dim, output_dim, K=K))
        self.layers.append(EdgeAggregation(hidden_dim, efeature_dim, hidden_dim, output_dim))
        
        self.mask_embd = nn.Sequential(
            nn.Linear(nfeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nfeature_dim)
        )

    def is_directed(self, edge_index):
        'determine if a graph id directed by reading only one edge'
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack(
                [edge_index[1,:], edge_index[0,:]],
                dim = 0
            )   # (2, E)
            edge_index = torch.cat(
                [edge_index, edge_index_dup],
                dim = 1
            )   # (2, 2*E)
            edge_attr = torch.cat(
                [edge_attr, edge_attr],
                dim = 0
            )   # (2*E, fe)
            
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    
    def forward(self, data):
        assert data.x.shape[-1] == self.nfeature_dim * 2 + 4 # features and their mask + one-hot node type embedding
        x = data.x[:, 4:4+self.nfeature_dim] # first four features: node type. not elegant at all this way. just saying. 
        input_x = x # problem if there is inplace operation on x, so pay attention
        mask = data.x[:, -self.nfeature_dim:]# last few dimensions: mask.
        edge_index = data.edge_index
        edge_features = data.edge_attr
                
        x = self.mask_embd(mask) + x
        
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)

        for i in range(len(self.layers)-1):
            if isinstance(self.layers[i], EdgeAggregation):
                x = self.layers[i](x=x, edge_index=edge_index, edge_attr=edge_features)
            else:
                x = self.layers[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        # x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
        if isinstance(self.layers[-1], EdgeAggregation):
            x = self.layers[-1](x=x, edge_index=edge_index, edge_attr=edge_features)
        else:
            x = self.layers[-1](x=x, edge_index=edge_index)
        
        return x

class WrappedMultiConv(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels, K, **kwargs):
        super().__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            self.convs.append(ChebConv(in_channels, out_channels, K, normalization=None, **kwargs))
        
    def forward(self, x, edge_index_list, edge_weights_list):
        out = 0.
        for i in range(self.num_convs):
            edge_index = edge_index_list[i]
            edge_weights = edge_weights_list[i]
            out += self.convs[i](x, edge_index, edge_weights)

        return out

class MultiConvNet(nn.Module):
    """Wrapped Message Passing Network
        - No Message Passing to aggregate edge features into node features
        - Multi-level parallel Conv layers for different edge features
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        assert efeature_dim == 5
        efeature_dim = efeature_dim - 3 # should be 2, only these two are meaningful
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.edge_trans = nn.Sequential(
            nn.Linear(efeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, efeature_dim)
        )
        self.convs = nn.ModuleList()

        if n_gnn_layers == 1:
            self.convs.append(WrappedMultiConv(efeature_dim, nfeature_dim, output_dim, K=K))
        else:
            self.convs.append(WrappedMultiConv(efeature_dim, nfeature_dim, hidden_dim, K=K))

        for l in range(n_gnn_layers-2):
            self.convs.append(WrappedMultiConv(efeature_dim, hidden_dim, hidden_dim, K=K))
            
        self.convs.append(WrappedMultiConv(efeature_dim, hidden_dim, output_dim, K=K))

    def is_directed(self, edge_index):
        'determine if a graph id directed by reading only one edge'
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack(
                [edge_index[1,:], edge_index[0,:]],
                dim = 0
            )   # (2, E)
            edge_index = torch.cat(
                [edge_index, edge_index_dup],
                dim = 1
            )   # (2, 2*E)
            edge_attr = torch.cat(
                [edge_attr, edge_attr],
                dim = 0
            )   # (2*E, fe)
            
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    
    def forward(self, data):
        assert data.x.shape[-1] == self.nfeature_dim * 2 + 4 # features and their mask + one-hot node type embedding
        x = data.x[:, 4:4+self.nfeature_dim] # first four features: node type. not elegant at all this way. just saying. 
        input_x = x # problem if there is inplace operation on x, so pay attention
        mask = data.x[:, -self.nfeature_dim:]# last few dimensions: mask.
        edge_index = data.edge_index
        edge_features = data.edge_attr
        
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)
        
        edge_features = edge_features[:, :2] + self.edge_trans(edge_features[:, :2]) # only take the first two meaningful features
        for i in range(len(self.convs)-1):
            x = self.convs[i](x=x, 
                              edge_index_list=[edge_index]*self.efeature_dim,
                              edge_weights_list=[edge_features[:,i] for i in range(self.efeature_dim)])
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        # x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
        x = self.convs[-1](x=x, 
                              edge_index_list=[edge_index]*self.efeature_dim,
                              edge_weights_list=[edge_features[:,i] for i in range(self.efeature_dim)])
        
        return x
    
    
class MPN_simplenet(nn.Module):
    """Wrapped Message Passing Network
        - One-time Message Passing to aggregate edge features into node features
        - Multiple Conv layers
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)
        self.convs = nn.ModuleList()

        if n_gnn_layers == 1:
            self.convs.append(TAGConv(hidden_dim, output_dim, K=K))
        else:
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))

        for l in range(n_gnn_layers-2):
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
            
        self.convs.append(TAGConv(hidden_dim, output_dim, K=K))

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_features = data.edge_attr
        
        x = self.edge_aggr(x, edge_index, edge_features)
        for i in range(len(self.convs)-1):
            x = self.convs[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        x = self.convs[-1](x=x, edge_index=edge_index)
        
        return x
    
class BusTypeEncoder(nn.Module):
    "encoder a bus type (integer) into a continous embedding"
    def __init__(self, num_bus_types:int=3, embd_dim:int=32):
        super().__init__()
        self.num_bus_types = num_bus_types
        self.embd_dim = embd_dim
        self.embd = nn.Embedding(num_bus_types, embd_dim)
        
    def forward(self, bus_type: Int[Tensor, 'B*N 1']) -> Float[Tensor, 'B*N D']:
        return self.embd(bus_type.long()).squeeze(1)
    
class GroupLinear(nn.Module):
    " split (N, D) into (N, D//k) to perform linear separately and then concatenate"
    def __init__(self, in_features:int, out_features:int, num_group:int=4):
        super().__init__()
        assert in_features % num_group == 0 and out_features % num_group == 0
        self.in_features = in_features
        self.out_features = out_features
        self.num_group = num_group
        self.linear = nn.ModuleList([nn.Linear(in_features//num_group, out_features//num_group) for _ in range(num_group)])
        
    def forward(self, x: Float[Tensor, 'N D']) -> Float[Tensor, 'N D']:
        x_split = torch.split(x, self.in_features//self.num_group, dim=-1) # list of (N, D//k)
        x_out = torch.cat([linear(x) for linear, x in zip(self.linear, x_split)], dim=-1) # (N, D)
        
        return x_out
    
class MaskEmbdMultiMPNV2(nn.Module):
    """Wrapped Message Passing Network [version 2. tiny adaptations for data generation v3]
        - Mask Embedding
        - Multi-step mixed MP+Conv
    """
    def __init__(self, in_channels_node, in_channels_edge, out_channels_node, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.in_channels_node = in_channels_node
        self.in_channels_edge = in_channels_edge
        self.out_channels_node = out_channels_node
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        
        self.init_groupnorm = nn.GroupNorm(in_channels_node, in_channels_node) # instance norm, norm in each channel individually
        self.mid_layers = nn.ModuleList()

        self.init_conv = nn.ModuleDict({
            'mp': EdgeAggregationV2(in_channels_node, in_channels_edge, hidden_dim, hidden_dim),
            'conv': TAGConv(hidden_dim, hidden_dim, K=K)
        })

        for l in range(n_gnn_layers-2):
            self.mid_layers.append(nn.ModuleDict({
                'bus_modulation_1': nn.Sequential(
                    nn.Linear(in_channels_node, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 2*hidden_dim)
                ),
                'bus_modulation_2': nn.Sequential(
                    nn.Linear(in_channels_node, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 2*hidden_dim)
                ),
                'ln1': nn.LayerNorm(hidden_dim, elementwise_affine=False),
                'ln2': nn.LayerNorm(hidden_dim, elementwise_affine=False),
                'mp': EdgeAggregationV2(hidden_dim, in_channels_edge, hidden_dim, hidden_dim),
                'conv': TAGConv(hidden_dim, hidden_dim, K=K),
                'mlp': nn.Sequential(
                    nn.Linear(hidden_dim, 4*hidden_dim),
                    nn.SiLU(),
                    nn.Linear(4*hidden_dim, hidden_dim)
                )
            }))
            
        # self.final_mp = EdgeAggregation(hidden_dim, in_channels_edge, hidden_dim, out_channels_node)
        # self.final_mlp = nn.Sequential(
        #     nn.Linear(hidden_dim, 4*hidden_dim),
        #     nn.SiLU(),
        #     nn.Linear(4*hidden_dim, out_channels_node)
        # )
        self.final_mlp = nn.Sequential(
            GroupLinear(hidden_dim, 4*hidden_dim, 4),
            nn.SiLU(),
            GroupLinear(4*hidden_dim, out_channels_node, 4)
        )
        
        self.bus_type_encoder = BusTypeEncoder(3, in_channels_node) # do i need an mlp to further process it? perhaps not since it's already a learned embedding.

    def is_directed(self, edge_index):
        'determine if a graph id directed by reading only one edge'
        if edge_index.shape[1] == 0:
            # no edge at all, only single nodes. automatically undirected
            return False
        # next line: if there is the reverse of the first edge does not exist, then directed. 
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack(
                [edge_index[1,:], edge_index[0,:]],
                dim = 0
            )   # (2, E)
            edge_index = torch.cat(
                [edge_index, edge_index_dup],
                dim = 1
            )   # (2, 2*E)
            edge_attr = torch.cat(
                [edge_attr, edge_attr],
                dim = 0
            )   # (2*E, fe)
            
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    
    def forward(self, data):
        x = data.x # shape (B*N, in_channels_node), usually in_channels_node = 4
        bus_type = data.bus_type # shape (B*N, 1)
        edge_index = data.edge_index
        edge_features = data.edge_attr
        
        # encode x with bus type
        encoded_bus_type = self.bus_type_encoder(bus_type) # shape (B*N, in_channels_node)
        x = x + encoded_bus_type
        x_init = x
        x = self.init_groupnorm(x)
        
        # init conv
        x = self.init_conv['mp'](x, edge_index, edge_features)
        x = self.init_conv['conv'](x, edge_index)
        
        # make graph undirected -> edge aggr does this internally
        # edge_index, edge_features = self.undirect_graph(edge_index, edge_features)

        # message passing and conv
        for layer in self.mid_layers:
            mp = layer['mp']
            conv = layer['conv']
            ln1 = layer['ln1']
            ln2 = layer['ln2']
            mlp = layer['mlp']
            x_copy = x
            x = ln1(x) # shape (B*N, hidden_dim)
            scale, shift = layer['bus_modulation_1'](encoded_bus_type).chunk(2, dim=-1)
            x = x * (1.+scale) + shift
            x = mp(x=x, edge_index=edge_index, edge_attr=edge_features)
            x = nn.SiLU()(x)
            x = conv(x=x, edge_index=edge_index)
            x = x + x_copy
            x_copy = x
            x = ln2(x)
            scale, shift = layer['bus_modulation_2'](encoded_bus_type).chunk(2, dim=-1)
            x = x * (1.+scale) + shift
            x = mlp(x) + x_copy
            x = nn.Dropout(self.dropout_rate)(x)
        
        # final message passing
        # x = self.final_mp(x, edge_index, edge_features) + x_init
        x = self.final_mlp(x) + x_init
        
        return x
    
class EdgeAggregationV2(MessagePassing):
    """MessagePassing for aggregating edge features 
    
    version 2: transforms the edge features before creating message

    """
    def __init__(self, nfeature_dim, efeature_dim, hidden_dim, output_dim):
        super().__init__(aggr='add')
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        
        self.edge_transformation = nn.Sequential(
            nn.Linear(efeature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ) # (E, Fe) -> (E, hidden_dim)
        self.edge_aggr = nn.Sequential(
            nn.Linear(nfeature_dim*2 + hidden_dim, 2*hidden_dim),
            nn.SiLU(),
            nn.Linear(2*hidden_dim, output_dim)
        )
        
    def message(self, x_i, x_j, edge_attr):
        '''
        x_j:        shape (N, nfeature_dim,)
        edge_attr:  shape (N, efeature_dim,)
        '''
        return self.edge_aggr(torch.cat([x_i, x_j, edge_attr], dim=-1)) # PNAConv style
    
    def forward(self, x, edge_index, edge_attr):
        '''
        input:
            x:          shape (N, num_nodes, nfeature_dim,)
            edge_attr:  shape (N, num_edges, efeature_dim,)
            
        output:
            out:        shape (N, num_nodes, output_dim,)
        '''
        # Step 1: Add edges in reverse direction to allow two-way message passing
        edge_index, edge_attr = copy_one_way_edges(edge_index, edge_attr)
        
        # Step 2: Calculate the degree of each node.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] 
        
        # Step 3: Edge transformation. 
        edge_attr = self.edge_transformation(edge_attr) # (E, Fe) -> (E, hidden_dim)
        
        # Step 4: Propagation
        out = self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr, norm=norm)
        #   no bias here
        
        return out

class MultiGCNBlock(nn.Module):
    """building block for MultiGCN
    
    Arguments:
        - dict_gcn_params: dictionary of parameters for each GCN layer
        e.g. {'slack': {'K': 2}, 'pv': {'K': 4}, 'pq': {'K': 6}
    """
    def __init__(self, in_channels_node, in_channels_edge, out_channels_node, hidden_dim,
                 dict_gcn_params):
        super().__init__()
        self.in_channels_node = in_channels_node
        self.in_channels_edge = in_channels_edge
        self.out_channels_node = out_channels_node
        self.hidden_dim = hidden_dim
        self.dict_gcn_params = dict_gcn_params
        self.dict_gcn = nn.ModuleDict()
        for name, params in dict_gcn_params.items():
            self.dict_gcn[name] = nn.ModuleDict()
            self.dict_gcn[name]['mp'] = EdgeAggregationV2(in_channels_node, in_channels_edge, hidden_dim, hidden_dim)
            self.dict_gcn[name]['conv'] = TAGConv(hidden_dim, out_channels_node, K=params['K'])
            
    def forward(self, x, edge_index, edge_attr, bus_type):
        """
        Arguments:
            - x: shape (N, in_channels_node,)
            - edge_index: shape (2, E,)
            - edge_attr: shape (E, in_channels_edge,)
            - bus_type: shape (N, 1,)
        """
        out = {}
        for name, module_dict in self.dict_gcn.items():
            mp = module_dict['mp']
            conv = module_dict['conv']
            out[name] = conv(
                x=nn.SiLU()(mp(x, edge_index, edge_attr)), 
                edge_index=edge_index
            )
            
        weighted_out = out['slack'] * (bus_type == 0) + out['pv'] * (bus_type == 1) + out['pq'] * (bus_type == 2)
        return weighted_out

class TypeSensitiveGCN(nn.Module):
    """Use different GCN layers for different types of buses
    Input:
        - data (actually need only x and bus_type)
    
    Output:
        - x (updated node features)
    """
    def __init__(
        self, 
        in_channels_node=4, 
        in_channels_edge=2,
        out_channels_node=4,
        hidden_dim=128,
        n_gnn_layers=5,
        K=5,
        dropout_rate=0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = n_gnn_layers
        
        self.init_ln = nn.LayerNorm(in_channels_node)
        self.mid_layers = nn.ModuleList()

        self.init_conv = nn.ModuleDict({
            'mp': EdgeAggregationV2(in_channels_node, in_channels_edge, hidden_dim, hidden_dim),
            'conv': TAGConv(hidden_dim, hidden_dim, K=5)
        }) # out x shape (B*N, hidden_dim)

        dict_gcn_params = {
            'slack': {'K': 10},
            'pv': {'K': 3},
            'pq': {'K': 3}
        }
        for l in range(self.num_hidden_layers):
            self.mid_layers.append(nn.ModuleDict({
                'ln1': nn.LayerNorm(hidden_dim),
                'ln2': nn.LayerNorm(hidden_dim),
                'multigcn': MultiGCNBlock(hidden_dim, in_channels_edge, hidden_dim, hidden_dim, dict_gcn_params),
                'mlp': nn.Sequential(
                    nn.Linear(hidden_dim, 4*hidden_dim),
                    nn.SiLU(),
                    nn.Linear(4*hidden_dim, hidden_dim)
                )
            }))
            
        self.final_mp = EdgeAggregationV2(hidden_dim, in_channels_edge, hidden_dim, out_channels_node)
        self.bus_type_encoder = BusTypeEncoder(3, in_channels_node) 
        
    def forward(self, data):
        x = data.x
        bus_type = data.bus_type
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        # encode bus type
        encoded_bus_type = self.bus_type_encoder(bus_type)
        x = self.init_ln(x) + encoded_bus_type
        
        # init conv
        x = self.init_conv['mp'](x, edge_index, edge_attr)
        x = self.init_conv['conv'](x, edge_index)
        
        # mid layers
        for layer in self.mid_layers:
            multigcn = layer['multigcn']
            mlp = layer['mlp']
            ln1 = layer['ln1']
            ln2 = layer['ln2']
            x_copy = x
            x = ln1(x)
            x = multigcn(x, edge_index, edge_attr, bus_type)
            x = x + x_copy
            x_copy = x
            x = mlp(ln2(x)) + x_copy
            x = nn.Dropout(0.1)(x)
            
        # final conv
        x = self.final_mp(x, edge_index, edge_attr)
        
        return x