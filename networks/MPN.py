from typing import Annotated as Float
from typing import Annotated as Int

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing, TAGConv, GCNConv, ChebConv
from torch_geometric.utils import degree

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
            nn.ReLU(),
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
        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0)) # no self loop because NO EDGE ATTR FOR SELF LOOP
        
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
        self.layers = nn.ModuleList()

        if n_gnn_layers == 1:
            self.layers.append(EdgeAggregation(in_channels_node, in_channels_edge, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, out_channels_node, K=K))
        else:
            self.layers.append(EdgeAggregation(in_channels_node, in_channels_edge, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, hidden_dim, K=K))

        for l in range(n_gnn_layers-2):
            self.layers.append(EdgeAggregation(hidden_dim, in_channels_edge, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, hidden_dim, K=K))
            
        # self.layers.append(TAGConv(hidden_dim, output_dim, K=K))
        self.layers.append(EdgeAggregation(hidden_dim, in_channels_edge, hidden_dim, out_channels_node))
        
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
        # encode x with bus type
        x = data.x # shape (B*N, in_channels_node), usually in_channels_node = 4
        bus_type = data.bus_type # shape (B*N, 1)
        encoded_bus_type = self.bus_type_encoder(bus_type) # shape (B*N, in_channels_node)
        x = x + encoded_bus_type
        
        edge_index = data.edge_index
        edge_features = data.edge_attr
        
        # make graph undirected -> shouldn't need it now in the data gen v3
        # edge_index, edge_features = self.undirect_graph(edge_index, edge_features)

        # message passing and conv
        for i in range(len(self.layers)-1):
            if isinstance(self.layers[i], EdgeAggregation):
                x = self.layers[i](x=x, edge_index=edge_index, edge_attr=edge_features)
            else:
                x = self.layers[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate)(x)
            x = nn.ReLU()(x)
        
        if isinstance(self.layers[-1], EdgeAggregation):
            x = self.layers[-1](x=x, edge_index=edge_index, edge_attr=edge_features)
        else:
            x = self.layers[-1](x=x, edge_index=edge_index)
        
        return x