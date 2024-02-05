## Synthetic Power Flow Dataset

This dataset is a part of the work in [PowerFlowNet: Leveraging Message Passing GNNs for Improved Power Flow Approximation](https://arxiv.org/abs/2311.03415). 
This dataset is the calculation results of power flows in three IEEE test systems. The goal of this dataset is to facilitate data-driven researches on power flow. So far there are four cases:
- IEEE 14-bus system
- IEEE 118-bus system
- RTE 6470-bus system [AC Power Flow Data in MATPOWER and QCQP Format: iTesla, RTE Snapshots, and PEGASE](http://arxiv.org/abs/1603.01533)

### File Description

Each case contains the following files:
- node_features_x.npy: the node features needed to start the power flow calculation
- node_features_y.npy: the node features containing the complete system state after the power flow calculation
- edge_features.npy: containing the edge index and edge features
- adjacency.npy: the adjacency matrix of the system

### Data Format

#### Node features

The format of `node_features_x.npy` and `node_features_y.npy` are the same. The unknown values in `node_features_x.npy` are filled with zeros. The node features are in the following order:
- index
- type: 0 for slack bus, 1 for PV bus, 2 for PQ bus
- voltage magnitude: p.u.
- voltage angle: degree
- active power flowing out of the node: p.u.
- reactive power flowing out of the node: p.u.

#### Edge features

The edge features in `edge_features.npy` are in the following order:
- index of the start node
- index of the end node
- resistance: p.u.
- reactance: p.u.
- (unused features. filled with zero)