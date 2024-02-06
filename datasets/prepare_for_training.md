## Prepare Power Flow Dataset for Training GNN

This document will introduce the structure of a sample in the power flow dataset and how to prepare the dataset for training a graph neural network (GNN) model.

### Structure of a Sample

In `torch_geometric`, we work with data objects of `torch_geometric.data.Data` type. This object contains the following attributes, specifically for power flow dataset.
- `y` (target node features) \- a tensor of shape `[num_nodes, 4]`. The prediction target. It contains complete power flow calculation results, i.e., the voltage magnitude, voltage angle, active power and reactive power (flown out of the node), at each node. 
- `x` (input node features) \- a tensor of shape `[num_nodes, 4]`. The input data. This attribute is obtained by masking some features in `y`. Which features to mask for each node depends on the bus type. The masked value can either 0. or standard normal noise depending on the choice of the `fill_noise` argument when preparing the dataset.
- `edge_index` (edge index) \- a tensor of shape `[2, num_edges]`. It contains the indices of the nodes that are connected by an edge. This list includes self-loops at every node because it is created from the **nodal admittance matrix**.
- `edge_attr` (edge features) \- a tensor of shape `[num_edges, 2]`. Parameters of power lines. The features are, namely, resistance, reactance.
- `bus_type` (bus type) \- a tensor of shape `[num_nodes, 1]`. It contains the bus type of each node. The bus type is encoded as 0, 1, 2, respectively, for slack, PV and PQ buses. This bus type will decide which nodes and features of `y` are masked to get `x`. 


### Prepare Dataset for Training

**Before we begin.** The recommended way to prepare the training dataset uses `torchdata` package. The steps are as follows. 

**1. import helper functions**

```python
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from datasets.power_flow_data import create_pf_dp, create_batch_dp
```

**2. create a dataset that you need** Suppose the dataset root directory is `data/power_flow_dataset`. This means, for example, case 14, is stored in `data/power_flow_dataset/raw/case14`. Make sure the files are stored in the correct location. To prepare the dataset of case 14, you can use the following code.

```python
dp = create_pf_dp(root='data/power_flow_dataset', case='14', task='train', fill_noise=True)
```

The root directory, case and task are mandatory arguments. The `fill_noise` argument is optional. If `fill_noise` is `True`, the unknown node features (such as voltage of a PQ node) will be filled with noise (standard normal). 

**3. create a batch and a dataloader** Now the previous `dp` can iterated to give us one scenario at a time. Each scenario contains a graph and its associated node and edge features. To train our model efficiently, we will shuffle and collate multiple scenarios into a batch and merge them into a giant graph. This will be done automatically with the helper functions. You only need to specify the batch size and the number of workers.

```python
batch_dp = create_batch_dp(dp, batch_size=32)
rs = MultiProcessingReadingService(num_workers=4)
train_loader = DataLoader2(batch_dp, reading_service=rs)
```

Or you can use a wrapped function.

```python
from datasets.power_flow_data import create_dataloader
batch_dp = create_batch_dp(dp, batch_size=32)
train_loader = create_dataloader(batch_dp, num_workers=4)
```

Now the `train_loader` can be used to train the GNN model.