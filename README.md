# PoweFlowNet

Dataset link: https://tud365-my.sharepoint.com/:f:/g/personal/sorfanoudakis_tudelft_nl/EmWGZcpct51Gp2np1Zbv7NEBdANZCyFMlD7Iiyamp2_ztw?e=m4dNyK

Trained models link [temporary]: https://tud365-my.sharepoint.com/personal/nlin2_tudelft_nl/_layouts/15/onedrive.aspx?ct=1691140898356&or=Teams%2DHL&ga=1&LOF=1&id=%2Fpersonal%2Fnlin2%5Ftudelft%5Fnl%2FDocuments%2FProjects%2FPoweFlowNet&view=0

## Description

## Installation

# File Structure
runnable files:
- `train.py` trains the model
- `results.py` plots the results

# Useful Information
First two dimensions out of seven in `edge_features` are `from_node` and `to_node`, and they are indexed from $1$. This is processed in the `PowerFlowData` dataset class. It is reindexed from $0$ and the `from_node` and `to_node` are removed from the `edge_features` tensor.

Raw data format: 
| Number | Description |
| --- | --- |
| N | number of nodes |
| E | number of edges |
| Fn = 9 | number of features per node |
| Fe = 5 | orginally 7, first two dims are `from_node` and `to_node` number of features per edge |
| Fn_out = 8 | number of output features per node |

| Tensor | Dimension |
| --- | --- |
| `Data.x` | (batch_size*N, Fe) |
| `Data.edge_index` | (2, E) |
| `Data.edge_attr` | (E, Fe) |
| `Data.y` | (batch_size*N, Fn) |



**Notes**
Some load nodes are assigned with 0 power. Becaus they are not created through `create_load(bus, ...)`. Consider does this matter? 
