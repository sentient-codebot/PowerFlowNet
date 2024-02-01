import os
import argparse

import pandapower as pp
import numpy as np

from datasets.generator import create_case3, generate_data_parallel

number_of_samples = 2000
number_of_processes = 10

def argument_parser():
    parser = argparse.ArgumentParser(prog='Power Flow Data Generator', description='')
    parser.add_argument('--case', type=str, default='118', help='e.g. 118, 14, 6470rte')
    parser.add_argument('--num_lines_to_remove', '-r', type=int, default=0, help='Number of lines to remove')
    parser.add_argument('--num_lines_to_add', '-a', type=int, default=0, help='Number of lines to add')
    args = parser.parse_args()
    return args

def main():
    args = argument_parser()
    
    num_lines_to_remove = args.num_lines_to_remove
    num_lines_to_add = args.num_lines_to_add
    case = args.case

    if case == '3':
        base_net_create = create_case3
    elif case == '14':
        base_net_create = pp.networks.case14
    elif case == '118':
        base_net_create = pp.networks.case118
    elif case == '6470rte':
        base_net_create = pp.networks.case6470rte
    else:
        print('Invalid test case.')
        exit()
    if num_lines_to_remove > 0 or num_lines_to_add > 0:
        complete_case_name = 'case' + case + 'perturbed' + f'{num_lines_to_remove:1d}' + 'r' + f'{num_lines_to_add:1d}' + 'a'
        
    base_net = base_net_create()
    base_net.bus['name'] = base_net.bus.index
    print(base_net.bus)
    print(base_net.line)
    
    # Generate data
    # generate_data(number_of_samples)
    edge_features_list, node_features_x_list, node_features_y_list = generate_data_parallel(number_of_samples, number_of_processes, base_net_create, num_lines_to_remove, num_lines_to_add)
    
    # Turn the lists into numpy arrays
    edge_features = np.array(edge_features_list)
    node_features_x = np.array(node_features_x_list)
    node_features_y = np.array(node_features_y_list)
    # graph_features = np.array(graph_feature_list)

    # Print the shapes
    # print(f'Adjacency matrix shape: {A.shape}')
    print(f'edge_features shape: {edge_features.shape}')
    print(f'node_features_x shape: {node_features_x.shape}')
    print(f'node_features_y shape: {node_features_y.shape}')
    # print(f'graph_features shape: {graph_features.shape}')

    print(f'range of edge_features "from": {np.min(edge_features[:,:,0])} - {np.max(edge_features[:,:,0])}')
    print(f'range of edge_features "to": {np.min(edge_features[:,:,1])} - {np.max(edge_features[:,:,1])}')

    print(f'range of node_features_x "index": {np.min(node_features_x[:,:,0])} - {np.max(node_features_x[:,:,0])}')

    print(f'range of node_features_y "index": {np.min(node_features_y[:,:,0])} - {np.max(node_features_y[:,:,0])}')

    # print(f"A. {A}")
    # print(f"edge_features. {edge_features}")
    # print(f"node_features_x. {node_features_x}")
    # print(f"node_features_y. {node_features_y}")

    # save the features
    os.makedirs("./data/raw", exist_ok=True)
    with open("./data/raw/"+complete_case_name+"_edge_features.npy", 'wb') as f:
        np.save(f, edge_features)

    with open("./data/raw/"+complete_case_name+"_node_features_x.npy", 'wb') as f:
        np.save(f, node_features_x)

    with open("./data/raw/"+complete_case_name+"_node_features_y.npy", 'wb') as f:
        np.save(f, node_features_y)

    # with open("./data/"+test_case+"_graph_features.npy", 'wb') as f:
    #     np.save(f, graph_features)

    # with open("./data/raw/"+test_case+"_adjacency_matrix.npy", 'wb') as f:
    #     np.save(f, A)
    


if __name__ == '__main__':
    main()