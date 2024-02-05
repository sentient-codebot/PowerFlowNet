import os
import argparse

import pandapower as pp
import numpy as np

from datasets.generator import create_case3, generate_data_parallel, generate_data, save_data_csv, compress_csv

def argument_parser():
    parser = argparse.ArgumentParser(prog='Power Flow Data Generator', description='')
    parser.add_argument('--case', type=str, default='118', help='e.g. 118, 14, 6470rte')
    parser.add_argument('--data_root', type=str, default='data/power_flow_dataset/')
    parser.add_argument('--num_samples', '-n', type=int, default=20, help='Number of samples to generate')
    parser.add_argument('--num_processes', '-p', type=int, default=2, help='Number of processes to use')
    parser.add_argument('--num_lines_to_remove', '-r', type=int, default=0, help='Number of lines to remove')
    parser.add_argument('--num_lines_to_add', '-a', type=int, default=0, help='Number of lines to add')
    args = parser.parse_args()
    return args

def main():
    args = argument_parser()
    num_lines_to_remove = args.num_lines_to_remove
    num_lines_to_add = args.num_lines_to_add
    case = args.case
    root = args.data_root
    num_samples = args.num_samples
    num_processes = args.num_processes

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
    # if num_lines_to_remove > 0 or num_lines_to_add > 0:
    #     complete_case_name = 'case' + case + 'perturbed' + f'{num_lines_to_remove:1d}' + 'r' + f'{num_lines_to_add:1d}' + 'a'
        
    sample_list = generate_data_parallel(num_samples, num_processes, base_net_create=base_net_create, max_cont_fails=10)
    save_data_csv(sample_list, os.path.join(root, 'raw'), case) 
    compress_csv(os.path.join(root, 'raw'), case)
    pass


if __name__ == '__main__':
    main()