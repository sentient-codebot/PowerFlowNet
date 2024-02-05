import os
import argparse

import pandapower as pp
import numpy as np

from datasets.generator import PowerFlowData, create_case3, generate_data_parallel, generate_data, save_data

number_of_samples = 20
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
        
    sample_list = generate_data(number_of_samples, base_net_create)
    save_data(np.array(sample_list,dtype=PowerFlowData), 'data/power_flow_dataset/raw', case) # TODO revise the datatype a bit
    pass


if __name__ == '__main__':
    main()