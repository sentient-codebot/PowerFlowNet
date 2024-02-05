"""this file generates the dataset using pandapower, it is the the 'dataset_generator_pandapower_v2.py' file in the original repository
Date: Feb 2024
Author: Nan Lin, Stavros Orfanoudakis
"""
import csv
import copy
from collections import namedtuple
import warnings
import multiprocessing as mp
import os

import pandas as pd
import pandapower as pp
from pandapower import LoadflowNotConverged
import numpy as np
import networkx as nx
from tqdm import tqdm

PowerFlowData = namedtuple('PowerFlowData', ['node_features', 'edge_features', 'sn_mva'])

def create_case3():
    net = pp.create_empty_network()
    net.sn_mva = 100
    b0 = pp.create_bus(net, vn_kv=345., name='bus 0')
    b1 = pp.create_bus(net, vn_kv=345., name='bus 1')
    b2 = pp.create_bus(net, vn_kv=345., name='bus 2')
    pp.create_ext_grid(net, bus=b0, vm_pu=1.02, name="Grid Connection")
    pp.create_load(net, bus=b2, p_mw=10.3, q_mvar=3, name="Load")
    # pp.create_gen(net, bus=b1, p_mw=0.5, vm_pu=1.03, name="Gen", max_p_mw=1)
    pp.create_line(net, from_bus=b0, to_bus=b1, length_km=10, name='line 01', std_type='NAYY 4x50 SE')
    pp.create_line(net, from_bus=b1, to_bus=b2, length_km=5, name='line 01', std_type='NAYY 4x50 SE')
    pp.create_line(net, from_bus=b2, to_bus=b0, length_km=20, name='line 01', std_type='NAYY 4x50 SE')
    
    net.line['c_nf_per_km'] = pd.Series(0., index=net.line['c_nf_per_km'].index, name=net.line['c_nf_per_km'].name)
    
    return net

def remove_c_nf(net):
    net.line['c_nf_per_km'] = pd.Series(0., index=net.line['c_nf_per_km'].index, name=net.line['c_nf_per_km'].name)
    
def unify_vn(net):
    for node_id in range(net.bus['vn_kv'].shape[0]):
        net.bus['vn_kv'][node_id] = max(net.bus['vn_kv'])

def get_trafo_z_pu(net):
    for trafo_id in net.trafo.index:
        net.trafo['i0_percent'][trafo_id] = 0.
        net.trafo['pfe_kw'][trafo_id] = 0.
    
    z_pu = net.trafo['vk_percent'].values / 100. * 1000. / net.sn_mva
    r_pu = net.trafo['vkr_percent'].values / 100. * 1000. / net.sn_mva
    x_pu = np.sqrt(z_pu**2 - r_pu**2)
    
    return x_pu, r_pu
    
def get_line_z_pu(net):
    r = net.line['r_ohm_per_km'].values * net.line['length_km'].values
    x = net.line['x_ohm_per_km'].values * net.line['length_km'].values
    from_bus = net.line['from_bus']
    to_bus = net.line['to_bus']
    vn_kv_to = net.bus['vn_kv'][to_bus].to_numpy()
    # vn_kv_to = pd.Series(vn_kv_to)
    zn = vn_kv_to**2 / net.sn_mva
    r_pu = r/zn
    x_pu = x/zn
    
    return r_pu, x_pu

def get_adjacency_matrix(net):
    multi_graph = pp.topology.create_nxgraph(net)
    A = nx.adjacency_matrix(multi_graph).todense() 
    
    return A

def perturb_topology(net, num_lines_to_remove=0, num_lines_to_add=0):
    """
    Steps:
        1. load topology
        2. randomly remove lines (<- control: e.g. how many?)
        3. check connectivity
        4. if yes, return; else revert step 2 and retry. 
    """
    if num_lines_to_remove == 0 and num_lines_to_add == 0:
        return 0, net

    max_attempts = 20
    # 1. load topology
    lines_indices = np.array(net.line.index)
    lines_from_bus = net.line['from_bus'].values # from 0, shape (num_lines,)
    lines_to_bus = net.line['to_bus'].values # shape (num_lines,)
    line_numbers = np.arange(start=0, stop=len(lines_from_bus))
    bus_numbers = net.bus.index.values # shape (num_buses,)

    rng = np.random.default_rng()
    # 2. remove lines
    num_disconnected_bus = 1
    num_attempts = 0
    while num_disconnected_bus > 0:
        net_perturbed = copy.deepcopy(net)
        if num_attempts == max_attempts:
            warnings.warn("Could not find a connected graph after {} attempts. Return original graph.".format(max_attempts))
            return 1, net
        to_be_removed = rng.choice(line_numbers, size=num_lines_to_remove, replace=False)
        pp.drop_lines(net_perturbed, lines_indices[to_be_removed])
        num_disconnected_bus = len(pp.topology.unsupplied_buses(net_perturbed))
        num_attempts += 1

    # 3. add lines
    for _ in range(num_lines_to_add):
        from_bus, to_bus = rng.choice(bus_numbers, size=2, replace=False)
        copied_line = net.line.iloc[rng.choice(line_numbers, size=1, replace=False)]
        pp.create_line_from_parameters(
            net_perturbed,
            from_bus,
            to_bus,
            copied_line['length_km'].item(),
            copied_line['r_ohm_per_km'].item(),
            copied_line['x_ohm_per_km'].item(),
            copied_line['c_nf_per_km'].item(),
            copied_line['max_i_ka'].item()
        )

    return 0, net_perturbed

def generate_data(num_samples, base_net_create, max_cont_fails=10, is_sub_process=False):
    sample_list = []
    num_cont_fails = 0 # number of continous failed pf calculations
    num_total_fails = 0 # number of total failed pf calculations

    with tqdm (total=num_samples, disable = is_sub_process) as pbar:
        while len(sample_list) < num_samples:
            if num_cont_fails > max_cont_fails:
                warnings.warn("Too many failed power flow calculations. Return current samples.")
                break
            
            net = base_net_create()
            # remove_c_nf(net)
            
            # success_flag, net = perturb_topology(net, num_lines_to_remove=num_lines_to_remove, num_lines_to_add=num_lines_to_add) # TODO 
            # if success_flag == 1:
            #     exit()
            n = net.bus.values.shape[0]
            # A = get_adjacency_matrix(net)
            
            net.bus['name'] = net.bus.index

            # get params: line
            r = net.line['r_ohm_per_km'].values    
            x = net.line['x_ohm_per_km'].values
            c = net.line['c_nf_per_km'].values
            le = net.line['length_km'].values
            theta_shift_degree = net.trafo['shift_degree'].values # transformer
            # also transformer tap position?

            # get params: bus
            Pg = net.gen['p_mw'].values
            Pd = net.load['p_mw'].values
            Qd = net.load['q_mvar'].values

            # alter params: line
            r = np.random.uniform(0.8*r, 1.2*r, r.shape[0])
            x = np.random.uniform(0.8*x, 1.2*x, x.shape[0])
            c = np.random.uniform(0.8*c, 1.2*c, c.shape[0])
            le = np.random.uniform(0.8*le, 1.2*le, le.shape[0])
            theta_shift_degree = np.random.uniform(-11.46, 11.46, theta_shift_degree.shape[0]) # -0.2 ~ 0.2 rad

            # alter params: bus
            Vg = np.random.uniform(0.95, 1.05, net.gen['vm_pu'].shape[0])
            Pg = np.random.normal(Pg, 0.1*np.abs(Pg), net.gen['p_mw'].shape[0])
            Pd = np.random.uniform(0.5*Pd, 1.5*np.abs(Pd), net.load['p_mw'].shape[0])
            Qd = np.random.uniform(0.5*Qd, 1.5*np.abs(Qd), net.load['q_mvar'].shape[0])
            
            # assign params
            net.line['r_ohm_per_km'] = r 
            net.line['x_ohm_per_km'] = x 
            net.line['c_nf_per_km'] = c
            net.line['length_km'] = le
            net.trafo['shift_degree'] = theta_shift_degree

            net.gen['vm_pu'] = Vg
            net.gen['p_mw'] = Pg
            net.load['p_mw'] = Pd
            net.load['q_mvar'] = Qd

            # Calculate power flow
            net['converged'] = False
            try:
                pp.runpp(net, algorithm='nr', init="results", numba=False)
            except LoadflowNotConverged:
                num_cont_fails += 1
                num_total_fails += 1
                continue
            if net['converged'] == False:
                num_cont_fails += 1
                num_total_fails += 1
                continue
            else:
                num_cont_fails = 0
            
            # Get results
            ybus = net._ppc["internal"]["Ybus"].todense() 

            # Extract edge index and features from ybus
            G = nx.Graph(ybus)
            edge_features_raw = np.array(list(G.edges.data('weight')))
            edge_weights = np.stack([edge_features_raw[:, 2].real, edge_features_raw[:, 2].imag], axis=1) # shape: (num_edges, 2)
            edge_features = np.concatenate([edge_features_raw[:, :2], edge_weights], axis=1) # shape: (num_edges, 4). from, to, r, x

            # Extract node features fron net.res. In total, we need: index, type, Vm, Va, Pd, Qd
            node_results = net.res_bus.values # shape: (num_nodes, 4)
            node_index = np.arange(len(node_results)) # shape: (num_nodes, )
            node_type = np.array([2.] * len(node_results)) # slack, generator, load: 0, 1, 2. shape: (num_nodes, )
            for idx in net.gen['bus'].values:
                node_type[idx] = 1. # this must be before ext_grid, because ext_grid is also a generator
            for idx in net.ext_grid['bus'].values:
                node_type[idx] = 0.
            node_features = np.stack([node_index, node_type], axis=1) # shape: (num_nodes, 2)
            node_features = np.concatenate([node_features, node_results], axis=1) # shape: (num_nodes, 6)
            node_features = node_features.real # shape: (num_nodes, 6)
            
            # Append to list
            sample_list.append(PowerFlowData(node_features, edge_features, net.sn_mva))
            
            # pbar
            pbar.set_description(f'#Success: {len(sample_list)}/{num_samples}, #Fails: {num_total_fails}, #Cont.Fails: {num_cont_fails}')
            pbar.update(1)
            
    return sample_list

def save_data(sample_array: np.ndarray[PowerFlowData], save_dir: str, case_name: str):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{case_name}_samples.npz')
    np.savez_compressed(save_path, sample_array)

def generate_data_parallel(num_samples, num_processes, *generation_args):
    sublist_size = num_samples // num_processes
    pool = mp.Pool(processes=num_processes)
    args = generation_args
    full_args = [sublist_size, *args]*num_processes
    results = pool.map(generate_data, full_args)
    pool.close()
    pool.join()
    
    edge_features_list = []
    node_features_x_list = []
    node_features_y_list = []
    for sub_res in results:
        edge_features_list += sub_res[0]
        node_features_x_list += sub_res[1]
        node_features_y_list += sub_res[2]
        
    return edge_features_list, node_features_x_list, node_features_y_list

def newton_raphson():
    raise NotImplementedError # so much work. i give up for now. 

def validate_results(net):
    " validate the pp NR PF results and the results of the custom newton-raphson iterations. "
    # pandapower pf
    pp.runpp(net, numba = False)
    bus_result_pp = net.res_bus
    line_result_pp = net.res_line
    Ybus = net._ppc['internal']['Ybus']
    P_loads = net.load.p_mw.values / net.sn_mva  # Convert to p.u.
    Q_loads = net.load.q_mvar.values / net.sn_mva  # Convert to p.u.
    P_gens = -net.gen.p_mw.values / net.sn_mva  # Generation is negative power injection
    Q_gens = -net.gen.q_mvar.values / net.sn_mva  # Convert to p.u.
    
    raise NotImplementedError