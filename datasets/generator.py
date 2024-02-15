"""this file generates the dataset using pandapower, it is the the 'dataset_generator_pandapower_v2.py' file in the original repository
Date: Feb 2024
Author: Nan Lin, Stavros Orfanoudakis
"""
import warnings
import multiprocessing as mp
import os
import shutil
import copy
from collections import namedtuple
from collections.abc import Iterable
from functools import partial

import pandas as pd
import pandapower as pp
from pandapower import LoadflowNotConverged
import numpy as np
import networkx as nx
from tqdm import tqdm

PowerFlowData = namedtuple('PowerFlowData', ['node_features', 'edge_features', 'sn_mva'])
# powerflowdata_dtype = np.dtype([('node_features', 'f8', (None, 6)), ('edge_features', 'f8', (None, 4)), ('sn_mva', 'f8')])

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

def generate_sample(base_net_create) -> list[PowerFlowData|None, bool]:
    """generate one sample. 
    :param base_net_create: a function that returns a pandapower network.
    :return PowerFlowData|None: a named tuple containing node_features, edge_features, and sn_mva.
    :return is_success: a flag indicating whether the power flow calculation was successful.
    """
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
    r = np.random.uniform(0.9*r, 1.1*r, r.shape[0])
    x = np.random.uniform(0.9*x, 1.1*x, x.shape[0])
    c = np.random.uniform(0.9*c, 1.1*c, c.shape[0])
    # le = np.random.uniform(0.8*le, 1.2*le, le.shape[0]) # keep line length unchanged. 
    # theta_shift_degree = np.random.uniform(-11.46, 11.46, theta_shift_degree.shape[0]) # -0.2 ~ 0.2 rad

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
    
    net['converged'] = False
    is_success = False
    try:
        pp.runpp(net, algorithm='nr', init="results", numba=False)
    except LoadflowNotConverged:
        is_success = False
    if net['converged'] == False:
        is_success = False
    else:
        is_success = True
        
    if not is_success:
        return None, is_success
    
    # Get results
    ybus = net._ppc["internal"]["Ybus"].todense() 

    # Extract edge index and features from ybus
    G = nx.Graph(ybus)
    edge_features_raw = np.array(list(G.edges.data('weight')))
    edge_weights = np.stack([
        (1./(1e-7+edge_features_raw[:, 2])).real, 
        (1./(1e-7+edge_features_raw[:, 2])).imag
    ], axis=1) # shape: (num_edges, 2)
    edge_features = np.concatenate([edge_features_raw[:, :2], edge_weights], axis=1) # shape: (num_edges, 4). from, to, r, x
    edge_features = edge_features.real # shape: (num_edges, 4)

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
    sample = PowerFlowData(node_features, edge_features, net.sn_mva)
    
    return sample, is_success

def generate_data(
    sample_indices: list[int], 
    max_cont_fails=10, 
    disable_pbar=False, 
    save_data=False,
    save_dir='data/power_flow_dataset/raw',
    case_name='unknown',
):
    """ sample_indices: list of indices of samples to generate. length = number of samples.
    
    Returns:
        - sample_list: 
            - if `save_data`, a list of indices of the saved samples.
            - else, a list of PowerFlowData namedtuples.
    """
    num_samples = len(sample_indices)
    sample_list = []
    num_cont_fails = 0 # number of continous failed pf calculations
    num_total_fails = 0 # number of total failed pf calculations
    
    # get base_net_create
    if case_name == '3':
        base_net_create = create_case3
    elif case_name == '14':
        base_net_create = pp.networks.case14
    elif case_name == '57':
        base_net_create = pp.networks.case57
    elif case_name == '118':
        base_net_create = pp.networks.case118
    elif case_name == '145':
        base_net_create = pp.networks.case145
    elif case_name == '6470rte':
        base_net_create = pp.networks.case6470rte
    else:
        raise ValueError('Invalid case.')

    it = iter(sample_indices)
    with tqdm (total=num_samples, disable = disable_pbar) as pbar:
        while len(sample_list) < num_samples:
            if num_cont_fails > max_cont_fails:
                warnings.warn("Too many failed power flow calculations. Return current samples.")
                break
            # generate sample
            sample, is_success = generate_sample(base_net_create)
            if not is_success:
                num_cont_fails += 1
                num_total_fails += 1
                continue
            else:
                num_cont_fails = 0
            
            # save sample
            if save_data:
                _idx = save_sample_csv(sample, save_dir, case_name, next(it))
                sample_list.append(_idx)
            else:
                # Append to list
                sample_list.append(sample)
            
            # pbar
            pbar.set_description(f'#Success: {len(sample_list)}/{num_samples}, #Fails: {num_total_fails}, #Cont.Fails: {num_cont_fails}')
            pbar.update(1)
            
    return sample_list

def save_data(sample_array: np.ndarray[PowerFlowData], save_dir: str, case_name: str):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{case_name}_samples.npz')
    np.savez_compressed(save_path, sample_array)
    
def save_sample_csv(sample: PowerFlowData, save_dir: str, case_name: str, idx: int) -> int:
    save_dir = os.path.join(save_dir, 'case' + case_name)
    os.makedirs(save_dir, exist_ok=True)
    # node features
    _path = os.path.join(save_dir, 'node_features', f'case{case_name}_node_features_{idx}.csv')
    os.makedirs(os.path.dirname(_path), exist_ok=True)
    np.savetxt(_path, sample.node_features,
                fmt=['%d', '%d', '%f', '%f', '%f', '%f'],
                delimiter=',', header='index,type,Vm,Va,Pd,Qd', comments='')
    # edge features
    _path = os.path.join(save_dir, 'edge_features', f'case{case_name}_edge_features_{idx}.csv')
    os.makedirs(os.path.dirname(_path), exist_ok=True)
    np.savetxt(_path, sample.edge_features,
                fmt=['%d', '%d', '%f', '%f'],
                delimiter=',', header='from,to,r,x', comments='')
    # sn_mva
    _path = os.path.join(save_dir, 'sn_mva', f'case{case_name}_sn_mva_{idx}.csv')
    os.makedirs(os.path.dirname(_path), exist_ok=True)
    np.savetxt(_path, np.array([sample.sn_mva]), delimiter=',', header='sn_mva', comments='')
    
    return idx
    
def save_data_csv(sample_list: list[PowerFlowData], save_dir: str, case_name: str):
    save_dir = os.path.join(save_dir, 'case' + case_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # node features
    for idx, sample in enumerate(sample_list):
        _path = os.path.join(save_dir, 'node_features', f'case{case_name}_node_features_{idx}.csv')
        os.makedirs(os.path.dirname(_path), exist_ok=True)
        np.savetxt(_path, sample.node_features, 
                   fmt=['%d', '%d', '%f', '%f', '%f', '%f'],
                   delimiter=',', header='index,type,Vm,Va,Pd,Qd', comments='')
        
    # edge features
    for idx, sample in enumerate(sample_list):
        _path = os.path.join(save_dir, 'edge_features', f'case{case_name}_edge_features_{idx}.csv')
        os.makedirs(os.path.dirname(_path), exist_ok=True)
        np.savetxt(_path, sample.edge_features, 
                   fmt=['%d', '%d', '%f', '%f'],
                   delimiter=',', header='from,to,r,x', comments='')
    
    # sn_mva
    for idx, sample in enumerate(sample_list):
        _path = os.path.join(save_dir, 'sn_mva', f'case{case_name}_sn_mva_{idx}.csv')
        os.makedirs(os.path.dirname(_path), exist_ok=True)
        np.savetxt(_path, np.array([sample.sn_mva]), delimiter=',', header='sn_mva', comments='')
        
def compress_csv(save_dir: str, case_name: str) -> None:
    dir_to_compress = os.path.join(save_dir, 'case' + case_name)
    output_filename = os.path.join(save_dir, 'case' + case_name) # without extension
    res = shutil.make_archive(output_filename, 'zip', dir_to_compress)
    print(res)

def generate_data_parallel(num_samples, num_processes, **generation_kwargs) -> list[PowerFlowData|int]:
    sublist_size = num_samples // num_processes
    pool = mp.Pool(processes=num_processes)
    _generate_data = partial(generate_data, **generation_kwargs,)
    list_sample_indices = [list(range(sublist_size*i, sublist_size*(i+1))) for i in range(num_processes)]
    args = list_sample_indices
    _res = pool.map(_generate_data, args)
    pool.close()
    pool.join()
    
    results = _res[0]
    for i in range(1, num_processes):
        results += _res[i]
         
    return results

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