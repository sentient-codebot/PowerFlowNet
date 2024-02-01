"""this file generates the dataset using pandapower, it is the the 'dataset_generator_pandapower_v2.py' file in the original repository
Date: Feb 2024
Author: Nan Lin, Stavros Orfanoudakis
"""
import copy
import time
import argparse
import warnings
import pandas as pd
import pandapower as pp
import numpy as np
import networkx as nx
import multiprocessing as mp
import os


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

def generate_data(sublist_size, base_net_create, num_lines_to_remove, num_lines_to_add):
    edge_features_list = []
    node_features_x_list = []
    node_features_y_list = []
    # graph_feature_list = []

    while len(edge_features_list) < sublist_size:
        net = base_net_create()
        remove_c_nf(net)
        
        success_flag, net = perturb_topology(net, num_lines_to_remove=num_lines_to_remove, num_lines_to_add=num_lines_to_add) # TODO 
        if success_flag == 1:
            exit()
        n = net.bus.values.shape[0]
        A = get_adjacency_matrix(net)
        
        net.bus['name'] = net.bus.index

        r = net.line['r_ohm_per_km'].values    
        x = net.line['x_ohm_per_km'].values
        # c = net.line['c_nf_per_km'].values
        le = net.line['length_km'].values
        # x = case['branch'][:, 3]
        # b = case['branch'][:, 4]
        # tau = case['branch'][:, 8]  # ratio

        Pg = net.gen['p_mw'].values
        # Pmin = 
        Pd = net.load['p_mw'].values
        Qd = net.load['q_mvar'].values

        r = np.random.uniform(0.8*r, 1.2*r, r.shape[0])
        x = np.random.uniform(0.8*x, 1.2*x, x.shape[0])
        # c = np.random.uniform(0.8*c, 1.2*c, c.shape[0])
        le = np.random.uniform(0.8*le, 1.2*le, le.shape[0])
        
        # tau = np.random.uniform(0.8*tau, 1.2*tau, case['branch'].shape[0])
        # angle = np.random.uniform(-0.2, 0.2, case['branch'].shape[0])
    
        Vg = np.random.uniform(1.00, 1.05, net.gen['vm_pu'].shape[0])
        Pg = np.random.normal(Pg, 0.1*np.abs(Pg), net.gen['p_mw'].shape[0])
        
        # Pd = np.random.uniform(0.5*Pd, 1.5*Pd, net.load['p_mw'].shape[0])
        Pd = np.random.normal(Pd, 0.1*np.abs(Pd), net.load['p_mw'].shape[0])
        # Qd = np.random.uniform(0.5*Qd, 1.5*Qd, net.load['q_mvar'].shape[0])
        Qd = np.random.normal(Qd, 0.1*np.abs(Qd), net.load['q_mvar'].shape[0])
        
        net.line['r_ohm_per_km'] = r 
        net.line['x_ohm_per_km'] = x 

        net.gen['vm_pu'] = Vg
        net.gen['p_mw'] = Pg

        net.load['p_mw'] = Pd
        net.load['q_mvar'] = Qd

        try:
            net['converged'] = False
            pp.runpp(net, algorithm='nr', init="results", numba=False)
        except:
            if not net['converged']:
                # print(f"net['converged'] = {net['converged']}")
                print(f'Failed to converge, current sample number: {len(edge_features_list)}')
                import pandapower as pp
                continue        

        # Graph feature
        # baseMVA = x[0]['baseMVA']

        # Create a vector od branch features including start and end nodes,r,x,b,tau,angle
        edge_features = np.zeros((net.line.shape[0], 7))
        edge_features[:, 0] = net.line['from_bus'].values + 1
        edge_features[:, 1] = net.line['to_bus'].values + 1
        edge_features[:, 2], edge_features[:, 3] = get_line_z_pu(net)
        edge_features[:, 4] = 0
        edge_features[:, 5] = 0
        edge_features[:, 6] = 0
        
        trafo_edge_features = np.zeros((net.trafo.shape[0], 7))
        trafo_edge_features[:, 0] = net.trafo['hv_bus'].values + 1
        trafo_edge_features[:, 1] = net.trafo['lv_bus'].values + 1
        trafo_edge_features[:, 2], trafo_edge_features[:, 3] = get_trafo_z_pu(net)
        trafo_edge_features[:, 4] = 0
        trafo_edge_features[:, 5] = 0
        trafo_edge_features[:, 6] = 0
        
        edge_features = np.concatenate((edge_features, trafo_edge_features), axis=0)

        # Create a vector of node features including index, type, Vm, Va, Pd, Qd, Gs, Bs, Pg
        # case['bus'] = x[0]['bus']

        node_features_x = np.zeros((n, 9))
        node_features_x[:, 0] = net.bus['name'].values + 1# index
        # Va ----This changes for every bus excecpt slack bus
        node_features_x[:, 3] = np.zeros((n, )) #Va
        
        # node_features_x[:, 6] = np.zeros((n,1)) # Gs
        # node_features_x[:, 7] = np.zeros((n,1)) # Bs
        # Vm is 1 if type is not "generator" else it is case['gen'][:,j]
        vm = np.ones(n)
        types = np.ones(n)*2
        for j in range(net.gen.shape[0]):    
            # find index of case['gen'][j,0] in case['bus'][:,0]
            index = np.where(net.gen['bus'].values[j] == net.bus['name'])[0][0]        
            vm[index] = net.gen['vm_pu'].values[j]  # Vm = Vg
            types[index] = 1  # type = generator
            node_features_x[index, 8] = net.gen['p_mw'].values[j] / net.sn_mva  # Pg / pu
        
        node_features_x[:, 2] = vm  # Vm
        node_features_x[:, 1] = types  # type
        
        for j in range(net.load.shape[0]):    
            # find index of case['gen'][j,0] in case['bus'][:,0]
            index = np.where(net.load['bus'].values[j] == net.bus['name'])[0][0]        
            node_features_x[index, 4] = Pd[j] / net.sn_mva  # Pd / pu
            node_features_x[index, 5] = Qd[j] / net.sn_mva  # Qd / pu

        # Create a vector of node features including index, type, Vm, Va, Pd, Qd, Gs, Bs    
        node_features_y = np.zeros((n, 8))
        node_features_y[:, 0] = net.bus['name'].values + 1 # index
        node_features_y[:, 1] = types  # type
        # Vm ----This changes for Load Buses
        # if net.res_bus['vm_pu'].shape[0] == 0:
        #     pass
        node_features_y[:, 2] = net.res_bus['vm_pu']  # Vm
        # Va ----This changes for every bus excecpt slack bus
        node_features_y[:, 3] = net.res_bus['va_degree']  # Va
        node_features_y[:, 4] = net.res_bus['p_mw'] / net.sn_mva    # P / pu
        node_features_y[:, 5] = net.res_bus['q_mvar'] / net.sn_mva  # Q / pu
        # node_features_y[:, 6] = case['bus'][:, 4]  # Gs
        # node_features_y[:, 7] = case['bus'][:, 5]  # Bs

        edge_features_list.append(edge_features)
        node_features_x_list.append(node_features_x)
        node_features_y_list.append(node_features_y)
        # graph_feature_list.append(baseMVA)

        if len(edge_features_list) % 10 == 0 or len(edge_features_list) == sublist_size:
            print(f'[Process {os.getpid()}] Current sample number: {len(edge_features_list)}')
            
    return edge_features_list, node_features_x_list, node_features_y_list

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
