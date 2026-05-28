import numbers

import networkx as nx
import numpy as np
import torch

from network import MLDHG
from utils import get_task_local_indices


def _compute_pos_weight(samples, min_pos_weight=0.65, max_pos_weight=6.0):
    """根据正负样本比例计算二分类损失函数中的正类权重（pos_weight）。"""
    if samples is None or len(samples) == 0:
        return 1.0
    labels = np.asarray(samples)[:, 2].astype(np.float32)
    pos = max(float(labels.sum()), 1.0)
    neg = max(float(len(labels) - labels.sum()), 1.0)
    return float(np.clip(neg / pos, min_pos_weight, max_pos_weight))


def build_train_network(original_network, task_train_samples, idx_to_node, sim_threshold=0.0):
    G_train = nx.Graph()
    G_train.add_nodes_from(original_network.G.nodes(data=True))

    for task, samples in task_train_samples.items():
        for node_idx1, node_idx2, label in samples:
            if int(label) != 1:
                continue
            node1 = idx_to_node[int(node_idx1)]
            node2 = idx_to_node[int(node_idx2)]
            if original_network.G.has_edge(node1, node2):
                G_train.add_edge(node1, node2, **original_network.G[node1][node2])
            else:
                if task == 'miRNA_disease':
                    edge_type = 'miRNA-disease'
                elif task == 'miRNA_lncRNA':
                    edge_type = 'miRNA-lncRNA'
                elif task == 'lncRNA_disease':
                    edge_type = 'lncRNA-disease'
                else:
                    edge_type = 'unknown'
                G_train.add_edge(node1, node2, edge_type=edge_type, weight=1.0)

    for u, v, data in original_network.G.edges(data=True):
        if G_train.has_edge(u, v):
            continue
        edge_type = str(data.get('edge_type', ''))
        if 'similarity' not in edge_type:
            continue
        sim_value = data.get('weight', data.get('sim', data.get('score', 0.0)))
        if not isinstance(sim_value, numbers.Real):
            sim_value = float(sim_value) if np.isscalar(sim_value) else 0.0
        if float(sim_value) > float(sim_threshold):
            G_train.add_edge(u, v, **data)

    network_train = MLDHG(data_dir=original_network.data_dir)
    network_train.G = G_train
    for attr in [
        'mirna_nodes', 'lncrna_nodes', 'disease_nodes',
        'mirna_sim', 'lncrna_sim', 'disease_sim',
        'mirna_names', 'lncrna_names', 'disease_names',
        'node_info', 'edge_info', 'association_edge_indices'
    ]:
        if hasattr(original_network, attr):
            setattr(network_train, attr, getattr(original_network, attr))
    if hasattr(network_train, 'refresh_metadata'):
        network_train.refresh_metadata()
    return network_train


def _build_idx_name_maps(node_info_dict):
    n_mirna = int(node_info_dict['n_mirna'])
    n_lncrna = int(node_info_dict['n_lncrna'])
    idx_to_name = {}
    for i, name in enumerate(node_info_dict['mirna_names']):
        idx_to_name[i] = name
    for i, name in enumerate(node_info_dict['lncrna_names']):
        idx_to_name[n_mirna + i] = name
    for i, name in enumerate(node_info_dict['disease_names']):
        idx_to_name[n_mirna + n_lncrna + i] = name
    return idx_to_name


def _prepare_semantic_graph_tensors(
    samples_dict,
    tasks,
    n_mirna,
    n_lncrna,
    n_disease,
    node_info_dict,
    device,
):
    mirna_disease_adj = torch.zeros(n_mirna, n_disease, device=device)
    mirna_lncrna_adj = torch.zeros(n_mirna, n_lncrna, device=device)
    lncrna_disease_adj = torch.zeros(n_lncrna, n_disease, device=device)

    for task in tasks:
        for sample in samples_dict[task]:
            u_idx, v_idx, label = sample
            if int(label) != 1:
                continue
            local_u_idx, local_v_idx = get_task_local_indices(node_info_dict, task, u_idx, v_idx)
            if task == 'miRNA_disease':
                mirna_disease_adj[local_u_idx, local_v_idx] = 1
            elif task == 'miRNA_lncRNA':
                mirna_lncrna_adj[local_u_idx, local_v_idx] = 1
            elif task == 'lncRNA_disease':
                lncrna_disease_adj[local_u_idx, local_v_idx] = 1

    return {
        'mirna_disease': mirna_disease_adj,
        'mirna_lncrna': mirna_lncrna_adj,
        'lncrna_disease': lncrna_disease_adj,
    }
