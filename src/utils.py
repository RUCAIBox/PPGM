import importlib
import networkx as nx
import numpy as np
import os
from scipy import stats
import math
import torch
from torch_geometric.data import Data
from torch_cluster import random_walk
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, dense_to_sparse
from torch_sparse import spspmm, coalesce


class TwoHopNeighbor(object):
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        N = data.num_nodes

        value = edge_index.new_ones((edge_index.size(1), ), dtype=torch.float)

        index, value = spspmm(edge_index, value, edge_index, value, N, N, N, True)
        value.fill_(0)
        index, value = remove_self_loops(index, value)

        edge_index = torch.cat([edge_index, index], dim=1)
        if edge_attr is None:
            data.edge_index, _ = coalesce(edge_index, None, N, N)
        else:
            value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
            value = value.expand(-1, *list(edge_attr.size())[1:])
            edge_attr = torch.cat([edge_attr, value], dim=0)
            data.edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
            data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


def hypergraph_construction(edge_index, edge_attr, num_nodes, k=2, mode='RW'):
    if mode == 'RW':
        # Utilize random walk to construct hypergraph
        row, col = edge_index
        start = torch.arange(num_nodes, device=edge_index.device)
        walk = random_walk(row, col, start, walk_length=k)
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float, device=edge_index.device)
        adj[walk[start], start.unsqueeze(1)] = 1.0
        edge_index, _ = dense_to_sparse(adj)
    else:
        # Utilize neighborhood to construct hypergraph
        if k == 1:
            edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr)
        else:
            neighbor_augment = TwoHopNeighbor()
            hop_data = Data(edge_index=edge_index, edge_attr=edge_attr)
            hop_data.num_nodes = num_nodes
            for _ in range(k-1):
                hop_data = neighbor_augment(hop_data)
            hop_edge_index = hop_data.edge_index
            hop_edge_attr = hop_data.edge_attr
            edge_index, edge_attr = add_remaining_self_loops(hop_edge_index, hop_edge_attr, num_nodes=num_nodes)
    
    return edge_index, edge_attr


def hyperedge_representation(x, edge_index):
    gloabl_edge_rep = x[edge_index[0]]
    gloabl_edge_rep = scatter(gloabl_edge_rep, edge_index[1], dim=0, reduce='mean')

    x_rep = x[edge_index[0]]
    gloabl_edge_rep = gloabl_edge_rep[edge_index[1]]

    coef = softmax(torch.sum(x_rep * gloabl_edge_rep, dim=1), edge_index[1], num_nodes=x_rep.size(0))
    weighted = coef.unsqueeze(-1) * x_rep

    hyperedge = scatter(weighted, edge_index[1], dim=0, reduce='sum')

    return hyperedge


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def metrics_spearmanr_rho(true, predication):
    assert true.shape == predication.shape
    rho, p_val = stats.spearmanr(true, predication)
    return rho


def metrics_kendall_tau(true, predication):
    assert true.shape == predication.shape
    tau, p_val = stats.kendalltau(true, predication)
    return tau


def metrics_mean_square_error(true, predication):
    assert true.shape == predication.shape
    mse = (np.square(true - predication).mean())
    return mse


def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return 'Make dirs of # {} '.format(directory)
    else:
        return "the dirs already exist! Cannot be created"


def write_log_file(file_name_path, log_str, print_flag=True):
    if print_flag:
        print(log_str)
    if log_str is None:
        log_str = 'None'
    if os.path.isfile(file_name_path):
        with open(file_name_path, 'a+') as log_file:
            log_file.write(log_str + '\n')
    else:
        with open(file_name_path, 'w+') as log_file:
            log_file.write(log_str + '\n')


def print_args(args, file_path):
    d = max(map(len, args.__dict__.keys())) + 1
    with open(file_path, 'w') as f:
        for k, v, in args.__dict__.items():
            f.write(k.ljust(d) + ': ' + str(v) + '\n')


def read_all_gexf_graphs(dir):
    """
    read all the files with .gexf to networkx graph
    :param dir:
    :return: list of graphs
    """
    graphs = []
    
    for file in os.listdir(dir):
        if file.rsplit('.')[-1] != 'gexf':
            continue
        file_path = os.path.join(dir, file)
        g = nx.readwrite.gexf.read_gexf(file_path)
        graphs.append(g)
    
    return graphs


class graph(object):
    def __init__(self, node_num=0, label=None, name=None, prefix_name_label=None):
        self.node_num = node_num
        self.label = label
        self.name = name
        self.prefix_name_label = prefix_name_label
        self.features = []  # node feature matrix
        self.succs = []
        self.preds = []
        if node_num > 0:
            for i in range(node_num):
                self.features.append([])
                self.succs.append([])
                self.preds.append([])
    
    def add_edge(self, u, v):
        self.succs[u].append(v)
        self.preds[v].append(u)


def generate_epoch_pair(graphs, classes, batch, output_id=False, load_id=None):
    epoch_data = []
    id_data = []  # [ ([(G0,G1),(G0,G1), ...], [(G0,H0),(G0,H0), ...]), ... ]
    
    if load_id is None:
        st = 0
        while st < len(graphs):
            if output_id:
                input1, input2, adj1, adj2, y, pos_id, neg_id = get_pair(graphs, classes, batch, st=st, output_id=True)
                id_data.append((pos_id, neg_id))
                g1id = [graphs[_[0]].name for _ in pos_id] + [graphs[_[0]].name for _ in neg_id]
                g2id = [graphs[_[1]].name for _ in pos_id] + [graphs[_[1]].name for _ in neg_id]
                y1 = parse_name_to_label(g1id)
                y2 = parse_name_to_label(g2id)
                epoch_data.append((input1, input2, adj1, adj2, y, y1, y2))
            else:
                input1, input2, adj1, adj2, y = get_pair(graphs, classes, batch, st=st)
                epoch_data.append((input1, input2, adj1, adj2, y))
            st += batch
    else:  # Load from previous id_data
        id_data = load_id
        for id_pair in id_data:
            g1id = [graphs[_[0]].name for _ in id_pair[0]] + [graphs[_[0]].name for _ in id_pair[1]]
            g2id = [graphs[_[1]].name for _ in id_pair[0]] + [graphs[_[1]].name for _ in id_pair[1]]
            y1 = parse_name_to_label(g1id)
            y2 = parse_name_to_label(g2id)
            input1, input2, adj1, adj2, y = get_pair(graphs, classes, batch, load_id=id_pair)
            epoch_data.append((input1, input2, adj1, adj2, y, y1, y2))
    
    if output_id:
        return epoch_data, id_data
    else:
        return epoch_data


def get_pair(graphs, classes, batch, st=-1, output_id=False, load_id=None, output_each_label=False):
    if load_id is None:
        len_class = len(classes)
        
        if st + batch > len(graphs):
            batch = len(graphs) - st
        ed = st + batch
        
        pos_ids = []  # [(G_0, G_1), ... ]
        neg_ids = []  # [(G_0, H_0), ... ]
        
        for g_id in range(st, ed):
            g0 = graphs[g_id]
            cls = g0.label  # function name label index of graph
            tot_g = len(classes[cls])
            
            # positive pair
            if len(classes[cls]) >= 2:
                g1_id = classes[cls][np.random.randint(tot_g)]
                while g_id == g1_id:
                    g1_id = classes[cls][np.random.randint(tot_g)]
                pos_ids.append((g_id, g1_id))
            else:
                pos_ids.append((g_id, g_id))
            
            # negative pair
            cls2 = np.random.randint(len_class)
            while (len(classes[cls2]) == 0) or (cls2 == cls):
                cls2 = np.random.randint(len_class)
            tot_g2 = len(classes[cls2])
            g2_id = classes[cls2][np.random.randint(tot_g2)]
            neg_ids.append((g_id, g2_id))
    
    else:
        pos_ids = load_id[0]
        neg_ids = load_id[1]
    
    batch_pos = len(pos_ids)
    batch_neg = len(neg_ids)
    batch = batch_pos + batch_neg
    
    max_num_1 = 0
    max_num_2 = 0
    for pair in pos_ids:
        max_num_1 = max(max_num_1, graphs[pair[0]].node_num)
        max_num_2 = max(max_num_2, graphs[pair[1]].node_num)
    for pair in neg_ids:
        max_num_1 = max(max_num_1, graphs[pair[0]].node_num)
        max_num_2 = max(max_num_2, graphs[pair[1]].node_num)
    
    feature_dim = len(graphs[0].features[0])
    x1_input = np.zeros((batch, max_num_1, feature_dim))
    x2_input = np.zeros((batch, max_num_2, feature_dim))
    adj1 = np.zeros((batch, max_num_1, max_num_1))
    adj2 = np.zeros((batch, max_num_2, max_num_2))
    y_input = np.zeros(batch)
    
    # if output_each_label:
    x1_labels = np.zeros(batch)
    x2_labels = np.zeros(batch)
    
    for i in range(batch_pos):
        y_input[i] = 1
        g1 = graphs[pos_ids[i][0]]
        g2 = graphs[pos_ids[i][1]]
        for u in range(g1.node_num):
            x1_input[i, u, :] = np.array(g1.features[u])
            for v in g1.succs[u]:
                adj1[i, u, v] = 1
        for u in range(g2.node_num):
            x2_input[i, u, :] = np.array(g2.features[u])
            for v in g2.succs[u]:
                adj2[i, u, v] = 1
        if output_each_label:
            x1_labels[i] = g1.prefix_name_label  # TODO: label or prefix label
            x2_labels[i] = g2.prefix_name_label
    
    for i in range(batch_pos, batch_pos + batch_neg):
        y_input[i] = -1
        g1 = graphs[neg_ids[i - batch_pos][0]]
        g2 = graphs[neg_ids[i - batch_pos][1]]
        for u in range(g1.node_num):
            x1_input[i, u, :] = np.array(g1.features[u])
            for v in g1.succs[u]:
                adj1[i, u, v] = 1
        for u in range(g2.node_num):
            x2_input[i, u, :] = np.array(g2.features[u])
            for v in g2.succs[u]:
                adj2[i, u, v] = 1
        if output_each_label:
            x1_labels[i] = g1.prefix_name_label  # TODO: label or prefix label
            x2_labels[i] = g2.prefix_name_label  # TODO: label or prefix label
    
    if not output_each_label:
        if output_id:
            return x1_input, x2_input, adj1, adj2, y_input, pos_ids, neg_ids
        else:
            return x1_input, x2_input, adj1, adj2, y_input
    else:
        if output_id:
            return x1_input, x2_input, adj1, adj2, y_input, x1_labels, x2_labels, pos_ids, neg_ids
        else:
            return x1_input, x2_input, adj1, adj2, y_input, x1_labels, x2_labels


def get_model(model_name):
    model_file_name = model_name.lower()
    model_module = None
    module_path = '.'.join(['model', model_file_name])
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)

    if model_module is None:
        raise ValueError(f'Model named {model_name} not found.')
    model_class = getattr(model_module, model_name)
    return model_class


ffmpeg_name2label = {
    'clang_O0': 0,
    'clang_O1': 1,
    'clang_O2': 2,
    'clang_O3': 3,
    'gcc_O0': 4,
    'gcc_O1': 5,
    'gcc_O2': 6,
    'gcc_O3': 7
}


openssl_name2label = {
    'openssl-1.0.1f-mips-linux-O0v54': 0,
    'openssl-1.0.1u-i586-linux-O1v54': 1,
    'openssl-1.0.1f-i586-linux-O2v54': 2,
    'openssl-1.0.1f-armeb-linux-O0v54': 3,
    'openssl-1.0.1f-armeb-linux-O1v54': 4,
    'openssl-1.0.1u-armeb-linux-O3v54': 5,
    'openssl-1.0.1f-i586-linux-O3v54': 6,
    'openssl-1.0.1f-mips-linux-O1v54': 7,
    'openssl-1.0.1u-armeb-linux-O1v54':8 ,
    'openssl-1.0.1f-i586-linux-O0v54': 9,
    'openssl-1.0.1f-mips-linux-O2v54': 10,
    'openssl-1.0.1u-mips-linux-O3v54': 11,
    'openssl-1.0.1f-armeb-linux-O3v54': 12,
    'openssl-1.0.1u-i586-linux-O0v54': 13,
    'openssl-1.0.1u-armeb-linux-O0v54': 14,
    'openssl-1.0.1u-i586-linux-O3v54': 15,
    'openssl-1.0.1u-mips-linux-O1v54': 16,
    'openssl-1.0.1f-mips-linux-O3v54': 17,
    'openssl-1.0.1u-mips-linux-O2v54': 18,
    'openssl-1.0.1u-i586-linux-O2v54': 19,
    'openssl-1.0.1u-mips-linux-O0v54': 20,
    'openssl-1.0.1f-i586-linux-O1v54': 21,
    'openssl-1.0.1f-armeb-linux-O2v54': 22,
    'openssl-1.0.1u-armeb-linux-O2v54': 23,
}


def parse_name_to_label(name_list):
    labels = []
    for name in name_list:
        if 'ffmpeg' in name:
            labels.append(ffmpeg_name2label[name.split('/')[-2]])
        elif 'openssl' in name:
            labels.append(openssl_name2label[name.split('/')[-2]])
        else:
            print(name_list)
            raise NotImplementedError()
    return np.array(labels)
