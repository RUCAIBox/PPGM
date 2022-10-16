import json
import numpy as np
import os
import utils


class CFGDataset(object):
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.func_name_dict = self.get_f_dict()
        self.gs, self.classes = self.read_graph()
        
        class_perm_path = os.path.join(self.data_dir, 'class_perm.npy')
        if os.path.isfile(class_perm_path):
            perm = np.load(class_perm_path)
        else:
            perm = np.random.permutation(len(self.classes))
            np.save(class_perm_path, perm)
        
        if len(perm) < len(self.classes):
            perm = np.random.permutation(len(self.classes))
            np.save(class_perm_path, perm)
        graphs_train, classes_train, graphs_dev, classes_dev, graphs_test, classes_test = self.partition_graph_dataset(self.gs, self.classes, [0.8, 0.1, 0.1], perm)
        self.graph_train = graphs_train
        self.classes_train = classes_train
        print("{} Train: {} graphs, {} functions".format(data_dir, len(graphs_train), len(classes_train)))
        print("{} Dev  : {} graphs, {} functions".format(data_dir, len(graphs_dev), len(classes_dev)))
        print("{} Test : {} graphs, {} functions".format(data_dir, len(graphs_test), len(classes_test)))
        
        # Fix the pairs for validation and testing
        if os.path.isfile(os.path.join(self.data_dir, 'valid.json')):
            with open(os.path.join(self.data_dir, 'valid.json')) as in_file:
                valid_ids = json.load(in_file)
            self.valid_epoch = utils.generate_epoch_pair(graphs_dev, classes_dev, batch_size, load_id=valid_ids)
        else:
            self.valid_epoch, valid_ids = utils.generate_epoch_pair(graphs_dev, classes_dev, batch_size, output_id=True)
            with open(os.path.join(self.data_dir, 'valid.json'), 'w') as out_file:
                json.dump(valid_ids, out_file)
        
        if os.path.isfile(os.path.join(self.data_dir, 'test.json')):
            with open(os.path.join(self.data_dir, 'test.json')) as in_file:
                test_ids = json.load(in_file)
            self.test_epoch = utils.generate_epoch_pair(graphs_test, classes_test, batch_size, load_id=test_ids)
        else:
            self.test_epoch, test_ids = utils.generate_epoch_pair(graphs_test, classes_test, batch_size, output_id=True)
            with open(os.path.join(self.data_dir, 'test.json'), 'w') as out_file:
                json.dump(test_ids, out_file)
    
    def get_f_dict(self):
        name_num = 0
        name_dict = {}
        for file in os.listdir(self.data_dir):
            if '.json' not in file or 'test' in file or 'valid' in file:
                continue
            f_name = os.path.join(self.data_dir, file)
            with open(f_name) as inf:
                print(f_name)
                for line in inf:
                    g_info = json.loads(line.strip())
                    if g_info['fname'] not in name_dict:
                        name_dict[g_info['fname']] = name_num
                        name_num += 1
        return name_dict
    
    def read_graph(self):
        graphs = []
        classes = []
        if self.func_name_dict is not None:
            for f in range(len(self.func_name_dict)):
                classes.append([])
        for file in os.listdir(self.data_dir):
            if '.json' not in file or 'test' in file or 'valid' in file:
                continue
            f_name = os.path.join(self.data_dir, file)
            with open(f_name) as inf:
                for line in inf:
                    g_info = json.loads(line.strip())
                    label = self.func_name_dict[g_info['fname']]
                    classes[label].append(len(graphs))
                    cur_graph = utils.graph(node_num=g_info['n_num'], label=label, name=g_info['src'])
                    for u in range(g_info['n_num']):
                        cur_graph.features[u] = np.array(g_info['features'][u])
                        for v in g_info['succs'][u]:
                            cur_graph.add_edge(u, v)
                    graphs.append(cur_graph)
        return graphs, classes
    
    @staticmethod
    def partition_graph_dataset(graphs, classes, partitions, perm):
        len_class = len(classes)
        st = 0.0
        ret = []
        for partition in partitions:
            cur_g = []
            cur_c = []
            ed = st + partition * len_class
            for cls in range(int(st), int(ed)):
                prev_class = classes[perm[cls]]
                cur_c.append([])
                for i in range(len(prev_class)):
                    cur_g.append(graphs[prev_class[i]])
                    cur_g[-1].label = len(cur_c) - 1
                    cur_c[-1].append(len(cur_g) - 1)
            ret.append(cur_g)
            ret.append(cur_c)
            st = ed
        return ret
