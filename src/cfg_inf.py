import os
import torch
from datetime import datetime
import uuid
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from cfg_config import parse_args
from data import CFGDataset
from layer import MLPLayers
from utils import create_dir_if_not_exists, write_log_file
from utils import generate_epoch_pair, get_model, ffmpeg_name2label, openssl_name2label


class PropInfModel(nn.Module):
    def __init__(self, base_model_name, node_init_dims, args, device):
        super().__init__()

        if 'filters' in args:
            self.hidden_dim = int(args.filters.split('_')[-1])
        else:
            self.hidden_dim = 100
        if args.dataset == 'ffmpeg':
            self.n_names = len(ffmpeg_name2label)
        elif args.dataset == 'OpenSSL':
            self.n_names = len(openssl_name2label)
        else:
            raise NotImplementedError()

        self.base_model = get_model(base_model_name)(node_init_dims=node_init_dims, arguments=args, device=device)

        self.mlp = MLPLayers([self.hidden_dim * 3, self.hidden_dim])
        self.cls = nn.Linear(self.hidden_dim, self.n_names)

        # Load best model
        self.base_model.load_state_dict(torch.load(args.model_path))
        for p in self.base_model.parameters():
            p.requires_grad = False
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'Trainable parameters: {params}', flush=True)

    def forward(self, batch_x_p, batch_x_h, batch_adj_p, batch_adj_h):
        mes_list = self.base_model.forward(batch_x_p, batch_x_h, batch_adj_p, batch_adj_h, return_mes=True)
        assert sum([_.shape[-1] for _ in mes_list]) == self.hidden_dim * 3, f'Expect n_mes={3}, but have {len(mes_list)}.'
        tensor_list = []
        for mes_tensor in mes_list:
            if len(mes_tensor.shape) == 3:   # node emb
                mes_tensor = torch.mean(mes_tensor, dim=1)
            tensor_list.append(mes_tensor)
        tot_mes = torch.cat(tensor_list, dim=-1)
        logits = self.cls(self.mlp(tot_mes))
        return logits


class PropInfAttacker:
    def __init__(self, model_name, node_init_dims, data_dir, device, log_file, args):
        # training parameters
        self.max_epoch = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.device = device
        
        self.log_file = log_file
        self.best_model_path = args.model_path + f'.{uuid.uuid4().hex[:8]}.tmp'
        
        self.model = PropInfModel(base_model_name=model_name, node_init_dims=node_init_dims, args=args, device=device).to(device)
        write_log_file(self.log_file, str(self.model))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        cfg = CFGDataset(data_dir=data_dir, batch_size=self.batch_size)

        self.n_graphs = len(cfg.graph_train) // 10
        self.graph_train = cfg.graph_train
        self.classes_train = cfg.classes_train
        self.epoch_data_valid = cfg.valid_epoch
        self.epoch_data_test = cfg.test_epoch

        init_val_auc = self.eval_auc_epoch(model=self.model, eval_epoch_data=self.epoch_data_valid)  # evaluate the auc of init model for validation dataset
        write_log_file(self.log_file, "Initial Validation AUC = {0} @ {1}".format(init_val_auc, datetime.now()))

    def fit(self):
        best_val_auc = None
        for i in range(1, self.max_epoch + 1):
            # train
            loss_avg = self.train_one_epoch(n_graphs=self.n_graphs, model=self.model, optimizer=self.optimizer, graphs=self.graph_train, classes=self.classes_train, batch_size=self.batch_size,
                                            device=self.device, load_data=None)
            write_log_file(self.log_file, "EPOCH {0}/{1}:\tMSE loss = {2} @ {3}".format(i, self.max_epoch, loss_avg, datetime.now()))
            # validation
            valid_auc = self.eval_auc_epoch(model=self.model, eval_epoch_data=self.epoch_data_valid)
            write_log_file(self.log_file, "Validation AUC = {0} @ {1}".format(valid_auc, datetime.now()))
            # save the best validation
            if best_val_auc is None or best_val_auc < valid_auc:
                write_log_file(self.log_file, 'Validation AUC increased ({} ---> {}), and saving the model ... '.format(best_val_auc, valid_auc))
                best_val_auc = valid_auc
                torch.save(self.model.state_dict(), self.best_model_path)
            write_log_file(self.log_file, 'Best Validation auc = {} '.format(best_val_auc))
        return best_val_auc

    def testing(self):
        # load the last checkpoint with the best model
        self.model.load_state_dict(torch.load(self.best_model_path))
        self.model.eval()
        # double check the save checkpoint model for validation
        double_val_auc = self.eval_auc_epoch(model=self.model, eval_epoch_data=self.epoch_data_valid)
        # evaluating on the testing dataset
        final_test_auc = self.eval_auc_epoch(model=self.model, eval_epoch_data=self.epoch_data_test)
        
        write_log_file(self.log_file, "\nDouble check for the saved best checkpoint model for validation {} ".format(double_val_auc))
        write_log_file(self.log_file, "Finally, testing auc = {} @ {}".format(final_test_auc, datetime.now()))
        return final_test_auc

    @staticmethod
    def train_one_epoch(n_graphs, model, optimizer, graphs, classes, batch_size, device, load_data=None):
        loss_fn = nn.CrossEntropyLoss()
        model.train()
        if load_data is None:
            epoch_data, id_data = generate_epoch_pair(graphs, classes, batch_size, output_id=True)
        else:
            raise NotImplementedError()
        
        perm = np.random.permutation(len(epoch_data))  # Random shuffle
        
        cum_loss = 0.0
        num = 0
        for index in perm:
            cur_data = epoch_data[index]
            cur_idx = id_data[index][0][0][0]
            if cur_idx > n_graphs:
                continue
            x1, x2, adj1, adj2, y, y1, y2 = cur_data
            logits = model(batch_x_p=x1, batch_x_h=x2, batch_adj_p=adj1, batch_adj_h=adj2)
            y1 = torch.LongTensor(y1).to(device)
            ce_loss = loss_fn(logits, y1)
            
            optimizer.zero_grad()
            ce_loss.backward()
            optimizer.step()
            
            cum_loss += ce_loss.detach()
            if num % int(len(perm) / 10) == 0:
                print('\tTraining: {}/{}: index = {} loss = {}'.format(num, len(epoch_data), index, ce_loss))
            num = num + 1
        return cum_loss / len(perm)

    @staticmethod
    def eval_auc_epoch(model, eval_epoch_data):
        model.eval()
        with torch.no_grad():
            tot_pred = []
            tot_truth = []
            for cur_data in eval_epoch_data:
                x1, x2, adj1, adj2, y, y1, y2 = cur_data
                logits = model(batch_x_p=x1, batch_x_h=x2, batch_adj_p=adj1, batch_adj_h=adj2)
                pred = torch.softmax(logits, dim=-1)
                tot_pred.append(pred.data.cpu().numpy())
                tot_truth.append(y1)

        pred = np.concatenate(tot_pred, axis=0)
        truth = np.concatenate(tot_truth, axis=0)

        model_auc = roc_auc_score(truth, pred, multi_class='ovr')
        return model_auc


if __name__ == '__main__':
    cfg_args = parse_args()
    d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg_args.gpu_index)
    
    model_name = cfg_args.model
    main_data_dir = cfg_args.data_dir
    graph_name = cfg_args.dataset
    graph_min = cfg_args.graph_size_min
    graph_max = cfg_args.graph_size_max
    graph_init_dim = cfg_args.graph_init_dim
    
    # <-><-><-> only for log, delete below if open source
    title = '{}_Min{}_Max{}'.format(graph_name, graph_min, graph_max)
    main_log_dir = cfg_args.log_path + '{}_{}_Min{}_Max{}_InitDims{}_Task_{}/'.format(model_name, graph_name, graph_min, graph_max, graph_init_dim, cfg_args.task)
    create_log_str = create_dir_if_not_exists(main_log_dir)
    best_model_dir = main_log_dir + 'BestModels_{}_Repeat_{}/'.format(get_model(model_name).log_name(cfg_args), cfg_args.repeat_run)
    create_BestModel_dir = create_dir_if_not_exists(best_model_dir)
    LOG_FILE = main_log_dir + 'repeat_{}_'.format(cfg_args.repeat_run) + title + '.txt'
    BestModel_FILE = best_model_dir + title + '.BestModel'
    CSV_FILE = main_log_dir + title + '.csv'
    
    # <-><-><-> only for log, delete above if open source
    sub_data_dir = '{}_{}ACFG_min{}_max{}'.format(graph_name, graph_init_dim, graph_min, graph_max)
    cfg_data_dir = os.path.join(main_data_dir, sub_data_dir) if 'ffmpeg' in sub_data_dir else os.path.join(main_data_dir, sub_data_dir, 'acfgSSL_6')
    assert os.path.exists(cfg_data_dir), "the path of {} is not exist!".format(cfg_data_dir)
    
    write_log_file(LOG_FILE, create_log_str)
    write_log_file(LOG_FILE, create_BestModel_dir)
    write_log_file(LOG_FILE, str(cfg_args))
    cfg_trainer = PropInfAttacker(model_name=model_name, node_init_dims=graph_init_dim, data_dir=cfg_data_dir, device=d, log_file=LOG_FILE, args=cfg_args)
    ret_best_val_auc = cfg_trainer.fit()
    ret_final_test_auc = cfg_trainer.testing()
