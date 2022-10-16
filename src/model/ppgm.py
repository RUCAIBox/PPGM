import torch
import torch.nn as nn
import torch.nn.functional as functional
from layer import DenseGGNN
from torch_geometric.nn.dense.dense_gcn_conv import DenseGCNConv
from torch_geometric.nn.dense.dense_gin_conv import DenseGINConv
from torch_geometric.nn.dense.dense_sage_conv import DenseSAGEConv


class MessageExtraction(torch.nn.Module):
    def __init__(self, n_queries, n_heads, hidden_size, inner_dim=1024, layer_norm_eps=1e-5):
        super().__init__()

        self.queries = nn.Parameter(torch.rand(n_queries, hidden_size))
        self.mhattn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.ffn_dropout = nn.Dropout(0.1)
        self.ffn_dropout2 = nn.Dropout(0.1)
        self.linear1 = nn.Linear(hidden_size, inner_dim)
        self.linear2 = nn.Linear(inner_dim, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def _ff_block(self, x):
        x = self.linear2(self.ffn_dropout(functional.relu(self.linear1(x))))
        return self.ffn_dropout2(x)

    def forward(self, x, outside_queries=None):
        if outside_queries is None:
            batch_queries = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
            mes, _ = self.mhattn(batch_queries, x, x)
        else:
            mes, _ = self.mhattn(outside_queries, x, x)
        mes = self.norm1(mes)
        mes = self.norm2(mes + self._ff_block(mes))
        return mes


class PPGM(torch.nn.Module):
    @staticmethod
    def add_model_configs(parser):
        parser.add_argument("--filters", type=str, default='100_100_100', help="filters (neurons) for graph neural networks")
        parser.add_argument("--conv", type=str, default='gcn', help="one kind of graph neural networks")
        parser.add_argument("--hidden_size", type=int, default=100, help='hidden size for the graph-level embedding')
        parser.add_argument("--n_queries", type=int, default=8, help='number of learnable queries')
        parser.add_argument("--n_heads", type=int, default=4, help='number of heads')

        # global-level information
        parser.add_argument("--global_flag", type=lambda x: (str(x).lower() == 'true'), default='True', help="Whether use global info ")
        parser.add_argument("--global_agg", type=str, default='lstm', help="aggregation function for global level gcn ")
        return parser

    @staticmethod
    def log_name(args):
        return f'{args.global_agg}_{args.n_queries}_{args.n_heads}'

    def __init__(self, node_init_dims, arguments, device):
        super(PPGM, self).__init__()
        
        self.node_init_dims = node_init_dims
        self.args = arguments
        self.device = device
        self.hidden_size = self.args.hidden_size
        
        self.dropout = arguments.dropout
        
        # ---------- Node Embedding Layer ----------
        filters = self.args.filters.split('_')
        self.gcn_filters = [int(n_filter) for n_filter in filters]  # GCNs' filter sizes
        self.gcn_numbers = len(self.gcn_filters)
        self.gcn_last_filter = self.gcn_filters[-1]  # last filter size of node embedding layer
        
        gcn_parameters = [dict(in_channels=self.gcn_filters[i - 1], out_channels=self.gcn_filters[i], bias=True) for i in range(1, self.gcn_numbers)]
        gcn_parameters.insert(0, dict(in_channels=node_init_dims, out_channels=self.gcn_filters[0], bias=True))
        
        gin_parameters = [dict(nn=nn.Linear(in_features=self.gcn_filters[i - 1], out_features=self.gcn_filters[i])) for i in range(1, self.gcn_numbers)]
        gin_parameters.insert(0, {'nn': nn.Linear(in_features=node_init_dims, out_features=self.gcn_filters[0])})
        
        ggnn_parameters = [dict(out_channels=self.gcn_filters[i]) for i in range(self.gcn_numbers)]
        
        conv_layer_constructor = {
            'gcn': dict(constructor=DenseGCNConv, kwargs=gcn_parameters),
            'graphsage': dict(constructor=DenseSAGEConv, kwargs=gcn_parameters),
            'gin': dict(constructor=DenseGINConv, kwargs=gin_parameters),
            'ggnn': dict(constructor=DenseGGNN, kwargs=ggnn_parameters)
        }
        
        conv = conv_layer_constructor[self.args.conv]
        constructor = conv['constructor']
        # build GCN layers
        setattr(self, 'gc{}'.format(1), constructor(**conv['kwargs'][0]))
        for i in range(1, self.gcn_numbers):
            setattr(self, 'gc{}'.format(i + 1), constructor(**conv['kwargs'][i]))
        
        # Learnable queries
        self.n_queries = self.args.n_queries
        self.message_extractor = MessageExtraction(
            n_queries=self.n_queries,
            n_heads=self.args.n_heads,
            hidden_size=self.gcn_last_filter
        )

        self.indevice_message_extractor = MessageExtraction(
            n_queries=self.n_queries,
            n_heads=self.args.n_heads,
            hidden_size=self.gcn_last_filter
        )

        # global aggregation
        self.global_flag = self.args.global_flag
        if self.global_flag is True:
            self.global_agg = self.args.global_agg
            if self.global_agg.lower() == 'max_pool':
                print("Only Max Pooling")
            elif self.global_agg.lower() == 'fc_max_pool':
                self.global_fc_agg = nn.Linear(2 * self.gcn_last_filter, self.gcn_last_filter)
            elif self.global_agg.lower() == 'mean_pool':
                print("Only Mean Pooling")
            elif self.global_agg.lower() == 'fc_mean_pool':
                self.global_fc_agg = nn.Linear(2 * self.gcn_last_filter, self.gcn_last_filter)
            elif self.global_agg.lower() == 'lstm':
                self.global_lstm_agg = nn.LSTM(input_size=2 * self.gcn_last_filter, hidden_size=self.gcn_last_filter, num_layers=1, bidirectional=True, batch_first=True)
            else:
                raise NotImplementedError
        
        # ---------- Prediction Layer ----------
        if self.args.task.lower() == 'regression':
            factor = 2
            self.predict_fc1 = nn.Linear(int(self.hidden_size * 2 * factor), int(self.hidden_size * factor))
            self.predict_fc2 = nn.Linear(int(self.hidden_size * factor), int((self.hidden_size * factor) / 2))
            self.predict_fc3 = nn.Linear(int((self.hidden_size * factor) / 2), int((self.hidden_size * factor) / 4))
            self.predict_fc4 = nn.Linear(int((self.hidden_size * factor) / 4), 1)
        elif self.args.task.lower() in ['classification', 'cls_attr_inf']:
            print("classification task")
        else:
            raise NotImplementedError

    def global_aggregation_info(self, v, agg_func_name):
        """
        :param v: (batch, len, dim)
        :param agg_func_name:
        :return: (batch, len)
        """
        if agg_func_name.lower() == 'max_pool':
            agg_v = torch.max(v, 1)[0]
        elif agg_func_name.lower() == 'fc_max_pool':
            agg_v = self.global_fc_agg(v)
            agg_v = torch.max(agg_v, 1)[0]
        elif agg_func_name.lower() == 'mean_pool':
            agg_v = torch.mean(v, dim=1)
        elif agg_func_name.lower() == 'fc_mean_pool':
            agg_v = self.global_fc_agg(v)
            agg_v = torch.mean(agg_v, dim=1)
        elif agg_func_name.lower() == 'lstm':
            _, (agg_v_last, _) = self.global_lstm_agg(v)
            agg_v = agg_v_last.permute(1, 0, 2).contiguous().view(-1, self.gcn_last_filter * 2)
        else:
            raise NotImplementedError
        return agg_v

    @staticmethod
    def div_with_small_value(n, d, eps=1e-8):
        # too small values are replaced by 1e-8 to prevent it from exploding.
        d = d * (d > eps).float() + eps * (d <= eps).float()
        return n / d

    def cosine_attention(self, v1, v2):
        """
        :param v1: (batch, len1, dim)
        :param v2: (batch, len2, dim)
        :return:  (batch, len1, len2)
        """
        # (batch, len1, len2)
        a = torch.bmm(v1, v2.permute(0, 2, 1))
        
        v1_norm = v1.norm(p=2, dim=2, keepdim=True)  # (batch, len1, 1)
        v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)  # (batch, len2, 1)
        d = v1_norm * v2_norm
        return self.div_with_small_value(a, d)

    def forward_dense_gcn_layers(self, feat, adj):
        feat_in = feat
        for i in range(1, self.gcn_numbers + 1):
            feat_out = functional.relu(getattr(self, 'gc{}'.format(i))(x=feat_in, adj=adj, mask=None, add_loop=False), inplace=True)
            feat_out = functional.dropout(feat_out, p=self.dropout, training=self.training)
            feat_in = feat_out
        return feat_out

    def forward(self, batch_x_p, batch_x_h, batch_adj_p, batch_adj_h, return_mes=False):
        # ---------- Node Embedding Layer ----------
        feature_p_init = torch.FloatTensor(batch_x_p).to(self.device)
        adj_p = torch.FloatTensor(batch_adj_p).to(self.device)
        feature_h_init = torch.FloatTensor(batch_x_h).to(self.device)
        adj_h = torch.FloatTensor(batch_adj_h).to(self.device)
        
        feature_p = self.forward_dense_gcn_layers(feat=feature_p_init, adj=adj_p)  # (batch, len_p, dim)
        feature_h = self.forward_dense_gcn_layers(feat=feature_h_init, adj=adj_h)  # (batch, len_h, dim)

        message_p = self.message_extractor(feature_p)
        message_h = self.message_extractor(feature_h)

        id_feat_p = self.indevice_message_extractor(x=feature_p, outside_queries=message_h)
        id_feat_h = self.indevice_message_extractor(x=feature_h, outside_queries=message_p)

        # global aggregation

        id_agg_p = torch.cat([id_feat_p, message_h], dim=-1)
        id_agg_h = torch.cat([id_feat_h, message_p], dim=-1)

        agg_p = self.global_aggregation_info(v=id_agg_p, agg_func_name=self.global_agg)
        agg_h = self.global_aggregation_info(v=id_agg_h, agg_func_name=self.global_agg)

        if return_mes:
            return [agg_p, message_p]

        # ---------- Prediction Layer ----------
        if self.args.task.lower() == 'regression':
            x = torch.cat([agg_p, agg_h], dim=1)
            x = functional.dropout(x, p=self.dropout, training=self.training)
            x = functional.relu(self.predict_fc1(x))
            x = functional.dropout(x, p=self.dropout, training=self.training)
            x = functional.relu(self.predict_fc2(x))
            x = functional.dropout(x, p=self.dropout, training=self.training)
            x = functional.relu(self.predict_fc3(x))
            x = functional.dropout(x, p=self.dropout, training=self.training)
            x = self.predict_fc4(x)
            x = torch.sigmoid(x).squeeze(-1)
            return x
        elif self.args.task.lower() == 'classification':
            sim = functional.cosine_similarity(agg_p, agg_h, dim=1).clamp(min=-1, max=1)
            return sim
        else:
            raise NotImplementedError
