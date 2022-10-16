import argparse
from utils import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Privacy-Preserved Neural Graph Similarity Learning")

    parser.add_argument('--data_dir', type=str, default='data/CFG', help='root directory for the data set')
    parser.add_argument('--dataset', type=str, default="ffmpeg", help='indicate the specific data set (ffmpeg/OpenSSL)')
    parser.add_argument('--graph_size_min', type=int, default=50, help='min node size for one graph ')
    parser.add_argument('--graph_size_max', type=int, default=200, help='max node size for one graph ')
    parser.add_argument('--graph_init_dim', type=int, default=6, help='init feature dimension for one graph')

    parser.add_argument('--model', type=str, default='PPGM', help='name of the model')
    parser.add_argument("--task", type=str, default='classification', help="classification/regression")

    # training parameters for classification tasks
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--patience', type=int, default=100, help='step of patience')
    parser.add_argument("--batch_size", type=int, default=5, help="Number of graph pairs per batch.")
    parser.add_argument("--lr", type=float, default=0.5e-3, help="Learning rate.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")

    # others
    parser.add_argument('--gpu_index', type=str, default='1', help="gpu index to use")
    parser.add_argument('--log_path', type=str, default='CFGLogs/', help='path for log file')
    parser.add_argument('--repeat_run', type=int, default=1, help='indicated the index of repeat run')

    # only test
    parser.add_argument('--only_test', type=lambda x: (str(x).lower() == 'true'), default='false')
    parser.add_argument('--model_path', type=str, default='.')

    base_args, _ = parser.parse_known_args()
    print(f'Model: [{base_args.model}]\n===================\n')
    model_class = get_model(base_args.model)
    model_parser = model_class.add_model_configs(parser)
    cfg_args = model_parser.parse_args()

    return cfg_args
