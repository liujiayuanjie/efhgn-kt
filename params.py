import argparse
import warnings

warnings.simplefilter('ignore')
parser = argparse.ArgumentParser(add_help = False)

# data-related parameters
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--device', type = str, default = 'cuda:0')
parser.add_argument('--data_path', type = str, default = '../dataset')
parser.add_argument('--data_name', type = str, default = 'assist2009')
# model-related parameters
parser.add_argument('--emb_dim', type = int, default = 64)
parser.add_argument('--batch_size', type = int, default = 8000)
parser.add_argument('--lr', type = float, default = 0.01)
parser.add_argument('--weight_decay', type = float, default = 0.01)
parser.add_argument('--neg_num', type = int, default = 8)
parser.add_argument('--layer_num', type = int, default = 2)
parser.add_argument('--ev_tau', type = float, default = 0.1)
parser.add_argument('--ve_tau', type = float, default = 1.0)
parser.add_argument('--r', type = int, default = 2)
parser.add_argument('--lamb', type = float, default = 0.9)
# classification-related parameters
parser.add_argument('--cls_seed', type = int, default = 0)
parser.add_argument('--cls_ratio', type = float, default = 0.8)
parser.add_argument('--cls_epoch_num', type = int, default = 1000)
parser.add_argument('--cls_lr', type = float, default = 1e-2)
parser.add_argument('--cls_weight_decay', type = float, default = 1e-4)
# clustering-related parameters
parser.add_argument('--clr_max_iters', type = float, default = 200)
# KT-related parameters
parser.add_argument('--kt_ratio', type = float, default = 0.8)
parser.add_argument('--kt_seq_len', type = int, default = 200)
parser.add_argument('--kt_batch_size', type = int, default = 200)
parser.add_argument('--kt_lr', type = float, default = 0.01)
parser.add_argument('--kt_weight_decay', type = float, default = 0.00)
parser.add_argument('--kt_dropout', type = float, default = 0.0)
parser.add_argument('--kt_quit_epoch_num', type = int, default = 10)
# result-related parameters
parser.add_argument('--save_path', type = str)
parser.add_argument('--print_detail', action = 'store_true')
parser.add_argument('--print_result', action = 'store_true')
parser.add_argument('--quit_epoch_num', type = int, default = 10)

args, _ = parser.parse_known_args()
