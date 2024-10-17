import argparse
import warnings

warnings.simplefilter('ignore')
parser = argparse.ArgumentParser(add_help = False)

# data-related parameters
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--device', type = str, default = 'cuda:0')
parser.add_argument('--data_path', type = str, default = '../../../dataset')
parser.add_argument('--data_name', type = str, default = 'assist2009')
# model-related parameters
parser.add_argument('--emb_dim', type = int, default = 64)
parser.add_argument('--lr', type = float, default = 0.0003)
parser.add_argument('--weight_decay', type = float, default = 0.001) 
parser.add_argument('--head_num', type = int, default = 4)
parser.add_argument('--nbr_num', type = int, default = 6)
parser.add_argument('--dropout', type = float, default = 0.3)
# classification-related parameters
parser.add_argument('--cls_seed', type = int, default = 0)
parser.add_argument('--cls_ratio', type = float, default = 0.8)
# result-related parameters
parser.add_argument('--print_detail', action = 'store_true')
parser.add_argument('--print_result', action = 'store_true')
parser.add_argument('--save_path', type = str)
parser.add_argument('--quit_epoch_num', type = int, default = 10)

args, _ = parser.parse_known_args()
