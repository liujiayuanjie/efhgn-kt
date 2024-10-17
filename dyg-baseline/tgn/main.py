import torch
import numpy as np
from params import args
from utils import set_seed, Timer, Maximizer, Averager, divide, Batcher
from data import load_data, init_ctdg_data
from model import TGN
import torch.nn.functional as F
from sklearn.metrics import f1_score

if __name__ == '__main__':
    # initialize data
    set_seed(args.seed)
    device = torch.device(args.device)
    path = '%s/%s' % (args.data_path, args.data_name)
    qst_num, usr_num, skl_num, qst_skl, events, qst_msk = load_data(path)
    eve_num = events.shape[0]
    
    qst_skl = torch.tensor(qst_skl, device = device)
    
    qsts = np.arange(qst_num)
    msk_qsts = qsts[qst_msk]
    
    idx = np.arange(msk_qsts.shape[0], dtype = np.int64)
    trn_idx, evl_idx = divide(idx, args.cls_ratio, args.cls_seed)
    trn_qsts = torch.tensor(msk_qsts[trn_idx], device = device)
    evl_qsts = torch.tensor(msk_qsts[evl_idx], device = device)
    
    v_num, train_edges, node_data, ctdg = init_ctdg_data(qst_num, usr_num, events, device)
    # initialize model
    r_num = 2
    tgn = TGN(v_num, r_num, args.emb_dim, args.head_num, args.nbr_num, skl_num, ctdg, args.dropout).to(device)
    optimizer = torch.optim.Adam(
        tgn.parameters(), 
        lr = args.lr, 
        weight_decay = args.weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()
    # train
    def train():
        tgn.train()
        
        for i in range(200):
            cls_res = tgn(node_data)
            loss = criterion(cls_res[trn_qsts], qst_skl[trn_qsts])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()
    # evaluate
    def evaluate():
        tgn.eval()

        cls_res = tgn(node_data)
        p = torch.argmax(cls_res, dim = -1)
        y = qst_skl
        
        trn_p = p[trn_qsts].cpu().numpy()
        trn_y = y[trn_qsts].cpu().numpy()
        
        trn_mi_f1 = f1_score(trn_y, trn_p, average = 'micro')
        
        evl_p = p[evl_qsts].cpu().numpy()
        evl_y = y[evl_qsts].cpu().numpy()
        
        evl_ma_f1 = f1_score(evl_y, evl_p, average = 'macro')
        evl_mi_f1 = f1_score(evl_y, evl_p, average = 'micro')

        return trn_mi_f1, evl_ma_f1, evl_mi_f1
    # conduct experiments
    epoch = 1
    quit_count = 0

    mi_f1_max = Maximizer()
    max_cls_res = None
    
    dur_avg = Averager()

    while quit_count <= args.quit_epoch_num:
        timer = Timer()
        loss = timer(train)
        trn_mi_f1, ma_f1, mi_f1 = timer(evaluate)

        if mi_f1_max.join(mi_f1):
            max_cls_res = ma_f1, mi_f1
            quit_count = 0

        if args.print_detail:
            max_ma_f1, max_mi_f1 = max_cls_res

            print('  '.join((
                'epoch: %-4d' % epoch,
                'loss: %-.4f' % loss,
                'trn_mi_f1: %-.4f' % trn_mi_f1,
                'ma_f1: %-.4f/%-.4f' % (ma_f1, max_ma_f1),
                'mi_f1: %-.4f/%-.4f' % (mi_f1, max_mi_f1),
                'dur: %-.2fs' % timer.get(),
            )))
        
        dur_avg.join(timer.get())
        epoch += 1
        quit_count += 1

    if args.print_result:
        print('%.4f' % dur_avg.get())
        print('%.4f' % (int(torch.cuda.max_memory_allocated()) / 1024 ** 3))
        print('%.4f %.4f' % max_cls_res)