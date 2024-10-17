import torch
import numpy as np
from params import args
from utils import set_seed, Timer, Maximizer, Averager, Batcher, divide
from data import load_data, create_hypergraph_data, sample_evo_flow, create_seqs, format_seqs
from model import HyEdgeEmb, VertexEmb, TimeEmb, HyGAttEmb, EvoFlowEmb
import torch.nn.functional as F
from cls_task import perform_classification_task
from clr_task import perform_clustering_task
from kt_task import perform_knowledge_tracing_task

if __name__ == '__main__':
    # initialize data
    set_seed(args.seed)
    device = torch.device(args.device)
    path = '%s/%s' % (args.data_path, args.data_name)
    qst_num, usr_num, skl_num, qst_skl, events = load_data(path)
    eve_num = events.shape[0]

    v_num, e_num, vv, ve, tme, qst_seqs, usr_seqs, qst_msk = create_hypergraph_data(qst_num, usr_num, events)
    ve_tsr = torch.tensor(ve, device = device)
    tme = torch.tensor(tme, device = device)
    
    qst_msk = torch.tensor(qst_msk, device = device)
    qst_skl = torch.tensor(qst_skl, device = device).long()

    qst_ef = sample_evo_flow(v_num, qst_seqs, args.r)
    usr_ef = sample_evo_flow(v_num, usr_seqs, args.r)
    qst_ef = torch.tensor(qst_ef, device = device)
    usr_ef = torch.tensor(usr_ef, device = device)

    seqs = create_seqs(usr_num, events)
    seqs = np.array(seqs, dtype = object)
    trn_seqs, evl_seqs = divide(seqs, args.kt_ratio, 0)
    trn_seqs = np.array(format_seqs(trn_seqs, args.kt_seq_len), dtype = np.int64)
    evl_seqs = np.array(format_seqs(evl_seqs, args.kt_seq_len), dtype = np.int64)
    # initialize model
    v_embed = HyEdgeEmb(v_num, args.emb_dim).to(device)
    e_embed = VertexEmb(e_num, args.emb_dim).to(device)
    t_embed = TimeEmb(args.emb_dim).to(device)
    hygatt_embed = HyGAttEmb(v_num, e_num, args.emb_dim, args.layer_num, args.ve_tau, args.ev_tau).to(device)
    evoflw_embed = EvoFlowEmb(args.emb_dim, 2, args.layer_num).to(device)

    v_params = {'params': v_embed.parameters(), 'weight_decay': args.weight_decay}
    e_params = {'params': e_embed.parameters(), 'weight_decay': args.weight_decay}
    t_params = {'params': t_embed.parameters()}
    hygatt_params = {'params': hygatt_embed.parameters()}
    evoflw_params = {'params': evoflw_embed.parameters()}

    parameters = [v_params, e_params, t_params, hygatt_params, evoflw_params]
    optimizer = torch.optim.Adam(parameters, lr = args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    # get feature
    def get_fea():
        v_emb = v_embed()
        e_emb = e_embed()
        t_emb = t_embed(tme)
        v_fea, e_fea = hygatt_embed(v_emb, t_emb, e_emb, ve_tsr)
        ef_fea = evoflw_embed(v_emb, t_emb, (qst_ef, usr_ef))
        return v_fea, e_fea, ef_fea, e_emb
    # prepare train
    def pretrain():
        v_embed.train()
        e_embed.train()
        t_embed.train()
        hygatt_embed.train()
        evoflw_embed.train()
    # prepare evaluate
    def preeval():
        v_embed.eval()
        e_embed.eval()
        t_embed.eval()
        hygatt_embed.eval()
        evoflw_embed.eval()
    # train
    def train():
        pretrain()

        ef_loss_avg = Averager()
        ef_batcher = Batcher(vv, args.batch_size)
        
        hg_loss_avg = Averager()
        hg_batcher = Batcher(ve, args.batch_size)
        
        for ef_batch, hy_batch in zip(ef_batcher, hg_batcher):
            # optimizer.zero_grad()
            v_fea, e_fea, ef_fea, e_emb = get_fea()
            
            ef_loss = get_ef_loss(v_fea, ef_batch)
            ef_loss_avg.join(ef_loss.item() / ef_batch.shape[0])
            
            hg_loss = get_hg_loss(ef_fea, e_emb, hy_batch)
            hg_loss_avg.join(hg_loss.item() / hy_batch.shape[0])
            
            loss = args.lamb * ef_loss + (1 - args.lamb) * hg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return ef_loss_avg.get(), hg_loss_avg.get()
    # get evolving flow loss
    def get_ef_loss(v_emb, batch):
        v1, v2 = torch.tensor(batch, device = device).T

        v1_emb = v_emb[v1]
        v2_emb = v_emb[v2]

        bn = v1.size(0)
        nn = args.neg_num

        neg = torch.randint(v_num, size = (bn, nn), device = device)
        neg_emb = v_emb[neg]
        
        pos = (v1_emb * v2_emb).sum(-1).unsqueeze(1)
        neg = (v1_emb.unsqueeze(1) * neg_emb).sum(-1)

        x = torch.cat((pos, neg), dim = 1)
        y = torch.tensor([0] * bn, device = device)

        loss = criterion(x, y) * bn

        return loss 
    # get hypergraph loss
    def get_hg_loss(v_fea, e_fea, hy_batch):
        v, e = torch.tensor(hy_batch, device = device).T

        e_fea_ = e_fea[e]
        v_fea_ = v_fea[v]

        bn = v.size(0)
        nn = args.neg_num 

        neg_e = torch.randint(e_num, size = (bn, nn), device = device)
        neg_e_fea = e_fea[neg_e]
        
        neg_v = torch.randint(v_num, size = (bn, nn), device = device)
        neg_v_fea = v_fea[neg_v]  

        pos = -F.logsigmoid((e_fea_ * v_fea_).sum(-1))
        neg_1 = -F.logsigmoid(-(v_fea_.unsqueeze(1) * neg_e_fea).sum(-1))
        neg_2 = -F.logsigmoid(-(e_fea_.unsqueeze(1) * neg_v_fea).sum(-1))

        loss = pos + neg_1.mean(-1) + neg_2.mean(-1)
        loss = loss.sum()

        return loss
    # evaluate
    def evaluate():
        preeval()

        v_fea, e_fea, ef_fea, e_emb = get_fea()        
        x = e_fea.detach()[: qst_num]
        y = qst_skl
        
        x_ = x[qst_msk]
        y_ = y[qst_msk]

        cls_res = perform_classification_task(
            x_, y_, 
            args.cls_ratio, args.cls_epoch_num, 
            args.cls_lr, args.cls_weight_decay, args.cls_seed
        )
        
        clr_res = perform_clustering_task(
            x_, y_, 
            skl_num, args.clr_max_iters 
        )
        
        kt_res = perform_knowledge_tracing_task(
            x, trn_seqs, evl_seqs, args.kt_batch_size, args.kt_lr, 
            args.kt_weight_decay, args.kt_dropout, args.kt_quit_epoch_num
        )

        return cls_res, clr_res, kt_res
    # conduct experiments
    epoch = 1
    quit_count = 0

    mi_f1_max = Maximizer()
    max_cls_res = None
    
    ari_max = Maximizer()
    max_clr_res = None
    
    auc_max = Maximizer()
    max_kt_res = None
    
    dur_avg = Averager()

    while quit_count <= args.quit_epoch_num:
        timer = Timer()
        ef_loss, hy_loss = timer(train)
        (trn_mi_f1, ma_f1, mi_f1), (nmi, ari), (auc, acc) = timer(evaluate)

        if mi_f1_max.join(mi_f1):
            max_cls_res = ma_f1, mi_f1
            quit_count = 0
        
        if ari_max.join(ari):
            max_clr_res = nmi, ari
            quit_count = 0
            
            # if args.save_path != None:
            #     v_fea, e_fea, ef_fea, e_emb = get_fea()
            #     qst_embedding = e_fea.detach().cpu().numpy()[: qst_num]
                
        if auc_max.join(auc):
            max_kt_res = auc, acc

            if args.save_path != None:
                v_fea, e_fea, ef_fea, e_emb = get_fea()
                qst_embedding = e_fea.detach().cpu().numpy()[: qst_num]

        if args.print_detail:
            max_ma_f1, max_mi_f1 = max_cls_res
            max_nmi, max_ari = max_clr_res
            max_auc, max_acc = max_kt_res

            print('  '.join((
                'epoch: %-4d' % epoch,
                'ef_loss: %-.4f' % ef_loss,
                'hy_loss: %-.4f' % hy_loss,
                'trn_mi_f1: %-.4f' % trn_mi_f1,
                'ma_f1: %-.4f/%-.4f' % (ma_f1, max_ma_f1),
                'mi_f1: %-.4f/%-.4f' % (mi_f1, max_mi_f1),
                'nmi: %-.4f/%-.4f' % (nmi, max_nmi),
                'ari: %-.4f/%-.4f' % (ari, max_ari),
                'auc: %-.4f/%-.4f' % (auc, max_auc),
                'acc: %-.4f/%-.4f' % (acc, max_acc),
                'dur: %-.2fs' % timer.get(),
            )))
        
        dur_avg.join(timer.get())
        epoch += 1
        quit_count += 1

    if args.print_result:
        print('%.4f' % dur_avg.get())
        print('%.4f' % (int(torch.cuda.max_memory_allocated()) / 1024 ** 3))
        print('%.4f %.4f' % max_cls_res)
        print('%.4f %.4f' % max_clr_res)
        print('%.4f %.4f' % max_kt_res)

    if args.save_path != None:
        np.save(args.save_path, qst_embedding)