import numpy as np
from ctdgraph import CTDGraph
import torch

def read_csv(path):
    fp = open(path, 'r')
    rows = fp.read().split('\n')[: -1]
    split = lambda row: [int(e) if e.isdigit() else e for e in row.split(',')]
    rows = [split(row) for row in rows]

    return rows

def load_data(path):
    qsts = read_csv('%s/qst.csv' % path)
    qst_num = len(qsts)

    usrs = read_csv('%s/usr.csv' % path)
    usr_num = len(usrs)

    skls = read_csv('%s/skl.csv' % path)
    skl_num = len(skls)

    qst_skl = read_csv('%s/qst_skl.csv' % path)
    qst_skl = np.array(qst_skl).astype(np.int64)[:, 1]

    events = read_csv('%s/event.csv' % path)
    events = np.array(events).astype(np.int64)
    
    uni_qst = np.unique(events[:, 1])
    qst_msk = np.zeros(qst_num)
    qst_msk[uni_qst] = 1.0
    qst_msk = qst_msk > 0.0
    
    return qst_num, usr_num, skl_num, qst_skl, events, qst_msk

def init_ctdg_data(qst_num, usr_num, events, device):
    v_num = qst_num + usr_num
    edges = events[:]
    edges[:, 0] = edges[:, 0] + qst_num
    ctdg = CTDGraph(v_num, edges, device)

    train_edges = np.array(ctdg.edge_data).astype(np.int64)
    node_data = torch.tensor(ctdg.final_node).to(device)
    
    return v_num, train_edges, node_data, ctdg
