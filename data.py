import numpy as np

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
    
    return qst_num, usr_num, skl_num, qst_skl, events
    
def create_hypergraph_data(qst_num, usr_num, events):
    usr_seqs = [[] for usr in range(usr_num)]
    qst_seqs = [[] for qst in range(qst_num)]

    time_list = []

    for idx, (usr, qst, rst, tme) in enumerate(events):
        time_list.append(tme)
        usr_seqs[usr].append([idx, usr + qst_num])
        qst_seqs[qst].append([idx, qst])
    
    qst_msk = np.array([len(seq) for seq in qst_seqs]) > 0

    tme = np.array(time_list).astype(np.int64)    
    
    seqs = qst_seqs + usr_seqs
    
    vv = []
    for seq in seqs:
        if len(seq) < 2:
            continue
        for (v0, e0), (v1, e1) in zip(seq[: -1], seq[1:]):
            vv.append([v0, v1])
    vv = np.array(vv).astype(np.int64)
    
    ve = []
    for seq in seqs:
        for v, e in seq:
            ve.append([v, e])
    ve = np.array(ve).astype(np.int64)
    
    v_num = events.shape[0]
    e_num = qst_num + usr_num

    return v_num, e_num, vv, ve, tme, qst_seqs, usr_seqs, qst_msk

def sample_evo_flow(v_num, seqs, r):
    evo_flow = [[] for v in range(v_num)]
    
    for seq in seqs:
        if len(seq) == 0:
            continue
        seq_ = [v for v, e in seq]
        fst = seq_[0]
        lst = seq_[-1]
        seq_len = len(seq_)
        seq_ = [fst] * r + seq_ + [lst] * r
        
        for i in range(seq_len):
            v = seq_[i + r]
            evo_flow[v] = seq_[i: i + 2 * r + 1]
    
    evo_flow = np.array(evo_flow).astype(np.int64)

    return evo_flow

def create_seqs(usr_num, events):
    seqs = [[] for usr in range(usr_num)]

    for usr, qst, rst, tme in events:
        seqs[usr].append([qst, rst])

    return seqs

def format_seqs(seqs, seq_len):
    seqs_ = []
    for seq in seqs:
        rmd = len(seq) % seq_len
        addition = 0 if rmd == 0 else seq_len - rmd
        seq_arr = np.array(seq + [[-1, -1]] * addition).reshape(-1, seq_len, 2)
        seqs_ = seqs_ + seq_arr.tolist()
        
    return seqs_