import torch
import torch.nn as nn
from utils import Maximizer, Averager, Batcher
from sklearn.metrics import roc_auc_score, accuracy_score

class DKT(nn.Module):
    def __init__(self, hidden_dim, dropout = 0.0):
        super(DKT, self).__init__()
        self.fnn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.lstm = nn.LSTM(2 * hidden_dim, hidden_dim, batch_first = True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedding, curr, next, rst):
        embedding = self.fnn(embedding)
        curr_emb = embedding[curr]
        rst = rst.unsqueeze(-1)

        input_emb = torch.cat((curr_emb * rst, curr_emb * (1 - rst)), dim = -1)
        input_emb = self.dropout(input_emb)

        output_emb, _ = self.lstm(input_emb)
        output_emb = self.dropout(output_emb)

        y = self.sigmoid((output_emb * embedding[next]).sum(-1))

        return y

def perform_knowledge_tracing_task(qst_fea, trn_seqs, evl_seqs, batch_size, lr, weight_decay, dropout, quit_epoch_num):
    device = qst_fea.device
    hidden_dim = qst_fea.size(-1)
    embedding = qst_fea
    
    dkt = DKT(hidden_dim, dropout).to(device)
    dkt_params = {'params': dkt.parameters(), 'weight_decay': weight_decay}
    parameters = [dkt_params]
    optimizer = torch.optim.Adam(parameters, lr = lr)
    criterion = torch.nn.BCELoss()

    def train():
        dkt.train()

        loss_avg = Averager()
        batcher = Batcher(trn_seqs, batch_size)
        
        for batch in batcher:
            batch = torch.tensor(batch, device = device) # batch_size, seq_len, 2
            qst = batch[:, :, 0]
            rst = batch[:, :, 1]

            curr_qst = qst[:, :-1]
            curr_rst = rst[:, :-1]

            next_qst = qst[:, 1:]
            next_rst = rst[:, 1:]

            pred = dkt(embedding, curr_qst, next_qst, curr_rst)
            mask = next_qst != -1

            loss = criterion(pred[mask], next_rst[mask].float())
            loss_avg.join(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss_avg.get()

    def evaluate(seqs):
        dkt.eval()

        batcher = Batcher(seqs, batch_size)
        auc_avg = Averager()
        acc_avg = Averager()
        
        for batch in batcher:
            batch = torch.tensor(batch, device = device) # batch_size, seq_len, 2
            qst = batch[:, :, 0]
            rst = batch[:, :, 1]

            curr_qst = qst[:, :-1]
            curr_rst = rst[:, :-1]

            next_qst = qst[:, 1:]
            next_rst = rst[:, 1:]

            pred = dkt(embedding, curr_qst, next_qst, curr_rst)
            mask = next_qst != -1

            pred = pred[mask].detach().cpu().numpy()
            true = next_rst[mask].detach().cpu().numpy()

            auc = roc_auc_score(true, pred)
            acc = accuracy_score(true, (pred > 0.5).astype(float))

            auc_avg.join(auc, true.shape[0])
            acc_avg.join(acc, true.shape[0])

        return auc_avg.get(), acc_avg.get()
    
    quit_count = 0

    auc_max = Maximizer()
    max_res = (0.0, 0.0)

    while quit_count <= quit_epoch_num:
        loss = train()
        evl_auc, evl_acc = evaluate(evl_seqs)

        if auc_max.join(evl_auc):
            max_res = (evl_auc, evl_acc)
            quit_count = 0

        quit_count += 1
    
    return max_res