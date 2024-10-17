import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy  as np

class TGN(nn.Module):
    def __init__(self, v_num, r_num, emb_dim, head_num, nbr_num, cls_num, ctdg, dropout):
        super(TGN, self).__init__()
        self.ctdg = ctdg
        self.nbr_num = nbr_num
        self.dropout = dropout

        self.v_emb = nn.Embedding(v_num, emb_dim)
        self.r_emb = nn.Embedding(r_num, emb_dim)
        self.time_encode = TimeEncode(emb_dim)

        self.multi_head_target = MultiheadAttention(
            head_num = head_num, 
            h_dim = emb_dim, 
            q_dim = 2 * emb_dim, 
            k_dim = 3 * emb_dim, 
            v_dim = 3 * emb_dim
        )

        self.marge = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.score = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 1),
            nn.Sigmoid()
        )
        
        self.cls = nn.Linear(emb_dim, cls_num)

    def forward(self, node_data):
        dev = node_data.device
        node_size = node_data.size(0)

        v, r, t, i = node_data.T

        nbr_data, nbr_mask = self.ctdg.neighbor_finder.find(v, self.nbr_num, i, mode = 'proximity')
        
        time_emb = self.time_encode(torch.zeros(node_size).to(dev))
        time_emb = time_emb.unsqueeze(0)
        edge_time_emb = self.time_encode((t.unsqueeze(-1) - nbr_data[:, :, 2]).float())
        
        v_emb = self.v_emb(v)
        nbr_emb = self.v_emb(nbr_data[:, :, 0])
        r_emb = self.r_emb(nbr_data[:, :, 1])
        
        v_emb = F.dropout(v_emb, self.dropout)
        nbr_emb = F.dropout(nbr_emb, self.dropout)
        r_emb = F.dropout(r_emb, self.dropout)
        
        query = torch.cat([v_emb.unsqueeze(0), time_emb], dim = -1)
        query = query.permute([1, 0, 2])
        key = torch.cat([nbr_emb, r_emb, edge_time_emb], dim = -1)

        output = self.multi_head_target(query, key, key, nbr_mask)

        flag = (~nbr_mask).sum(-1) == 0
        output = output.squeeze(0)
        output[flag] = torch.zeros_like(output[flag]).to(dev).float()
        output = output.squeeze(1)
        
        # output = self.marge(torch.cat((output, v_emb), dim = -1))
        
        cls_res = self.cls(F.dropout(output, self.dropout))

        return cls_res

class TimeEncode(nn.Module):
    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim))).
            float().
            reshape(dim, -1)
        )
        self.w.bias = nn.Parameter(torch.zeros(dim).float())

    def forward(self, t):
        t = t.unsqueeze(dim = -1)
        output = torch.cos(self.w(t))
        return output

class MultiheadAttention(nn.Module):
    def __init__(self, head_num, h_dim, q_dim, k_dim, v_dim):
        super(MultiheadAttention, self).__init__()
        self.head_num = head_num
        self.h_dim = h_dim
        self.head_dim = h_dim // head_num
        
        self.query_proj = nn.Linear(q_dim, h_dim)
        self.key_proj = nn.Linear(k_dim, h_dim)
        self.value_proj = nn.Linear(v_dim, h_dim)
        
        self.final_proj = nn.Linear(h_dim, h_dim)
        
    def forward(self, query, key, value, mask = None):
        batch_size, seq_len, _ = key.size()
        query_size = query.size(1)
        query = self.query_proj(query).view(batch_size, query_size, self.head_num, self.head_dim)
        key = self.key_proj(key).view(batch_size, seq_len, self.head_num, self.head_dim)
        value = self.value_proj(value).view(batch_size, seq_len, self.head_num, self.head_dim)

        scores = (query * key / (self.h_dim ** 0.5)).sum(-1)
        
        if mask is not None:
            scores[mask] = -1e9

        att = F.softmax(scores, dim=-1)
        output = (att.unsqueeze(-1) * value).sum(1)
    
        output = output.view(batch_size, query_size, self.h_dim)
        output = self.final_proj(output)
        
        return output
