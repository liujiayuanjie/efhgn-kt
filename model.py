import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter

class HyEdgeEmb(nn.Module):
    def __init__(self, e_num, emb_dim):
        super(HyEdgeEmb, self).__init__()
        self.embed = nn.Parameter(torch.zeros((e_num, emb_dim)))
        nn.init.normal_(self.embed)
    
    def forward(self):
        return self.embed

class VertexEmb(nn.Module):
    def __init__(self, v_num, emb_dim):
        super(VertexEmb, self).__init__()
        self.embed = nn.Parameter(torch.zeros((v_num, emb_dim)))
        nn.init.normal_(self.embed)
    
    def forward(self):
        return self.embed

class TimeEmb(nn.Module):
    def __init__(self, emb_dim):
        super(TimeEmb, self).__init__()
        self.emb_dim = emb_dim
        self.emb_dim = int(emb_dim / 2)
        self.w = nn.Parameter(torch.zeros((1, self.emb_dim)))
        nn.init.normal_(self.w)
    
    def forward(self, t):
        t = t.unsqueeze(-1)
        t_cos = torch.cos(t * self.w).squeeze(-1)
        t_sin = torch.sin(t * self.w).squeeze(-1)
        t_emb = torch.cat((t_cos, t_sin), dim = -1)
        t_emb = t_emb / (self.emb_dim ** 0.5)
        return t_emb

class E2VAggLayer(nn.Module):
    def __init__(self, input_dim, v_num, tau):
        super(E2VAggLayer, self).__init__()
        self.v_num = v_num
        self.tau = tau
        
        self.a_v = nn.Parameter(torch.zeros(input_dim, 1))
        self.a_t = nn.Parameter(torch.zeros(input_dim, 1))
        self.a_e = nn.Parameter(torch.zeros(input_dim, 1))
        
        nn.init.xavier_normal_(self.a_v.data, gain = 2 ** 0.5)
        nn.init.xavier_normal_(self.a_t.data, gain = 2 ** 0.5)
        nn.init.xavier_normal_(self.a_e.data, gain = 2 ** 0.5)

    def forward(self, v_fea, t_fea, e_fea, ve):
        a_v = torch.mm(v_fea, self.a_v)
        a_t = torch.mm(t_fea, self.a_t)
        a_e = torch.mm(e_fea, self.a_e)

        v, e = ve.T

        a = F.tanh(a_v[v] + a_t[v] + a_e[e]) / self.tau
        a = scatter_softmax(a, index = v, dim = 0)
        v_fea_ = scatter(a * e_fea[e], index = v, dim = 0, dim_size = self.v_num, reduce = 'sum')

        return v_fea_

class V2EAggLayer(nn.Module):
    def __init__(self, input_dim, e_num, tau):
        super(V2EAggLayer, self).__init__()
        self.e_num = e_num
        self.tau = tau
        
        self.a_v = nn.Parameter(torch.zeros(input_dim, 1))
        self.a_t = nn.Parameter(torch.zeros(input_dim, 1))
        self.a_e = nn.Parameter(torch.zeros(input_dim, 1))
        
        nn.init.xavier_normal_(self.a_v.data, gain = 2 ** 0.5)
        nn.init.xavier_normal_(self.a_t.data, gain = 2 ** 0.5)
        nn.init.xavier_normal_(self.a_e.data, gain = 2 ** 0.5)

    def forward(self, v_fea, t_fea, e_fea, ve):
        a_v = torch.mm(v_fea, self.a_v)
        a_t = torch.mm(t_fea, self.a_t)
        a_e = torch.mm(e_fea, self.a_e)

        v, e = ve.T

        a = F.tanh(a_v[v] + a_t[v] + a_e[e]) / self.tau
        a = scatter_softmax(a, index = e, dim = 0)
        e_fea_ = scatter(a * v_fea[v], index = e, dim = 0, dim_size = self.e_num, reduce = 'sum')

        return e_fea_

class HyGAttEmb(nn.Module):
    def __init__(self, v_num, e_num, emb_dim, layer_num, ve_tau, ev_tau):
        super(HyGAttEmb, self).__init__()
        self.v2e_layers = nn.ModuleList([V2EAggLayer(emb_dim, e_num, ve_tau) for i in range(layer_num)])
        self.e2v_layers = nn.ModuleList([E2VAggLayer(emb_dim, v_num, ev_tau) for i in range(layer_num)])

    def forward(self, v_emb, t_emb, e_emb, ve):
        v_fea_stack = [v_emb]
        e_fea_stack = [e_emb]

        for v2e_layer, e2v_layer in zip(self.v2e_layers, self.e2v_layers):
            v_fea_ = v_fea_stack[-1]
            e_fea_ = e_fea_stack[-1]
            
            e_fea_ = v2e_layer(v_fea_, t_emb, e_fea_, ve)
            e_fea_stack.append(F.relu(e_fea_))
            
            v_fea_ = e2v_layer(v_fea_, t_emb, e_fea_, ve)
            v_fea_stack.append(F.relu(v_fea_))

        v_fea = torch.stack(v_fea_stack, dim = 1)
        v_fea = v_fea.mean(1)
        
        e_fea = torch.stack(e_fea_stack, dim = 1)
        e_fea = e_fea.mean(1)
        
        return v_fea, e_fea

class EvoFlowAttNet(nn.Module):
    def __init__(self, emb_dim):
        super(EvoFlowAttNet, self).__init__()
        self.w = nn.Linear(2 * emb_dim, 2 * emb_dim, bias = False)
    
    def forward(self, v_fea, t_emb, ef):
        v = v_fea[ef]
        q = torch.cat((v_fea, t_emb), dim = -1) 
        k = torch.cat((v, t_emb[ef]), dim = -1) 
        
        r = torch.bmm(k, self.w(q).unsqueeze(-1)) 
        r = torch.softmax(r, dim = 1)
        
        ef_fea = (r * v).sum(dim = 1)
        
        return ef_fea

class EvoFlowAggLayer(nn.Module):
    def __init__(self, input_dim, ef_num):
        super(EvoFlowAggLayer, self).__init__()
        self.ef_nets = nn.ModuleList([EvoFlowAttNet(input_dim) for i in range(ef_num)])
        self.mlp = nn.Sequential(
            nn.Linear(2 * input_dim, 1, bias = False),
            nn.LeakyReLU(),
        )

    def forward(self, v_fea, t_emb, efs):
        ef_feas = []
        ef_ws = []

        for ef_net, ef in zip(self.ef_nets, efs):
            ef_fea = ef_net(v_fea, t_emb, ef)
            ef_feas.append(F.relu(ef_fea))
            ef_ws.append(self.mlp(torch.cat((v_fea, ef_fea), dim = -1)))
            
        ef_fea = torch.stack(ef_feas, dim = 1)
        ef_w = torch.stack(ef_ws, dim = 1)
        ef_w = torch.softmax(ef_w, dim = 1)
        
        ef_fea = (ef_fea * ef_w).sum(1)
        
        return ef_fea

class EvoFlowEmb(nn.Module):
    def __init__(self, emb_dim, ef_num, layer_num):
        super(EvoFlowEmb, self).__init__()
        self.ef_att_layers = nn.ModuleList([EvoFlowAggLayer(emb_dim, ef_num) for i in range(layer_num)])
    
    def forward(self, v_emb, t_emb, efs):
        ef_fea_stack = [v_emb]
        
        for ef_att_layer in self.ef_att_layers:
            ef_fea_ = ef_fea_stack[-1]
            ef_fea_ = ef_att_layer(ef_fea_, t_emb, efs)
            ef_fea_stack.append(F.relu(ef_fea_))
        
        ef_fea = torch.stack(ef_fea_stack, dim = 1)
        ef_fea = ef_fea.mean(1)
        
        return ef_fea