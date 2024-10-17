import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from utils import divide

class Classifier(nn.Module):
    def __init__(self, layer_dims):
        super(Classifier, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(layer_dims)):
            layer = nn.Linear(layer_dims[i - 1], layer_dims[i])
            self.layers.append(layer)
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = torch.tanh(layer(x))
            else:
                x = layer(x)
        return x
    
def perform_classification_task(x, y, ratio, epoch_num, lr, weight_decay, seed):
    device = x.device
    
    idx = np.arange(x.size(0), dtype = np.int64)
    trn_idx, evl_idx = divide(idx, ratio, seed)

    trn_idx = torch.tensor(trn_idx, device = device)
    evl_idx = torch.tensor(evl_idx, device = device)

    trn_x = x[trn_idx]
    trn_y = y[trn_idx]

    evl_x = x[evl_idx]
    evl_y = y[evl_idx]

    input_dim = x.size(-1)
    class_num = y.max().item() + 1

    classifier = Classifier([input_dim, class_num]).to(device)
    optimizer = torch.optim.Adam([{'params': classifier.parameters(), 'weight_decay': weight_decay}], lr = lr)
    criterion = torch.nn.CrossEntropyLoss()

    for i in range(epoch_num):
        outputs = classifier(trn_x)
        loss = criterion(outputs, trn_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    outputs = classifier(trn_x).detach()
    trn_p = torch.argmax(outputs, dim = -1).cpu().numpy()
    trn_y = trn_y.cpu().numpy()

    # trn_ma_f1 = f1_score(trn_y, trn_p, average = 'macro')
    trn_mi_f1 = f1_score(trn_y, trn_p, average = 'micro')

    outputs = classifier(evl_x).detach()
    evl_p = torch.argmax(outputs, dim = -1).cpu().numpy()
    evl_y = evl_y.cpu().numpy()

    evl_ma_f1 = f1_score(evl_y, evl_p, average = 'macro')
    evl_mi_f1 = f1_score(evl_y, evl_p, average = 'micro')

    return trn_mi_f1, evl_ma_f1, evl_mi_f1