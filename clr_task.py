import torch
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def perform_clustering_task(x, y, cluster_num, max_iters):
    y = y.cpu().numpy()
    x_num, _ = x.shape

    clr_idx = np.random.choice(x_num, cluster_num, replace = False)
    clr_idx = torch.tensor(clr_idx, device = x.device).long()
    centroids = x[clr_idx]

    for i in range(max_iters):
        distances = torch.cdist(x, centroids)
        cluster_assignments = torch.argmin(distances, dim = -1)
        new_centroids = torch.stack([x[cluster_assignments == k].mean(dim = 0) for k in range(cluster_num)])
        
        if torch.all(torch.eq(new_centroids, centroids)):
            break
        
        centroids = new_centroids
    
    z = cluster_assignments.cpu().numpy()
    
    nmi = normalized_mutual_info_score(y, z)
    ari = adjusted_rand_score(y, z)

    return nmi, ari