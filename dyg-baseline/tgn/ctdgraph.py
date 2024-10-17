import numpy as np
import torch

class NeighborFinder:
    def __init__(self, node_num, neighbor_data, device = None):
        self.device = torch.device('cpu') if device == None else device
        self.node_num = node_num
        self.neighbor_data = neighbor_data

        neighbor_data_row = []

        for n in range(node_num):
            data = neighbor_data[n]
            neighbor_data_row += data

        neighbor_num = [len(neighbors) for neighbors in neighbor_data]
        neighbor_start = [0] + [num for num in neighbor_num]
        neighbor_start.pop()
        neighbor_start = np.cumsum(neighbor_start).tolist()

        self.neighbor_data_row = torch.tensor(neighbor_data_row).to(device)
        self.neighbor_num = torch.tensor(neighbor_num).to(device)
        self.neighbor_start = torch.tensor(neighbor_start).to(device)
    
    def find(self, nodes, neighbor_size, indices = None, mode = 'random'): # random proximity
        device = self.device

        nodes = torch.tensor(nodes).long().to(device)
        node_size = nodes.size(0)

        if indices == None:
            indices = self.neighbor_num[nodes] - 1
        else:
            indices = torch.tensor(indices).long().to(device)
        
        if mode == 'random':
            neighbor_idx = torch.randint(int(1e10), (node_size, neighbor_size)).to(device)
            neighbor_mask = torch.zeros_like(neighbor_idx).to(device)
            neighbor_mask[indices <= 0] = 1

            indices[indices <= 0] = 1
            neighbor_idx = neighbor_idx % indices.unsqueeze(-1)
            neighbor_idx = self.neighbor_start[nodes].unsqueeze(-1) + neighbor_idx
            neighbor_idx[neighbor_mask == 1] = 0

            neighbor_idx[neighbor_idx >= self.neighbor_data_row.size(0)] = 0

            neighbor_data = self.neighbor_data_row[neighbor_idx]

        elif mode == 'proximity':
            neighbor_range = torch.arange(-neighbor_size, 0)
            neighbor_range = neighbor_range.unsqueeze(0).to(device)
            # print(indices.shape, neighbor_range.shape)
            neighbor_idx = indices.unsqueeze(-1) + neighbor_range

            neighbor_mask = torch.zeros_like(neighbor_idx).to(device)
            neighbor_mask[neighbor_idx < 0] = 1

            neighbor_idx[neighbor_idx < 0] = 0
            neighbor_idx = self.neighbor_start[nodes].unsqueeze(-1) + neighbor_idx
            neighbor_idx[neighbor_mask == 1] = 0

            neighbor_data = self.neighbor_data_row[neighbor_idx]

        return neighbor_data, neighbor_mask

class CTDGraph:
    def __init__(self, node_num, edges, device = None):
        self.device = torch.device('cpu') if device == None else device
        self.node_num = node_num
        self.edges = [[n1, n2, r, t] for n1, n2, r, t in edges]

        self.neighbor_data = CTDGraph.generate_neighbor_data(node_num, edges)
        self.edge_data = CTDGraph.generate_edge_data(node_num, self.neighbor_data)
        self.head_edge_data, self.tail_edge_data = CTDGraph.separate_edge_data(self.edge_data)
        self.final_node = CTDGraph.generate_final_node(self.neighbor_data)

        self.neighbor_finder = NeighborFinder(node_num, self.neighbor_data, device = device)
    
    @staticmethod
    def generate_neighbor_data(node_num, edges):
        neighbor_data = [[] for n in range(node_num)]

        for n1, n2, r, t in edges:
            neighbor_data[n1].append([n2, r, t])
            neighbor_data[n2].append([n1, r, t])
        
        for n in range(node_num):
            data = sorted(neighbor_data[n], key = lambda item: item[-1])
            neighbor_data[n] = [[n, r, t, i] for i, (n, r, t) in enumerate(data)]
        
        return neighbor_data
    
    @staticmethod
    def generate_edge_data(node_num, neighbor_data):
        edge_idx_map = {}

        for n1 in range(node_num):
            for n2, r, t, i in neighbor_data[n1]:
                n1_, n2_, idx = [n1, n2, 0] if n1 < n2 else [n2, n1, 1]
                key = '%d-%d-%d-%d' % (n1_, n2_, r, t)

                if key not in edge_idx_map:
                    edge_idx_map[key] = [None] * 2

                edge_idx_map[key][idx] = i
        
        edge_data = []

        for key in edge_idx_map:
            data = [int(e) for e in key.split('-')]
            data = [*data, *edge_idx_map[key]]
            edge_data.append(data)

        edge_data = sorted(edge_data, key = lambda data: data[3])
        
        return edge_data

    @staticmethod
    def generate_final_node(neighbor_data):
        node_data = []

        for n, neighbors in enumerate(neighbor_data):
            if len(neighbors) == 0:
                node_data.append([n, 0, 0, 0])
            else:
                _, r, t, i = neighbors[-1]
                node_data.append([n, r, t, i])
        
        return node_data

    @staticmethod
    def separate_edge_data(edge_data):
        head_edge_data = []
        tail_edge_data = []

        flag_set = set()

        for n1, n2, r, t, i1, i2 in edge_data[:: -1]:
            if n2 in flag_set:
                head_edge_data.append([n1, n2, r, t, i1, i2])
            else:
                tail_edge_data.append([n1, n2, r, t, i1, i2])
                flag_set.add(n2)
        
        return head_edge_data, tail_edge_data
