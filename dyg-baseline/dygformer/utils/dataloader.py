from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd


class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False)
    return data_loader

class Data:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray, node_labels: np.ndarray):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.node_labels = node_labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)

def read_csv(path):
    fp = open(path, 'r')
    rows = fp.read().split('\n')[: -1]
    split = lambda row: [int(e) if e.isdigit() else e for e in row.split(',')]
    rows = [split(row) for row in rows]

    return rows

def divide(items, ratio, seed = 0):
    item_num = len(items)
    rdm = np.random.RandomState(seed)
    idx = np.arange(item_num)
    rdm.shuffle(idx)
    div_num = int(ratio * item_num)
    trn, evl = items[idx[: div_num]], items[idx[div_num:]]
    return trn, evl

def get_node_classification_data(dataset_name: str, test_ratio: float):
    graph_df = read_csv('../../dataset/{}/event.csv'.format(dataset_name))
    graph_df = np.array(graph_df).astype(np.longlong)
    
    qst_skl = read_csv('../../dataset/{}/qst_skl.csv'.format(dataset_name))
    qst_skl = np.array(qst_skl).astype(np.longlong)[:, 1]
    
    qst = read_csv('../../dataset/{}/qst.csv'.format(dataset_name))
    qst_num = np.array(qst).shape[0]
    usr = read_csv('../../dataset/{}/usr.csv'.format(dataset_name))
    usr_num = np.array(usr).shape[0]
    skl = read_csv('../../dataset/{}/skl.csv'.format(dataset_name))
    skl_num = np.array(skl).shape[0]
    
    src_node_ids = graph_df[:, 0] + qst_num
    dst_node_ids = graph_df[:, 1]
    labels = graph_df[:, 2]
    node_interact_times = graph_df[:, 3].astype(np.float64)
    
    full_num = src_node_ids.shape[0]
    
    edge_ids = np.arange(full_num, dtype = np.int64) #+ 1
         
    idx = np.arange(qst_num, dtype = np.int64)
    trn_idx, evl_idx = divide(idx, 1 - test_ratio, 0)
    
    qst_idx = [-1] * qst_num
    for idx, qst in enumerate(dst_node_ids[::-1]):
        if qst_idx[qst] == -1:
            qst_idx[qst] = full_num - idx - 1
    qst_idx = np.array(qst_idx, dtype = np.int64)

    
    full_data = Data(
        src_node_ids=src_node_ids, 
        dst_node_ids=dst_node_ids, 
        node_interact_times=node_interact_times, 
        edge_ids=edge_ids, 
        labels=labels,
        node_labels = qst_skl[dst_node_ids],
    )
    train_data = Data(
        src_node_ids=src_node_ids[qst_idx[trn_idx]], 
        dst_node_ids=dst_node_ids[qst_idx[trn_idx]],
        node_interact_times=node_interact_times[qst_idx[trn_idx]],
        edge_ids=edge_ids[qst_idx[trn_idx]], labels=labels[qst_idx[trn_idx]],
        node_labels = qst_skl[dst_node_ids][qst_idx[trn_idx]],
    )
    test_data = Data(
        src_node_ids=src_node_ids[qst_idx[evl_idx]], 
        dst_node_ids=dst_node_ids[qst_idx[evl_idx]],
        node_interact_times=node_interact_times[qst_idx[evl_idx]], 
        edge_ids=edge_ids[qst_idx[evl_idx]], 
        labels=labels[qst_idx[evl_idx]],
        node_labels = qst_skl[dst_node_ids][qst_idx[evl_idx]],
    )
    
    node_raw_features = np.random.normal(size = (qst_num + usr_num, 64))
    edge_raw_features = np.random.normal(size = (2, 64))[labels]
    
    return qst_num, usr_num, skl_num, node_raw_features, edge_raw_features, full_data, train_data, test_data
