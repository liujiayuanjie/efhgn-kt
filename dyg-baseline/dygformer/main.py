import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import torch
import torch.nn as nn

from models.memorymodel import MemoryModel, compute_src_dst_node_time_shifts
from models.dygformer import DyGFormer
from models.modules import MLPClassifier
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler
from utils.dataloader import get_idx_data_loader, get_node_classification_data
from utils.load_configs import get_node_classification_args

from sklearn.metrics import f1_score


if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    args = get_node_classification_args()

    qst_num, usr_num, skl_num, node_raw_features, edge_raw_features, full_data, train_data, test_data = \
        get_node_classification_data(dataset_name=args.dataset_name, test_ratio=args.test_ratio)
    
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)
    
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, test_metric_all_runs = [], []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.load_model_name = f'{args.model_name}_seed{args.seed}'
        args.save_model_name = f'node_classification_{args.model_name}_seed{args.seed}'



        run_start_time = time.time()

        # create model
        if args.model_name in ['TGN']:
            # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
            dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                           time_feat_dim=args.time_feat_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
    
        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")

        node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], cls_num=skl_num, dropout=args.dropout)
        model = nn.Sequential(dynamic_backbone, node_classifier)

        # follow previous work, we freeze the dynamic_backbone and only optimize the node_classifier
        optimizer = create_optimizer(model=model[1], optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)
        # put the node raw messages of memory-based models on device
        if args.model_name in ['TGN']:
            for node_id, node_raw_messages in model[0].memory_bank.node_raw_messages.items():
                new_node_raw_messages = []
                for node_raw_message in node_raw_messages:
                    new_node_raw_messages.append((node_raw_message[0].to(args.device), node_raw_message[1]))
                model[0].memory_bank.node_raw_messages[node_id] = new_node_raw_messages

        loss_func = torch.nn.CrossEntropyLoss()

        # set the dynamic_backbone in evaluation mode
        model[0].eval()

        for epoch in range(args.num_epochs):

            model[1].train()
            
            if args.model_name in ['TGN', 'DyGFormer']:
                model[0].set_neighbor_sampler(full_neighbor_sampler)
            if args.model_name in ['TGN']:
                model[0].memory_bank.__init_memory_bank__()

            train_total_loss, train_y_trues, train_y_predicts = 0.0, [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels, batch_node_labels = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], train_data.node_interact_times[train_data_indices], \
                    train_data.edge_ids[train_data_indices], train_data.labels[train_data_indices], train_data.node_labels[train_data_indices]

                with torch.no_grad():
                    if args.model_name in ['TGN']:
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                              dst_node_ids=batch_dst_node_ids,
                                                                              node_interact_times=batch_node_interact_times,
                                                                              edge_ids=batch_edge_ids,
                                                                              edges_are_positive=True,
                                                                              num_neighbors=args.num_neighbors)
                    elif args.model_name in ['DyGFormer']:
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                              dst_node_ids=batch_dst_node_ids,
                                                                              node_interact_times=batch_node_interact_times)
                    else:
                        raise ValueError(f"Wrong value for model_name {args.model_name}!")
                # get predicted probabilities, shape (batch_size, )
                predicts = model[1](x=batch_dst_node_embeddings)
                labels = torch.from_numpy(batch_node_labels).to(predicts.device)

                loss = loss_func(input=predicts, target=labels)

                train_total_loss += loss.item()
            
                train_y_trues.append(labels.cpu().numpy())
                train_y_predicts.append(torch.argmax(predicts, dim = -1).cpu().numpy())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

            train_total_loss /= (batch_idx + 1)
            train_y_trues = np.concatenate(train_y_trues, axis=0)
            train_y_predicts = np.concatenate(train_y_predicts, axis=0)

            trn_ma_f1 = f1_score(train_y_trues, train_y_predicts, average = 'macro')
            trn_mi_f1 = f1_score(train_y_trues, train_y_predicts, average = 'micro')
            
            train_metrics = trn_mi_f1

            model[0].set_neighbor_sampler(full_neighbor_sampler)
            model.eval()   
                        
            with torch.no_grad():
                if args.model_name in ['TGN']:
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=test_data.src_node_ids,
                                                                            dst_node_ids=test_data.dst_node_ids,
                                                                            node_interact_times=test_data.node_interact_times,
                                                                            edge_ids=test_data.edge_ids,
                                                                            edges_are_positive=True,
                                                                            num_neighbors=args.num_neighbors)
                elif args.model_name in ['DyGFormer']:
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=test_data.src_node_ids,
                                                                            dst_node_ids=test_data.dst_node_ids,
                                                                            node_interact_times=test_data.node_interact_times)
            
            predicts = model[1](x=batch_dst_node_embeddings)
            labels = torch.from_numpy(batch_node_labels).to(predicts.device)
            
            evl_p = torch.argmax(predicts, dim = -1).cpu().numpy()
            evl_y = test_data.node_labels
            
            evl_ma_f1 = f1_score(evl_y, evl_p, average = 'macro')
            evl_mi_f1 = f1_score(evl_y, evl_p, average = 'micro')
            
            test_total_loss = loss.item()
            test_metrics = evl_mi_f1

            print('  '.join((
                'evl_ma_f1: %-.4f' % evl_ma_f1,
                'evl_mi_f1: %-.4f' % evl_mi_f1,
            )))
            