# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import random
import h5py
import os
import re
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from torch.cuda.amp import autocast
ENABLE_AUTOCAST = False
import gc
import glob
from collections import namedtuple
from tensorflow.keras.utils import pad_sequences


def save_hdf5(data_array, data_path, key="data"):
    logging.info("Saving data to h5: " + data_path)
    dir_name = os.path.dirname(data_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with h5py.File(data_path, 'a') as hf:
        hf.create_dataset(key, data=data_array)


def load_hdf5(data_path, key=None, verbose=True):
    if verbose:
        logging.info('Loading data from h5: ' + data_path)
    with h5py.File(data_path, 'r') as hf:
        if key is not None:
            data_array = hf[key][()]
        else:
            data_array = hf[list(hf.keys())[0]][()]
    return data_array


# NOTE: this function can not differentiate repeated values in each row of 'u', 
# therefore 'u' is expected to avoid repeated values in each row
@autocast(enabled=ENABLE_AUTOCAST)
def jaccard_similarity(u, v, padding_idx=0, enable_clean=False):
    assert u.ndim == v.ndim == 2, "jaccard_torch requires u: XxD, v: YxD' inputs"
    if isinstance(u, np.ndarray) and isinstance(v, np.ndarray):
        u_lens = (u != padding_idx) # XxD
        v_lens = (v != padding_idx) # YxD'
        # Xx1xDx1, 1xYx1xD'
        sims = (np.expand_dims(u, axis=(-1, -3)) == np.expand_dims(v, axis=(0, 2))) # XxYxDxD'
        # XxYxDxD', XxD => XxYxD, Xx1xD => XxY
        count_intersection = (sims.any(-1) * np.expand_dims(u_lens, axis=1)).sum(-1) # XxY
        if enable_clean:
            del sims
            gc.collect()
        # XxD, YxD', XxY => Xx1, Y, XxY => XxY
        count_union = u_lens.sum(-1, keepdims=True) + v_lens.sum(-1) - count_intersection # XxY
    elif isinstance(u, torch.Tensor) and isinstance(v, torch.Tensor):
        u_lens = (u != padding_idx) # XxD
        v_lens = (v != padding_idx) # YxD'
        # Xx1xDx1, 1xYx1xD'
        sims = (u.unsqueeze(-1).unsqueeze(-3) == v.unsqueeze(0).unsqueeze(2)) # XxYxDxD'
        # XxYxDxD', XxD => XxYxD, Xx1xD => XxY
        count_intersection = (sims.any(-1) * u_lens.unsqueeze(1)).sum(-1) # XxY
        if enable_clean:
            del sims
            gc.collect()
        # XxD, YxD', XxY => Xx1, Y, XxY => XxY
        count_union = u_lens.sum(-1, keepdim=True) + v_lens.sum(-1) - count_intersection # XxY
    if enable_clean:
        del u_lens, v_lens
        gc.collect()
    results = count_intersection / (count_union + 1e-3)
    if enable_clean:
        del count_intersection, count_union
        gc.collect()
    return results


# map the elements of queries to the indices of keys
# NOT safe: need to assure that keys and queries have same the value set, or will cause IndexError
@autocast(enabled=ENABLE_AUTOCAST)
def map_indices(keys, queries, missing=-1, is_key_sorted=False, enable_clean=False):
    assert keys.ndim == 1
    if isinstance(keys, np.ndarray) and isinstance(queries, np.ndarray):
        if is_key_sorted:
            sorter = np.arange(len(keys))
        else:
            sorter = np.argsort(keys, kind='mergesort')
        insertion = np.searchsorted(keys, queries, sorter=sorter)
    elif isinstance(keys, torch.Tensor) and isinstance(queries, torch.Tensor):
        if is_key_sorted:
            sorter = torch.arange(len(keys), device=keys.device)
        else:
            sorter = torch.argsort(keys, stable=True)
        insertion = torch.searchsorted(keys, queries, sorter=sorter)
    else:
        raise TypeError(f"The type of 'keys' ({type(keys)}) doesn't match the 'queries' ({type(queries)})")
    
    indices = sorter[insertion]
    invalid = keys[indices] != queries
    indices[invalid] = missing
    if enable_clean:
        del sorter, insertion, invalid
        gc.collect()
    return indices


def graph_collate_fn(batch):
    if len(batch[0]) == 2:
        batch_graphs, batch_labels = map(list, zip(*batch))
        batch_graphs = dgl.batch(batch_graphs)
        batch_labels = torch.from_numpy(np.stack(batch_labels))
        return batch_graphs, batch_labels
    elif len(batch[0]) == 4:
        batch_graphs, batch_labels, batch_retr_values, batch_retr_lens = map(list, zip(*batch))
        batch_graphs = dgl.batch(batch_graphs)
        batch_labels = torch.from_numpy(np.stack(batch_labels))
        batch_retr_values = torch.from_numpy(np.stack(batch_retr_values))
        batch_retr_lens = torch.from_numpy(np.stack(batch_retr_lens))
        return batch_graphs, batch_labels, batch_retr_values, batch_retr_lens


class PETGraphProcessor:
    @staticmethod
    def convert_indices(X, feature_specs):
        offset = 0
        for _, feature_spec in feature_specs.items():
            X[..., feature_spec["index"]] += offset
            offset += feature_spec["vocab_size"]
        return X

    @staticmethod
    def build_instance_graph(X_i, y_i):  
        # X_i: F, y_i: () or
        # X_i: (1+K)xF, y_i: (1+K)
        y_i = y_i.copy() # make a copy
        if y_i.ndim == 0:
            y_i = np.expand_dims(y_i, axis=0)
        y_i[0] = 2 # the label embedding_id of target instance node is set to 2 ([MASK])

        ## declare graph topology
        count_target_instances = 1
        count_instances = y_i.shape[0] # val=1 or (1+K)
        feature_nodes = X_i.reshape(-1) + count_instances  # F or (1xK)*F, add offset
        instance_nodes = np.repeat(np.arange(count_instances), X_i.shape[-1]) # 1 => F or (1+K) => (1+K)*F

        # map ids to continous indices
        all_nodes = np.concatenate((instance_nodes, feature_nodes))  # (1+K)*F*2
        unique_node_ids = np.unique(all_nodes)
        mapped_instance_nodes = map_indices(unique_node_ids, instance_nodes, is_key_sorted=True)
        mapped_feature_nodes = map_indices(unique_node_ids, feature_nodes, is_key_sorted=True)

        # bidirectional edges
        edge_src = np.concatenate((mapped_feature_nodes, mapped_instance_nodes))  # (1+K)*F*2
        edge_dst = np.concatenate((mapped_instance_nodes, mapped_feature_nodes))  # (1+K)*F*2
        graph = dgl.graph((edge_src, edge_dst))

        ## declare graph attributes
        # the label_id of feature node is set to 2 ([MASK]), not used in the training
        labels = np.concatenate((y_i, [2] * (graph.num_nodes() - len(y_i))))
        graph.ndata['label'] = torch.tensor(labels).long()
        original_node_ids = unique_node_ids - count_instances  # remove the offset
        graph.ndata['original_node_ids'] = torch.tensor(original_node_ids).long()
        graph.ndata['is_target'] = (graph.nodes() < count_target_instances)
        graph.ndata['is_instance'] = (graph.nodes() < count_instances)
        graph.ndata['is_feature'] = (graph.nodes() >= count_instances)

        return graph


# 只支持BM25匹配，不支持精确匹配
@autocast(enabled=ENABLE_AUTOCAST)
def BM25_topk_retrieval_v1(
    db_np_data: np.ndarray,
    qry_np_data: np.ndarray,
    qry_batch_size: int = None,
    db_chunk_size: int = None,
    device: str = 'cpu',
    topK: int = 10,
    **kwargs
    ):
    ResultsNamedTuple = namedtuple("ResultsNameTuple", ["values", "indices", "lens"])
    def sort_results(values, indices):
        # values: BxK, indices: BxK
        drop_mask = (values == 0)  # BxK
        indices[drop_mask] = -1
        results = torch.sort(values, descending=True)
        values = results.values
        indices = torch.gather(indices, -1, results.indices)
        lens = drop_mask.shape[-1] - drop_mask.sum(-1)  # B
        return ResultsNamedTuple(values, indices, lens)
    
    def map_data_to_IDF_v1(np_data, IDF_stats):
        IDF_np_data = np.zeros_like(np_data, dtype=float)
        for i, col_IDF_stats in enumerate(IDF_stats):
            IDF_np_data[:, i] = np.vectorize(lambda x: col_IDF_stats.get(x, 0))(np_data[:, i])
        return IDF_np_data
    
    def map_data_to_IDF_v2(np_data, IDF_stats):
        # NOT safe: need to assure that the query and the db have the same value set, or will cause IndexError
        IDF_np_data = []
        for i, col_IDF_stats in enumerate(IDF_stats):
            IDF_np_data.append(col_IDF_stats.values[map_indices(col_IDF_stats.index.to_numpy(), np_data[:, i], missing=-1)])
        return np.stack(IDF_np_data, axis=-1)
    
    map_data_to_IDF = map_data_to_IDF_v1
    
    # pre-compute the IDF statists for each column in the database
    N = len(db_np_data)
    IDF_stats = []
    db_df = pd.DataFrame(db_np_data)
    for col in db_df:
        col_IDF_stats = db_df[col].value_counts()
        col_IDF_stats = np.log((N - col_IDF_stats + 0.5) / (col_IDF_stats + 0.5))
        col_IDF_stats[-1] = 0
        IDF_stats.append(col_IDF_stats)

    # if not process by chunks, then load the whole db_data to the device
    if db_chunk_size is None:
        db_data = torch.from_numpy(db_np_data).to(device)
    # if not specific qry_batch_size, then process the whole qry_data at one time
    qry_batch_size = len(qry_np_data) if qry_batch_size is None else qry_batch_size

    topK_values = np.zeros((len(qry_np_data), topK), dtype=float)
    topK_indices = np.full((len(qry_np_data), topK), -1, dtype=int)
    topK_indices_len = np.zeros(len(qry_np_data), dtype=int)
    for qry_idx in tqdm(range(0, len(qry_np_data), qry_batch_size), desc="retrieve samples"):
        qry_data_batch = qry_np_data[qry_idx: qry_idx + qry_batch_size]
        # map the IDF statists to each position in qry_data_batch
        qry_IDF_data_batch = map_data_to_IDF(qry_data_batch, IDF_stats)
        qry_data_batch = torch.from_numpy(qry_data_batch).to(device)
        qry_IDF_data_batch = torch.from_numpy(qry_IDF_data_batch).to(device)
        if db_chunk_size is None:
            # B: qry_batch_size, N: database_size, F: field_num
            # (Bx1xF compare 1xNxF) * Bx1xF => BxNxF => BxN
            BM25_values_batch = ((qry_data_batch.unsqueeze(1) == db_data.unsqueeze(0)) * qry_IDF_data_batch.unsqueeze(1)).sum(-1)
            topK_results_batch = torch.topk(BM25_values_batch, k=topK)
            topK_results_batch = sort_results(topK_results_batch.values, topK_results_batch.indices)
            topK_indices_batch = topK_results_batch.indices.cpu().numpy()  # BxK
            topK_values_batch = topK_results_batch.values.cpu().numpy()  # BxK
            topK_lens_batch = topK_results_batch.lens.cpu().numpy()  # B
        else:
            local_topK_values_batch = []
            local_topK_indices_batch = []
            for db_idx in range(0, len(db_np_data), db_chunk_size):
                db_data = torch.from_numpy(db_np_data[db_idx: db_idx + db_chunk_size]).to(device)
                # B: qry_batch_size, C: database_chunk_size, F: field_num
                # (Bx1xF compare 1xNxF) * Bx1xF => BxCxF => BxC
                local_BM25_values_batch = ((qry_data_batch.unsqueeze(1) == db_data.unsqueeze(0)) * qry_IDF_data_batch.unsqueeze(1)).sum(-1)
                local_results = torch.topk(local_BM25_values_batch, k=topK)
                local_topK_values_batch.append(local_results.values) # BxK
                local_topK_indices_batch.append(local_results.indices + db_idx) # BxK
            # X: database_chunk_num, BxXK
            local_topK_values_batch = torch.cat(local_topK_values_batch, dim=-1) 
            local_topK_indices_batch = torch.cat(local_topK_indices_batch, dim=-1)
            # aggregate the chunk-wise topK results, rank and finally get the global topK results
            topK_results_batch = torch.topk(local_topK_values_batch, k=topK)
            topK_indices_batch = torch.gather(
                local_topK_indices_batch, 
                dim=-1, 
                index=topK_results_batch.indices
            )
            topK_results_batch = sort_results(topK_results_batch.values, topK_indices_batch)
            topK_indices_batch = topK_results_batch.indices.cpu().numpy()  # BxK
            topK_values_batch = topK_results_batch.values.cpu().numpy()  # BxK
            topK_lens_batch = topK_results_batch.lens.cpu().numpy()  # B
        topK_values[qry_idx: qry_idx + qry_batch_size] = topK_values_batch
        topK_indices[qry_idx: qry_idx + qry_batch_size] = topK_indices_batch
        topK_indices_len[qry_idx: qry_idx + qry_batch_size] = topK_lens_batch
    return ResultsNamedTuple(topK_values, topK_indices, topK_indices_len)


# 添加精确匹配（用类似BM25的矩阵乘法方式实现精确匹配过滤）
@autocast(enabled=ENABLE_AUTOCAST)
def BM25_topk_retrieval_v2(
    db_np_data: np.ndarray,
    qry_np_data: np.ndarray,
    exact_match_col_indices: list = None,
    qry_batch_size: int = None,
    db_chunk_size: int = None,
    device: str = 'cpu',
    topK: int = 10,
    enable_clean: bool = False,
    **kwargs
    ):
    ResultsNamedTuple = namedtuple("ResultsNameTuple", ["values", "indices", "lens"])

    def sort_results(values, indices):
        # values: BxK, indices: BxK
        drop_mask = (values == 0) # BxK
        indices[drop_mask] = -1
        results = torch.sort(values, descending=True)
        values = results.values
        indices = torch.gather(indices, -1, results.indices)
        lens = drop_mask.shape[-1] - drop_mask.sum(-1) # B
        return ResultsNamedTuple(values, indices, lens)

    def padded_topk(input_values, K, index_offs=None):
        assert input_values.ndim == 2, "input shape must be [BxN]"
        output_lens = torch.zeros_like(input_values[:, 0], dtype=int)
        if K >= input_values.shape[-1]:
            output_values = F.pad(input_values, (0, K - input_values.shape[-1]))
            output_indices = torch.zeros_like(output_values, dtype=torch.long)
            for col_i in range(input_values.shape[-1]):
                output_indices[:, col_i] = col_i
            if index_offs:
                output_indices += index_offs
            output_indices[:, input_values.shape[-1]:] = -1
            output_lens[:] = input_values.shape[-1]
        else:
            output_results = torch.topk(input_values, k=K)
            output_values = output_results.values
            output_indices = output_results.indices
            if index_offs:
                output_indices += index_offs
            output_lens[:] = K
        return ResultsNamedTuple(output_values, output_indices, output_lens)
    
    def masked_gather(input, index, mask_index_value=-1):
        if mask_index_value not in index:
            return torch.gather(input, -1, index)
        else:
            mask = (index == mask_index_value)
            index[mask] = 0
            results = torch.gather(input, -1, index)
            results[mask] = mask_index_value
            if enable_clean:
                del mask
                gc.collect()
            return results
    
    def map_data_to_IDF_v1(np_data, IDF_stats):
        IDF_np_data = np.zeros_like(np_data, dtype=float)
        for i, col_IDF_stats in enumerate(IDF_stats):
            IDF_np_data[:, i] = np.vectorize(lambda x: col_IDF_stats.get(x, 0))(np_data[:, i])
        return IDF_np_data
    
    def map_data_to_IDF_v2(np_data, IDF_stats):
        # NOT safe: need to assure that the query and the db have the same value set, or will cause IndexError
        IDF_np_data = []
        for i, col_IDF_stats in enumerate(IDF_stats):
            IDF_np_data.append(col_IDF_stats.values[map_indices(col_IDF_stats.index.to_numpy(), np_data[:, i], missing=-1)])
        return np.stack(IDF_np_data, axis=-1)
    
    map_data_to_IDF = map_data_to_IDF_v1
    
    if exact_match_col_indices:
        db_df = pd.DataFrame(db_np_data)
        exm_max_size = db_df.groupby(exact_match_col_indices).size().max()
        if exm_max_size < topK:
            logging.info(f"[WARNING] the max number ({exm_max_size}) of exact matching items is smaller than topK ({topK})")
            # topK = exm_max_size
        exm_cols_mask = np.zeros(db_np_data.shape[-1], dtype=bool)
        exm_cols_mask[exact_match_col_indices] = True
        rest_cols_mask = ~exm_cols_mask
        db_exm_cols_np_data = db_np_data[:, exm_cols_mask]
        qry_exm_cols_np_data = qry_np_data[:, exm_cols_mask]
        db_np_data = db_np_data[:, rest_cols_mask]
        qry_np_data = qry_np_data[:, rest_cols_mask]
        if db_np_data.shape[-1] > 0:
            exm_chunk_size_scaling_factor = db_np_data.shape[-1] / len(exact_match_col_indices)
        else:
            exm_chunk_size_scaling_factor = 1

    # pre-compute the IDF statists for each column in the database
    N = len(db_np_data)
    db_df = pd.DataFrame(db_np_data)
    IDF_stats = []
    for col in db_df:
        col_IDF_stats = db_df[col].value_counts()
        # col_IDF_stats = np.log((N - col_IDF_stats + 0.5) / (col_IDF_stats + 0.5))
        col_IDF_stats = np.log(N / col_IDF_stats)
        IDF_stats.append(col_IDF_stats)

    # if not process by chunks, then load the whole db_data to the device
    if db_chunk_size is None:
        if exact_match_col_indices:
            db_exm_cols_data = torch.from_numpy(db_exm_cols_np_data).to(device)
        db_data = torch.from_numpy(db_np_data).to(device)
    # if not specific qry_batch_size, then process the whole qry_data at one time
    qry_batch_size = len(qry_np_data) if qry_batch_size is None else qry_batch_size
    topK_indices = []
    topK_values = []
    topK_indices_len = []
    for qry_idx in tqdm(range(0, len(qry_np_data), qry_batch_size), desc="retrieve samples"):
        if exact_match_col_indices:
            qry_exm_cols_data_batch = qry_exm_cols_np_data[qry_idx: qry_idx + qry_batch_size]
            qry_exm_cols_data_batch = torch.from_numpy(qry_exm_cols_data_batch).to(device)
        qry_data_batch = qry_np_data[qry_idx: qry_idx + qry_batch_size]
        # map the IDF statists to each position in qry_data_batch
        qry_IDF_data_batch = map_data_to_IDF(qry_data_batch, IDF_stats)
        qry_data_batch = torch.from_numpy(qry_data_batch).to(device)
        qry_IDF_data_batch = torch.from_numpy(qry_IDF_data_batch).to(device)
        if db_chunk_size is None:
            if exact_match_col_indices:
                # B: qry_batch_size, N: database_size, F: field_num(exact matching), E: exact_matching_max_len
                # (Bx1xF compare 1xNxF) * Bx1xF => BxNxF => BxN
                exm_values_batch = (qry_exm_cols_data_batch.unsqueeze(1) == db_exm_cols_data.unsqueeze(0)).all(-1).float()
                exm_padding_size = topK if qry_np_data.shape[-1] == 0 else max(exm_max_size, topK)
                exm_results_batch = padded_topk(exm_values_batch, exm_padding_size) # 2x(BxE) 
                exm_values_batch = exm_results_batch.values  # BxE, fliter the exact matching values and indices
                exm_indices_batch = exm_results_batch.indices  # BxE
            if exact_match_col_indices and exm_max_size < topK:  # no need to further filter by BM25
                topK_results_batch = sort_results(exm_values_batch, exm_indices_batch)
                topK_values_batch = topK_results_batch.values.cpu().numpy()
                topK_indices_batch = topK_results_batch.indices.cpu().numpy()
                topK_indices_len_batch = topK_results_batch.lens.cpu().numpy()
            elif qry_np_data.shape[-1] > 0:
                if exact_match_col_indices:
                    db_data_batch = db_data[exm_indices_batch] # NxF.lookup(BxE) => BxExF
                    # (Bx1xF compare BxExF) * Bx1xF => BxExF => BxE, E can be N
                    BM25_values_batch = ((qry_data_batch.unsqueeze(1) == db_data_batch) * qry_IDF_data_batch.unsqueeze(1)).sum(-1)
                    BM25_values_batch = (BM25_values_batch + 1) * exm_values_batch # BxE, BxE => BxE
                else:
                    BM25_values_batch = ((qry_data_batch.unsqueeze(1) == db_data.unsqueeze(0)) * qry_IDF_data_batch.unsqueeze(1)).sum(-1)
                topK_results_batch = padded_topk(BM25_values_batch, topK)
                topK_values_batch = topK_results_batch.values
                topK_indices_batch = topK_results_batch.indices
                # gather the global indices according to the indices in the filtered list 'exm_indices_batch' (shape: BxExF)
                if exact_match_col_indices:
                    topK_indices_batch = masked_gather(exm_indices_batch, index=topK_indices_batch)
                topK_results_batch = sort_results(topK_values_batch, topK_indices_batch)
                topK_values_batch = topK_results_batch.values.cpu().numpy()
                topK_indices_batch = topK_results_batch.indices.cpu().numpy()
                topK_indices_len_batch = topK_results_batch.lens.cpu().numpy()
            else:  # exact matching only
                assert exact_match_col_indices is not None, "detected empty query tensor input"
                # already been truncated to [:, :topK] via 'padding_size = topK if qry_np_data.shape[-1] == 0 else max(exm_max_size, topK)'
                topK_indices_batch = exm_indices_batch.cpu().numpy()
                topK_indices_len_batch = (topK_indices_batch != -1).sum(-1)
                topK_values_batch = exm_values_batch.cpu().numpy()
        else:
            if exact_match_col_indices:
                ## fliter the data by exact matching
                local_filtered_exm_values_batch = []
                local_filtered_exm_indices_batch = []
                exm_db_chunk_size = int(db_chunk_size * exm_chunk_size_scaling_factor)
                exm_padding_size = topK if qry_np_data.shape[-1] == 0 else max(exm_max_size, topK)
                for db_idx in range(0, len(db_exm_cols_np_data), exm_db_chunk_size):
                    # B: qry_batch_size, C: database_chunk_size, F: field_num
                    local_db_exm_cols_data = torch.from_numpy(db_exm_cols_np_data[db_idx: db_idx + exm_db_chunk_size]).to(device) # CxF
                    # (Bx1xF compare 1xCxF) * Bx1xF => BxCxF => BxC
                    local_exm_values_batch = (qry_exm_cols_data_batch.unsqueeze(1) == local_db_exm_cols_data.unsqueeze(0)).all(-1).float()
                    local_results = padded_topk(local_exm_values_batch, exm_padding_size, db_idx)
                    local_filtered_exm_values_batch.append(local_results.values) # BxK
                    local_filtered_exm_indices_batch.append(local_results.indices) # BxK
                # X: database_chunk_num, BxXK
                local_filtered_exm_values_batch = torch.cat(local_filtered_exm_values_batch, dim=-1) 
                local_filtered_exm_indices_batch = torch.cat(local_filtered_exm_indices_batch, dim=-1)
                # aggregate the chunk-wise exm results, rank and finally get the global exm results
                exm_results_batch = padded_topk(local_filtered_exm_values_batch, exm_padding_size)
                exm_values_batch = exm_results_batch.values # BxE
                exm_indices_batch = masked_gather(local_filtered_exm_indices_batch, index=exm_results_batch.indices)  # BxE
            if exact_match_col_indices and exm_max_size < topK:  # no need to further filter by BM25
                topK_results_batch = sort_results(exm_values_batch, exm_indices_batch)
                topK_values_batch = topK_results_batch.values.cpu().numpy()
                topK_indices_batch = topK_results_batch.indices.cpu().numpy()
                topK_indices_len_batch = topK_results_batch.lens.cpu().numpy()
                if enable_clean:
                    del local_filtered_exm_values_batch, local_filtered_exm_indices_batch
                    gc.collect()
            elif qry_np_data.shape[-1] > 0:
                local_topK_values_batch = []
                local_topK_indices_batch = []
                if exact_match_col_indices:
                    db_np_data_batch = db_np_data[exm_indices_batch.cpu().numpy()] # NxF.lookup(BxE) => BxExF
                    for db_idx in range(0, db_np_data_batch.shape[1], db_chunk_size):
                        local_db_data = torch.from_numpy(db_np_data_batch[:, db_idx: db_idx + db_chunk_size]).to(device)
                        local_exm_values_batch = exm_values_batch[:, db_idx: db_idx + db_chunk_size] # BxC
                        # B: qry_batch_size, C: database_chunk_size, F: field_num
                        # (Bx1xF compare BxCxF) * Bx1xF => BxCxF => BxC
                        local_BM25_values_batch = ((qry_data_batch.unsqueeze(1) == local_db_data) * qry_IDF_data_batch.unsqueeze(1)).sum(-1)
                        local_BM25_values_batch = (local_BM25_values_batch + 1) * local_exm_values_batch
                        local_results = padded_topk(local_BM25_values_batch, topK, db_idx)
                        local_topK_values_batch.append(local_results.values)  # BxK
                        local_topK_indices_batch.append(local_results.indices) # BxK
                else:
                    for db_idx in range(0, len(db_np_data), db_chunk_size):
                        local_db_data = torch.from_numpy(db_np_data[db_idx: db_idx + db_chunk_size]).to(device)
                        # B: qry_batch_size, C: database_chunk_size, F: field_num
                        # (Bx1xF compare 1xNxF) * Bx1xF => BxCxF => BxC
                        local_BM25_values_batch = ((qry_data_batch.unsqueeze(1) == local_db_data.unsqueeze(0)) * qry_IDF_data_batch.unsqueeze(1)).sum(-1)
                        local_results = padded_topk(local_BM25_values_batch, topK, db_idx)
                        local_topK_values_batch.append(local_results.values)  # BxK
                        local_topK_indices_batch.append(local_results.indices) # BxK
                # X: database_chunk_num, BxXK
                local_topK_values_batch = torch.cat(local_topK_values_batch, dim=-1) 
                local_topK_indices_batch = torch.cat(local_topK_indices_batch, dim=-1)
                # aggregate the chunk-wise topK results, rank and finally get the global topK results
                topK_results_batch = padded_topk(local_topK_values_batch, topK)
                topK_values_batch = topK_results_batch.values
                topK_indices_batch = masked_gather(local_topK_indices_batch, index=topK_results_batch.indices)
                # gather the global indices according to the indices in the filtered list 'exm_indices_batch' (shape: BxExF)
                if exact_match_col_indices:
                    topK_indices_batch = masked_gather(exm_indices_batch, index=topK_indices_batch)
                topK_results_batch = sort_results(topK_values_batch, topK_indices_batch)
                topK_values_batch = topK_results_batch.values.cpu().numpy()
                topK_indices_batch = topK_results_batch.indices.cpu().numpy()
                topK_indices_len_batch = topK_results_batch.lens.cpu().numpy()
                if enable_clean:
                    del local_topK_values_batch, local_topK_indices_batch
                    gc.collect()
            else:  # exact matching only
                assert exact_match_col_indices is not None, "detected empty query tensor input"
                # already been truncated to [:, :topK] via 'padding_size = topK if qry_np_data.shape[-1] == 0 else max(exm_max_size, topK)'
                topK_indices_batch = exm_indices_batch.cpu().numpy()
                topK_indices_len_batch = (topK_indices_batch != -1).sum(-1)
                topK_values_batch = exm_values_batch.cpu().numpy()
        topK_values.append(topK_values_batch)
        topK_indices.append(topK_indices_batch)
        topK_indices_len.append(topK_indices_len_batch)
    topK_values = np.concatenate(topK_values)
    topK_indices = np.concatenate(topK_indices)
    topK_indices_len = np.concatenate(topK_indices_len)
    if enable_clean:
        del IDF_stats
        gc.collect()
    return ResultsNamedTuple(topK_values, topK_indices, topK_indices_len)


# 添加精确匹配（用pandas MultiIndex实现精确匹配过滤）
@autocast(enabled=ENABLE_AUTOCAST)
def BM25_topk_retrieval_v3(
    db_np_data: np.ndarray,
    qry_np_data: np.ndarray,
    exact_match_col_indices: list = None,
    qry_batch_size: int = None,
    db_chunk_size: int = None,
    device: str = 'cpu',
    topK: int = 10,
    enable_clean: bool = False,
    **kwargs
    ):
    ResultsNamedTuple = namedtuple("ResultsNameTuple", ["values", "indices", "lens"])

    def sort_results(values, indices):
        # values: BxK, indices: BxK
        drop_mask = (values == 0) # BxK
        indices[drop_mask] = -1
        results = torch.sort(values, descending=True)
        values = results.values
        indices = torch.gather(indices, -1, results.indices)
        lens = drop_mask.shape[-1] - drop_mask.sum(-1) # B
        if enable_clean:
            del drop_mask
            gc.collect()
        return ResultsNamedTuple(values, indices, lens)

    def padded_topk(input_values, K, index_offs=None):
        assert input_values.ndim == 2, "input shape must be [BxN]"
        output_lens = torch.zeros_like(input_values[:, 0], dtype=int)
        if K >= input_values.shape[-1]:
            output_values = F.pad(input_values, (0, K - input_values.shape[-1]))
            output_indices = torch.zeros_like(output_values, dtype=torch.long)
            for col_i in range(input_values.shape[-1]):
                output_indices[:, col_i] = col_i
            if index_offs:
                output_indices += index_offs
            output_indices[:, input_values.shape[-1]:] = -1
            output_lens[:] = input_values.shape[-1]
        else:
            output_results = torch.topk(input_values, k=K)
            output_values = output_results.values
            output_indices = output_results.indices
            if index_offs:
                output_indices += index_offs
            output_lens[:] = K
        return ResultsNamedTuple(output_values, output_indices, output_lens)

    def masked_gather(input, index, mask_index_value=-1):
        if mask_index_value not in index:
            return torch.gather(input, -1, index)
        else:
            mask = (index == mask_index_value)
            index[mask] = 0
            results = torch.gather(input, -1, index)
            results[mask] = mask_index_value
            if enable_clean:
                del mask
                gc.collect()
            return results

    def map_data_to_IDF_v1(np_data, IDF_stats):
        IDF_np_data = np.zeros_like(np_data, dtype=float)
        for i, col_IDF_stats in enumerate(IDF_stats):
            IDF_np_data[:, i] = np.vectorize(lambda x: col_IDF_stats.get(x, 0))(np_data[:, i])
        return IDF_np_data
    
    def map_data_to_IDF_v2(np_data, IDF_stats):
        # NOT safe: need to assure that the query and the db have the same value set, or will cause IndexError
        IDF_np_data = []
        for i, col_IDF_stats in enumerate(IDF_stats):
            IDF_np_data.append(col_IDF_stats.values[map_indices(col_IDF_stats.index.to_numpy(), np_data[:, i], missing=-1)])
        return np.stack(IDF_np_data, axis=-1)
    
    map_data_to_IDF = map_data_to_IDF_v1

    if exact_match_col_indices:
        db_df = pd.DataFrame(db_np_data)
        db_groups = pd.Series(db_df.groupby(exact_match_col_indices).groups)
        exm_cols_mask = np.zeros(db_np_data.shape[-1], dtype=bool)
        exm_cols_mask[exact_match_col_indices] = True
        rest_cols_mask = ~exm_cols_mask
        qry_exm_cols_df = pd.DataFrame(qry_np_data[:, exm_cols_mask])
        qry_exm_cols_df = qry_exm_cols_df.set_index(list(qry_exm_cols_df.columns))
        qry_exm_grp_ids = db_groups.index.get_indexer(qry_exm_cols_df.index)
        # 用 qry_exm_grp_ids 查询 db_groups 就可以得到对应的id序列了
        db_np_data = db_np_data[:, rest_cols_mask]
        qry_np_data = qry_np_data[:, rest_cols_mask]
        if enable_clean:
            del exm_cols_mask, rest_cols_mask, qry_exm_cols_df
            gc.collect()

    # pre-compute the IDF statists for each column in the database
    N = len(db_np_data)
    db_df = pd.DataFrame(db_np_data)
    IDF_stats = []
    for col in db_df:
        col_IDF_stats = db_df[col].value_counts()
        # col_IDF_stats = np.log((N - col_IDF_stats + 0.5) / (col_IDF_stats + 0.5))
        col_IDF_stats = np.log(N / col_IDF_stats)
        IDF_stats.append(col_IDF_stats)

    # if not process by chunks, then load the whole db_data to the device
    if db_chunk_size is None:
        db_data = torch.from_numpy(db_np_data).to(device)
    # if not specific qry_batch_size, then process the whole qry_data at one time
    qry_batch_size = len(qry_np_data) if qry_batch_size is None else qry_batch_size
    topK_values = np.zeros((len(qry_np_data), topK), dtype=float)
    topK_indices = np.full((len(qry_np_data), topK), -1, dtype=int)
    topK_indices_len = np.zeros(len(qry_np_data), dtype=int)
    for qry_idx in tqdm(range(0, len(qry_np_data), qry_batch_size), desc="retrieve samples"):
        if exact_match_col_indices:
            qry_exm_grp_ids_batch = qry_exm_grp_ids[qry_idx: qry_idx + qry_batch_size]
            valid_qry_exm_grp_ids_batch = qry_exm_grp_ids_batch[qry_exm_grp_ids_batch != -1]
            if len(valid_qry_exm_grp_ids_batch) == 0:
                continue
            exm_indices_batch = db_groups[db_groups.index[valid_qry_exm_grp_ids_batch]]
            exm_indices_batch = pad_sequences(exm_indices_batch, padding='post', 
                                              maxlen=topK if qry_np_data.shape[-1] == 0 else None,
                                              value=-1, dtype="int64") # valid_q x topK (or max_batch_len)
            exm_max_size_batch = exm_indices_batch.shape[-1]
            if enable_clean:
                del valid_qry_exm_grp_ids_batch
                gc.collect()

        if exact_match_col_indices and exm_max_size_batch <= topK:
            # padding to topK length
            topK_indices_len_batch = (exm_indices_batch != -1).sum(-1)
            topK_indices_batch = np.pad(exm_indices_batch,
                                        ((0, 0), (0, topK - exm_max_size_batch)),
                                        constant_values=-1)  # BxK
            topK_values_batch = (topK_indices_batch != -1).astype(float)
        elif qry_np_data.shape[-1] > 0:  # need to further compute BM25 values
            qry_data_batch = qry_np_data[qry_idx: qry_idx + qry_batch_size]
            if exact_match_col_indices: 
                # filter out those query samples without any matched retrieval sample
                qry_data_batch = qry_data_batch[qry_exm_grp_ids_batch != -1]
            # map the IDF statists to each position in qry_data_batch
            qry_IDF_data_batch = map_data_to_IDF(qry_data_batch, IDF_stats)
            qry_data_batch = torch.from_numpy(qry_data_batch).to(device)
            qry_IDF_data_batch = torch.from_numpy(qry_IDF_data_batch).to(device)
            
            if db_chunk_size is None:
                if exact_match_col_indices:
                    exm_indices_batch = torch.from_numpy(exm_indices_batch).to(device).long()
                    exm_values_batch = (exm_indices_batch != -1).float()
                    db_data_batch = db_data[exm_indices_batch] # NxF.lookup(BxE) => BxExF
                    # (Bx1xF compare BxExF) * Bx1xF => BxExF => BxE, E can be N
                    BM25_values_batch = ((qry_data_batch.unsqueeze(1) == db_data_batch) * qry_IDF_data_batch.unsqueeze(1)).sum(-1)
                    BM25_values_batch = (BM25_values_batch + 1) * exm_values_batch # BxE, BxE => BxE
                else:
                    BM25_values_batch = ((qry_data_batch.unsqueeze(1) == db_data.unsqueeze(0)) * qry_IDF_data_batch.unsqueeze(1)).sum(-1)
                topK_results_batch = padded_topk(BM25_values_batch, topK)
                topK_values_batch = topK_results_batch.values
                topK_indices_batch = topK_results_batch.indices
                # gather the global indices according to the indices in the filtered list 'exm_indices_batch' (shape: BxExF)
                if exact_match_col_indices:
                    topK_indices_batch = masked_gather(exm_indices_batch, index=topK_indices_batch)
                topK_results_batch = sort_results(topK_values_batch, topK_indices_batch)
                topK_values_batch = topK_results_batch.values.cpu().numpy()
                topK_indices_batch = topK_results_batch.indices.cpu().numpy()
                topK_indices_len_batch = topK_results_batch.lens.cpu().numpy()
            else:
                local_topK_values_batch = []
                local_topK_indices_batch = []
                if exact_match_col_indices:
                    db_np_data_batch = db_np_data[exm_indices_batch] # NxF.lookup(BxE) => BxExF
                    exm_values_batch = (exm_indices_batch != -1).astype(float) # BxE
                    for db_idx in range(0, db_np_data_batch.shape[1], db_chunk_size):
                        local_db_data = torch.from_numpy(db_np_data_batch[:, db_idx: db_idx + db_chunk_size]).to(device)
                        local_exm_values_batch = torch.from_numpy(exm_values_batch[:, db_idx: db_idx + db_chunk_size]).to(device) # BxC
                        # B: qry_batch_size, C: database_chunk_size, F: field_num
                        # (Bx1xF compare BxCxF) * Bx1xF => BxCxF => BxC
                        local_BM25_values_batch = ((qry_data_batch.unsqueeze(1) == local_db_data) * qry_IDF_data_batch.unsqueeze(1)).sum(-1)
                        local_BM25_values_batch = (local_BM25_values_batch + 1) * local_exm_values_batch
                        local_results = padded_topk(local_BM25_values_batch, topK, db_idx)
                        local_topK_values_batch.append(local_results.values)  # BxK
                        local_topK_indices_batch.append(local_results.indices) # BxK
                else:
                    for db_idx in range(0, len(db_np_data), db_chunk_size):
                        local_db_data = torch.from_numpy(db_np_data[db_idx: db_idx + db_chunk_size]).to(device)
                        # B: qry_batch_size, C: database_chunk_size, F: field_num
                        # (Bx1xF compare 1xNxF) * Bx1xF => BxCxF => BxC
                        local_BM25_values_batch = ((qry_data_batch.unsqueeze(1) == local_db_data.unsqueeze(0)) * qry_IDF_data_batch.unsqueeze(1)).sum(-1)
                        local_results = padded_topk(local_BM25_values_batch, topK, db_idx)
                        local_topK_values_batch.append(local_results.values)  # BxK
                        local_topK_indices_batch.append(local_results.indices) # BxK
                # X: database_chunk_num, BxXK
                local_topK_values_batch = torch.cat(local_topK_values_batch, dim=-1) 
                local_topK_indices_batch = torch.cat(local_topK_indices_batch, dim=-1)
                # aggregate the chunk-wise topK results, rank and finally get the global topK results
                topK_results_batch = padded_topk(local_topK_values_batch, topK)
                topK_values_batch = topK_results_batch.values
                topK_indices_batch = masked_gather(local_topK_indices_batch, index=topK_results_batch.indices)
                # gather the global indices according to the indices in the filtered list 'exm_indices_batch' (shape: BxExF)
                if exact_match_col_indices:
                    exm_indices_batch = torch.from_numpy(exm_indices_batch).to(device)
                    topK_indices_batch = masked_gather(exm_indices_batch, index=topK_indices_batch)
                topK_results_batch = sort_results(topK_values_batch, topK_indices_batch)
                topK_values_batch = topK_results_batch.values.cpu().numpy()
                topK_indices_batch = topK_results_batch.indices.cpu().numpy()
                topK_indices_len_batch = topK_results_batch.lens.cpu().numpy()
                if enable_clean:
                    del local_topK_values_batch, local_topK_indices_batch
                    gc.collect()
        else: # exact matching only
            assert exact_match_col_indices is not None, "detected empty query tensor input"
            # already been truncated to [:, :topK] via 'exm_indices_batch = pad_sequences(exm_indices_batch, padding='post', maxlen=topK, value=-1, dtype="int64")'
            topK_indices_batch = exm_indices_batch
            topK_indices_len_batch = (topK_indices_batch != -1).sum(-1)
            topK_values_batch = (topK_indices_batch != -1).astype(float)
        
        # saved the results
        if exact_match_col_indices:
            topK_values[qry_idx: qry_idx + qry_batch_size][qry_exm_grp_ids_batch != -1] = topK_values_batch
            topK_indices[qry_idx: qry_idx + qry_batch_size][qry_exm_grp_ids_batch != -1] = topK_indices_batch
            topK_indices_len[qry_idx: qry_idx + qry_batch_size][qry_exm_grp_ids_batch != -1] = topK_indices_len_batch
            if enable_clean:
                del qry_exm_grp_ids_batch
                gc.collect()
        else:
            topK_values[qry_idx: qry_idx + qry_batch_size] = topK_values_batch
            topK_indices[qry_idx: qry_idx + qry_batch_size] = topK_indices_batch
            topK_indices_len[qry_idx: qry_idx + qry_batch_size] = topK_indices_len_batch
        if enable_clean:
            del topK_values_batch, topK_indices_batch, topK_indices_len_batch
            gc.collect()
    if enable_clean:
        del IDF_stats
        gc.collect()
    return ResultsNamedTuple(topK_values, topK_indices, topK_indices_len)


# 对v3的优化：把pad_sequences改成merge成一个大的共享的id序列，作为batch-wise的db数据
@autocast(enabled=ENABLE_AUTOCAST)
def BM25_topk_retrieval_v4(
    db_np_data: np.ndarray,
    qry_np_data: np.ndarray,
    exact_match_col_indices: list = None,
    qry_batch_size: int = None,
    db_chunk_size: int = None,
    device: str = 'cpu',
    topK: int = 10,
    enable_clean: bool = False,
    **kwargs
    ):
    ResultsNamedTuple = namedtuple("ResultsNameTuple", ["values", "indices", "lens"])

    def sort_results(values, indices):
        # values: BxK, indices: BxK
        drop_mask = (values == 0) # BxK
        indices[drop_mask] = -1
        results = torch.sort(values, descending=True)
        values = results.values
        indices = torch.gather(indices, -1, results.indices)
        lens = drop_mask.shape[-1] - drop_mask.sum(-1) # B
        if enable_clean:
            del drop_mask
            gc.collect()
        return ResultsNamedTuple(values, indices, lens)

    def padded_topk(input_values, K, index_offs=None):
        assert input_values.ndim == 2, "input shape must be [BxN]"
        output_lens = torch.zeros_like(input_values[:, 0], dtype=int)
        if K >= input_values.shape[-1]:
            output_values = F.pad(input_values, (0, K - input_values.shape[-1]))
            output_indices = torch.zeros_like(output_values, dtype=torch.long)
            for col_i in range(input_values.shape[-1]):
                output_indices[:, col_i] = col_i
            if index_offs:
                output_indices += index_offs
            output_indices[:, input_values.shape[-1]:] = -1
            output_lens[:] = input_values.shape[-1]
        else:
            output_results = torch.topk(input_values, k=K)
            output_values = output_results.values
            output_indices = output_results.indices
            if index_offs:
                output_indices += index_offs
            output_lens[:] = K
        return ResultsNamedTuple(output_values, output_indices, output_lens)

    def masked_gather(input, index, mask_index_value=-1):
        if mask_index_value not in index:
            return torch.gather(input, -1, index)
        else:
            mask = (index == mask_index_value)
            index[mask] = 0
            results = torch.gather(input, -1, index)
            results[mask] = mask_index_value
            if enable_clean:
                del mask
                gc.collect()
            return results

    def masked_indexing(input, index, mask_index_value=-1):
        if mask_index_value not in index:
            return input[index]
        else:
            mask = (index == mask_index_value)
            index[mask] = 0
            results = input[index]
            results[mask] = mask_index_value
            if enable_clean:
                del mask
                gc.collect()
            return results

    def map_data_to_IDF_v1(np_data, IDF_stats):
        IDF_np_data = np.zeros_like(np_data, dtype=float)
        for i, col_IDF_stats in enumerate(IDF_stats):
            IDF_np_data[:, i] = np.vectorize(lambda x: col_IDF_stats.get(x, 0))(np_data[:, i])
        return IDF_np_data

    def map_data_to_IDF_v2(np_data, IDF_stats):
        # NOT safe: need to assure that the query and the db have the same value set, or will cause IndexError
        IDF_np_data = []
        for i, col_IDF_stats in enumerate(IDF_stats):
            IDF_np_data.append(col_IDF_stats.values[map_indices(col_IDF_stats.index.to_numpy(), np_data[:, i], missing=-1)])
        return np.stack(IDF_np_data, axis=-1)

    map_data_to_IDF = map_data_to_IDF_v1

    if exact_match_col_indices:
        db_df = pd.DataFrame(db_np_data)
        db_groups = pd.Series(db_df.groupby(exact_match_col_indices).groups)
        exm_cols_mask = np.zeros(db_np_data.shape[-1], dtype=bool)
        exm_cols_mask[exact_match_col_indices] = True
        rest_cols_mask = ~exm_cols_mask
        qry_exm_cols_df = pd.DataFrame(qry_np_data[:, exm_cols_mask])
        qry_exm_cols_df = qry_exm_cols_df.set_index(list(qry_exm_cols_df.columns))
        qry_exm_grp_ids = db_groups.index.get_indexer(qry_exm_cols_df.index)
        # use qry_exm_grp_ids to query db_groups and get idx sequence
        db_np_data = db_np_data[:, rest_cols_mask]
        qry_np_data = qry_np_data[:, rest_cols_mask]
        if enable_clean:
            del exm_cols_mask, rest_cols_mask, qry_exm_cols_df
            gc.collect()

    # pre-compute the IDF statists for each column in the database
    N = len(db_np_data)
    db_df = pd.DataFrame(db_np_data)
    IDF_stats = []
    for col in db_df:
        col_IDF_stats = db_df[col].value_counts()
        # col_IDF_stats = np.log((N - col_IDF_stats + 0.5) / (col_IDF_stats + 0.5))
        col_IDF_stats = np.log(N / col_IDF_stats)
        IDF_stats.append(col_IDF_stats)

    # if not process by chunks, then load the whole db_data to the device
    if db_chunk_size is None:
        db_data = torch.from_numpy(db_np_data).to(device)
    # if not specific qry_batch_size, then process the whole qry_data at one time
    qry_batch_size = len(qry_np_data) if qry_batch_size is None else qry_batch_size
    topK_values = np.zeros((len(qry_np_data), topK), dtype=float)
    topK_indices = np.full((len(qry_np_data), topK), -1, dtype=int)
    topK_indices_len = np.zeros(len(qry_np_data), dtype=int)
    for qry_idx in tqdm(range(0, len(qry_np_data), qry_batch_size), desc="retrieve samples"):
        if exact_match_col_indices:
            qry_exm_grp_ids_batch = qry_exm_grp_ids[qry_idx: qry_idx + qry_batch_size]
            valid_qry_exm_grp_ids_batch = qry_exm_grp_ids_batch[qry_exm_grp_ids_batch != -1]
            if len(valid_qry_exm_grp_ids_batch) == 0:
                continue
            exm_indices_batch = db_groups[db_groups.index[valid_qry_exm_grp_ids_batch]]
            exm_indices_batch = pad_sequences(exm_indices_batch, padding='post', 
                                              maxlen=topK if qry_np_data.shape[-1] == 0 else None,
                                              value=-1, dtype="int64") # valid_q x topK (or max_batch_len)
            exm_max_size_batch = exm_indices_batch.shape[-1]
            if enable_clean:
                del valid_qry_exm_grp_ids_batch
                gc.collect()

        if exact_match_col_indices and exm_max_size_batch <= topK:
            # padding to topK length
            topK_indices_len_batch = (exm_indices_batch != -1).sum(-1)
            topK_indices_batch = np.pad(exm_indices_batch,
                                        ((0, 0), (0, topK - exm_max_size_batch)),
                                        constant_values=-1)  # BxK
            topK_values_batch = (topK_indices_batch != -1).astype(float)
            if enable_clean:
                del exm_indices_batch
                gc.collect()
        elif qry_np_data.shape[-1] > 0:  # need to further compute BM25 values
            qry_data_batch = qry_np_data[qry_idx: qry_idx + qry_batch_size]
            if exact_match_col_indices: 
                # filter out those query samples without any matched retrieval sample
                qry_data_batch = qry_data_batch[qry_exm_grp_ids_batch != -1]
            # map the IDF statists to each position in qry_data_batch
            qry_IDF_data_batch = map_data_to_IDF(qry_data_batch, IDF_stats)
            qry_data_batch = torch.from_numpy(qry_data_batch).to(device)
            qry_IDF_data_batch = torch.from_numpy(qry_IDF_data_batch).to(device)

            # aggregate the exm ids of samples in current batch as the batch-wise db
            if exact_match_col_indices:
                exm_indices_batch = torch.from_numpy(exm_indices_batch).long().to(device)
                all_exm_indices_batch = torch.unique(exm_indices_batch) # BxE => N' in [0, N)
                if all_exm_indices_batch[0] == -1:
                    all_exm_indices_batch = all_exm_indices_batch[1:] # skip the -1
                mapped_exm_indices_batch = map_indices(all_exm_indices_batch, exm_indices_batch, missing=-1, is_key_sorted=True)

            if db_chunk_size is None:
                if exact_match_col_indices:
                    exm_values_batch = (mapped_exm_indices_batch != -1).float() # BxE
                    db_data_batch = db_data[all_exm_indices_batch] # NxF.lookup(N') => N'xF
                    # (Bx1xF compare 1xN'xF) * Bx1xF => BxN'xF => BxN'
                    BM25_values_batch = ((qry_data_batch.unsqueeze(1) == db_data_batch.unsqueeze(0)) * qry_IDF_data_batch.unsqueeze(1)).sum(-1)
                    BM25_values_batch = masked_gather(BM25_values_batch, mapped_exm_indices_batch) # BxN', BxE in [0, N') => BxE
                    BM25_values_batch = (BM25_values_batch + 1) * exm_values_batch # BxE, BxE => BxE
                    if enable_clean:
                        del exm_values_batch, mapped_exm_indices_batch, db_data_batch
                        gc.collect()
                else:
                    BM25_values_batch = ((qry_data_batch.unsqueeze(1) == db_data.unsqueeze(0)) * qry_IDF_data_batch.unsqueeze(1)).sum(-1)
                if enable_clean:
                    del qry_data_batch, qry_IDF_data_batch
                    gc.collect()
                topK_results_batch = padded_topk(BM25_values_batch, topK)
                topK_values_batch = topK_results_batch.values
                topK_indices_batch = topK_results_batch.indices
                if enable_clean:
                    del BM25_values_batch
                    gc.collect()
                # gather the global indices according to the indices in the filtered list 'exm_indices_batch' (shape: BxExF)
                if exact_match_col_indices:
                    topK_indices_batch = masked_gather(exm_indices_batch, index=topK_indices_batch) # BxE in [0, N), BxK in [0, E) => BxK in [0, N)
                    if enable_clean:
                        del exm_indices_batch
                        gc.collect()
                topK_results_batch = sort_results(topK_values_batch, topK_indices_batch)
                topK_values_batch = topK_results_batch.values.cpu().numpy()
                topK_indices_batch = topK_results_batch.indices.cpu().numpy()
                topK_indices_len_batch = topK_results_batch.lens.cpu().numpy()
            else:
                local_topK_values_batch = []
                local_topK_indices_batch = []
                if exact_match_col_indices:
                    db_np_data_batch = db_np_data[all_exm_indices_batch.cpu().numpy()] # NxF.lookup(N') => N'xF
                    mapped_exm_indices_batch = mapped_exm_indices_batch.cpu().numpy()  # BxE
                    exm_values_batch = np.zeros((len(mapped_exm_indices_batch), 
                                                 len(all_exm_indices_batch) + 1), 
                                                dtype=float) # Bx(N'+1)
                    exm_values_invalid_mask_batch = (mapped_exm_indices_batch == -1) # BxE
                    mapped_exm_indices_batch[exm_values_invalid_mask_batch] = len(all_exm_indices_batch)  # BxE in [0, N')
                    np.put_along_axis(arr=exm_values_batch,  # Bx(N'+1)
                                      indices=mapped_exm_indices_batch,  # BxE in [0, N')
                                      values=(~exm_values_invalid_mask_batch),  # 1 for valid
                                      axis=-1)
                    exm_values_batch = exm_values_batch[:, :len(all_exm_indices_batch)]  # BxN'
                    # mapped_exm_indices_batch[exm_values_invalid_mask_batch] = -1 # BxE in [0, N'), not necessary to update as it will not be used
                    if enable_clean:
                        del mapped_exm_indices_batch, exm_values_invalid_mask_batch
                        gc.collect()
                    for db_idx in range(0, len(db_np_data_batch), db_chunk_size):
                        local_db_data = torch.from_numpy(db_np_data_batch[db_idx: db_idx + db_chunk_size]).to(device) # N'xF => CxF
                        local_exm_values_batch = torch.from_numpy(exm_values_batch[:, db_idx: db_idx + db_chunk_size]).to(device) # BxN' => BxC
                        # B: qry_batch_size, C: database_chunk_size, F: field_num
                        # (Bx1xF compare 1xCxF) * Bx1xF => BxCxF => BxC
                        local_BM25_values_batch = ((qry_data_batch.unsqueeze(1) == local_db_data.unsqueeze(0)) * qry_IDF_data_batch.unsqueeze(1)).sum(-1)
                        local_BM25_values_batch = (local_BM25_values_batch + 1) * local_exm_values_batch  # BxC
                        if enable_clean:
                            del local_db_data, local_exm_values_batch
                            gc.collect()
                        local_results = padded_topk(local_BM25_values_batch, topK, db_idx)  # BxK
                        local_topK_values_batch.append(local_results.values)  # BxK
                        local_topK_indices_batch.append(local_results.indices) # BxK
                        if enable_clean:
                            del local_BM25_values_batch
                            gc.collect()
                else:
                    for db_idx in range(0, len(db_np_data), db_chunk_size):
                        local_db_data = torch.from_numpy(db_np_data[db_idx: db_idx + db_chunk_size]).to(device)
                        # B: qry_batch_size, C: database_chunk_size, F: field_num
                        # (Bx1xF compare 1xNxF) * Bx1xF => BxCxF => BxC
                        local_BM25_values_batch = ((qry_data_batch.unsqueeze(1) == local_db_data.unsqueeze(0)) * qry_IDF_data_batch.unsqueeze(1)).sum(-1)
                        local_results = padded_topk(local_BM25_values_batch, topK, db_idx)
                        local_topK_values_batch.append(local_results.values)  # BxK
                        local_topK_indices_batch.append(local_results.indices) # BxK
                        if enable_clean:
                            del local_db_data, local_BM25_values_batch
                            gc.collect()
                # X: database_chunk_num, BxXK
                local_topK_values_batch = torch.cat(local_topK_values_batch, dim=-1) 
                local_topK_indices_batch = torch.cat(local_topK_indices_batch, dim=-1)
                # aggregate the chunk-wise topK results, rank and finally get the global topK results
                topK_results_batch = padded_topk(local_topK_values_batch, topK)
                topK_values_batch = topK_results_batch.values
                topK_indices_batch = masked_gather(local_topK_indices_batch, index=topK_results_batch.indices) # BxK in [0, N')
                if enable_clean:
                    del local_topK_values_batch, local_topK_indices_batch
                    gc.collect()
                # gather the global indices according to the indices in the filtered list 'exm_indices_batch' (shape: BxExF)
                if exact_match_col_indices:
                    topK_indices_batch = masked_indexing(all_exm_indices_batch, topK_indices_batch)  # BxK in [0, N') => BxK in [0, N)
                    if enable_clean:
                        del all_exm_indices_batch
                        gc.collect()
                topK_results_batch = sort_results(topK_values_batch, topK_indices_batch)
                topK_values_batch = topK_results_batch.values.cpu().numpy()
                topK_indices_batch = topK_results_batch.indices.cpu().numpy()
                topK_indices_len_batch = topK_results_batch.lens.cpu().numpy()
        else: # exact matching only
            assert exact_match_col_indices is not None, "detected empty query tensor input"
            # already been truncated to [:, :topK] via 'exm_indices_batch = pad_sequences(exm_indices_batch, padding='post', maxlen=topK, value=-1, dtype="int64")'
            topK_indices_batch = exm_indices_batch
            topK_indices_len_batch = (topK_indices_batch != -1).sum(-1)
            topK_values_batch = (topK_indices_batch != -1).astype(float)

        # saved the results
        if exact_match_col_indices:
            topK_values[qry_idx: qry_idx + qry_batch_size][qry_exm_grp_ids_batch != -1] = topK_values_batch
            topK_indices[qry_idx: qry_idx + qry_batch_size][qry_exm_grp_ids_batch != -1] = topK_indices_batch
            topK_indices_len[qry_idx: qry_idx + qry_batch_size][qry_exm_grp_ids_batch != -1] = topK_indices_len_batch
            if enable_clean:
                del qry_exm_grp_ids_batch
                gc.collect()
        else:
            topK_values[qry_idx: qry_idx + qry_batch_size] = topK_values_batch
            topK_indices[qry_idx: qry_idx + qry_batch_size] = topK_indices_batch
            topK_indices_len[qry_idx: qry_idx + qry_batch_size] = topK_indices_len_batch
        if enable_clean:
            del topK_values_batch, topK_indices_batch, topK_indices_len_batch
            gc.collect()
    if enable_clean:
        del IDF_stats
        gc.collect()
    return ResultsNamedTuple(topK_values, topK_indices, topK_indices_len)


def split_train_test(train_ddf=None, valid_ddf=None, test_ddf=None, valid_size=0, 
                     test_size=0, split_type="sequential"):
    num_samples = len(train_ddf)
    train_size = num_samples
    instance_IDs = np.arange(num_samples)
    if split_type == "random":
        np.random.shuffle(instance_IDs)
    if test_size > 0:
        if test_size < 1:
            test_size = int(num_samples * test_size)
        train_size = train_size - test_size
        test_ddf = train_ddf.loc[instance_IDs[train_size:], :].reset_index()
        instance_IDs = instance_IDs[0:train_size]
    if valid_size > 0:
        if valid_size < 1:
            valid_size = int(num_samples * valid_size)
        train_size = train_size - valid_size
        valid_ddf = train_ddf.loc[instance_IDs[train_size:], :].reset_index()
        instance_IDs = instance_IDs[0:train_size]
    if valid_size > 0 or test_size > 0:
        train_ddf = train_ddf.loc[instance_IDs, :].reset_index()
    return train_ddf, valid_ddf, test_ddf


def build_dataset(feature_encoder, train_data=None, valid_data=None, test_data=None, valid_size=0, 
                  test_size=0, split_type="sequential", retrieval_configs=None, **kwargs):
    """ Build feature_map and transform h5 data """
    # Load csv data
    train_ddf = feature_encoder.read_csv(train_data)
    valid_ddf = feature_encoder.read_csv(valid_data) if valid_data else None
    test_ddf = feature_encoder.read_csv(test_data) if test_data else None
    
    # fill na and filter out inactive columns
    train_ddf = feature_encoder.preprocess(train_ddf)
    if valid_ddf is not None:
        valid_ddf = feature_encoder.preprocess(valid_ddf)
    if test_ddf is not None:
        test_ddf = feature_encoder.preprocess(test_ddf)
    
    # Split data for train/validation/test
    if valid_size > 0 or test_size > 0:
        train_ddf, valid_ddf, test_ddf = split_train_test(train_ddf, valid_ddf, test_ddf, 
                                                          valid_size, test_size, split_type)
    if retrieval_configs is not None: # enable retrieval for records
        if "retrieval_pool_data" in retrieval_configs:
            retrieval_pool_data = retrieval_configs["retrieval_pool_data"]
            retrieval_pool_ddf = feature_encoder.read_csv(retrieval_pool_data)
            retrieval_pool_ddf = feature_encoder.preprocess(retrieval_pool_ddf)
            # fit the train dataframe and retrieval pool dataframe
            feature_encoder.fit(pd.concat([train_ddf, retrieval_pool_ddf], copy=False), **kwargs)
        else:
            assert "pool_ratio" in retrieval_configs
            assert "split_type" in retrieval_configs
            feature_encoder.fit(train_ddf, **kwargs)
            # "sequential" or "random"
            if re.match("\d+-fold", retrieval_configs["split_type"]) is None: 
                retrieval_pool_ddf, train_ddf, _ = split_train_test(
                    train_ddf=train_ddf, 
                    valid_size=(1-retrieval_configs["pool_ratio"]),
                    split_type=retrieval_configs["split_type"])
    else:
        feature_encoder.fit(train_ddf, **kwargs)
    
    # fit and transform train_ddf
    train_array = feature_encoder.transform(train_ddf)
    block_size = int(kwargs.get("data_block_size", 0))
    if block_size > 0:
        block_id = 0
        for idx in range(0, len(train_array), block_size):
            save_hdf5(train_array[idx:(idx + block_size), :], 
                      os.path.join(feature_encoder.data_dir, 
                                   'train_part_{}.h5'.format(block_id)))
            block_id += 1
    else:
        save_hdf5(train_array, os.path.join(feature_encoder.data_dir, 'train.h5'))
    del train_array, train_ddf
    gc.collect()

    # Transfrom retrieval_pool_ddf
    if retrieval_configs is not None and re.match("\d+-fold", retrieval_configs["split_type"]) is None:
        retrieval_pool_array = feature_encoder.transform(retrieval_pool_ddf)
        if block_size > 0:
            block_id = 0
            for idx in range(0, len(retrieval_pool_array), block_size):
                save_hdf5(retrieval_pool_array[idx:(idx + block_size), :], 
                          os.path.join(feature_encoder.data_dir, 
                                       'retrieval_pool_part_{}.h5'.format(block_id)))
                block_id += 1
        else:
            save_hdf5(retrieval_pool_array, 
                      os.path.join(feature_encoder.data_dir, 'retrieval_pool.h5'))
        del retrieval_pool_array, retrieval_pool_ddf
        gc.collect()

    # Transfrom valid_ddf
    if valid_ddf is not None:
        valid_array = feature_encoder.transform(valid_ddf)
        if block_size > 0:
            block_id = 0
            for idx in range(0, len(valid_array), block_size):
                save_hdf5(valid_array[idx:(idx + block_size), :], os.path.join(feature_encoder.data_dir, 'valid_part_{}.h5'.format(block_id)))
                block_id += 1
        else:
            save_hdf5(valid_array, os.path.join(feature_encoder.data_dir, 'valid.h5'))
        del valid_array, valid_ddf
        gc.collect()

    # Transfrom test_ddf
    if test_ddf is not None:
        test_array = feature_encoder.transform(test_ddf)
        if block_size > 0:
            block_id = 0
            for idx in range(0, len(test_array), block_size):
                save_hdf5(test_array[idx:(idx + block_size), :], os.path.join(feature_encoder.data_dir, 'test_part_{}.h5'.format(block_id)))
                block_id += 1
        else:
            save_hdf5(test_array, os.path.join(feature_encoder.data_dir, 'test.h5'))
        del test_array, test_ddf
        gc.collect()
    logging.info("Transform csv data to h5 done.")


def h5_generator(feature_map, stage="both", train_data=None, valid_data=None, test_data=None,
                 batch_size=32, shuffle=True, retrieval_configs=None, retrieval_augmented=False, **kwargs):
    logging.info("Loading data...")
    from ..pytorch.data_generator import get_data_generator

    if retrieval_configs is not None:
        retrieval_col_indices = []
        for col in retrieval_configs["used_cols"]:
            retrieval_col_indices.append(feature_map.feature_specs[col]['index'])
        retrieval_configs["used_col_indices"] = retrieval_col_indices
        exact_match_col_indices = None
        if "exact_match_cols" in retrieval_configs and \
            len(retrieval_configs["exact_match_cols"]) > 0:
            exact_match_col_indices = [retrieval_configs["used_cols"].index(item)
                for item in retrieval_configs["exact_match_cols"]]
            # the indices in retrieval_configs["used_col_indices"]
        retrieval_configs["exact_match_col_indices"] = exact_match_col_indices

    train_gen = None
    valid_gen = None
    test_gen = None
    if stage in ["both", "train"]:
        train_blocks = glob.glob(train_data)
        valid_blocks = glob.glob(valid_data)
        assert len(train_blocks) > 0 and len(valid_blocks) > 0, "invalid data files or paths."
        if len(train_blocks) > 1:
            train_blocks.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if len(valid_blocks) > 1:
            valid_blocks.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if retrieval_configs is None:
            train_retrieval_pool_fname = None
            valid_retrieval_pool_fname = None
        elif re.match("\d+-fold", retrieval_configs["split_type"]) is not None:
            train_retrieval_pool_fname = "self"
            valid_retrieval_pool_fname = train_blocks[0] # only use the first block as the pool
        else:  # "sequential" or "random"
            train_retrieval_pool_fname = retrieval_configs["retrieval_pool_data"]
            valid_retrieval_pool_fname = retrieval_configs["retrieval_pool_data"]
        train_gen = get_data_generator(train_blocks, 
                                       batch_size=batch_size, 
                                       shuffle=shuffle, 
                                       feature_map=feature_map, 
                                       retrieval_configs=retrieval_configs, 
                                       retrieval_pool_fname=train_retrieval_pool_fname, 
                                       retrieval_augmented=retrieval_augmented,
                                       **kwargs)
        valid_gen = get_data_generator(valid_blocks, 
                                       batch_size=batch_size, 
                                       shuffle=False, 
                                       feature_map=feature_map, 
                                       retrieval_configs=retrieval_configs,
                                       retrieval_pool_fname=valid_retrieval_pool_fname,
                                       retrieval_augmented=retrieval_augmented, 
                                       **kwargs)
        logging.info("Train samples: total/{:d}, pos/{:.0f}, neg/{:.0f}, ratio/{:.2f}%, blocks/{:.0f}" \
                     .format(train_gen.num_samples, train_gen.num_positives, train_gen.num_negatives,
                             100. * train_gen.num_positives / train_gen.num_samples, train_gen.num_blocks))
        logging.info("Validation samples: total/{:d}, pos/{:.0f}, neg/{:.0f}, ratio/{:.2f}%, blocks/{:.0f}" \
                     .format(valid_gen.num_samples, valid_gen.num_positives, valid_gen.num_negatives,
                             100. * valid_gen.num_positives / valid_gen.num_samples, valid_gen.num_blocks))
        if stage == "train":
            logging.info("Loading train data done.")
            return train_gen, valid_gen

    if stage in ["both", "test"]:
        test_blocks = glob.glob(test_data)
        if len(test_blocks) > 0:
            if len(test_blocks) > 1:
                test_blocks.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            if retrieval_configs is None:
                test_retrieval_pool_fname = None
            elif re.match("\d+-fold", retrieval_configs["split_type"]) is not None:
                test_retrieval_pool_fname = glob.glob(train_data)[0] # only use the first block as the pool
            else:  # "sequential" or "random"
                test_retrieval_pool_fname = retrieval_configs["retrieval_pool_data"]
            test_gen = get_data_generator(test_blocks, 
                                          batch_size=batch_size, 
                                          shuffle=False, 
                                          feature_map=feature_map, 
                                          retrieval_configs=retrieval_configs,
                                          retrieval_pool_fname=test_retrieval_pool_fname,
                                          retrieval_augmented=retrieval_augmented, 
                                          **kwargs)
            logging.info("Test samples: total/{:d}, pos/{:.0f}, neg/{:.0f}, ratio/{:.2f}%, blocks/{:.0f}" \
                         .format(test_gen.num_samples, test_gen.num_positives, test_gen.num_negatives,
                                 100. * test_gen.num_positives / test_gen.num_samples, test_gen.num_blocks))
        if stage == "test":
            logging.info("Loading test data done.")
            return test_gen

    logging.info("Loading data done.")
    return train_gen, valid_gen, test_gen


def tfrecord_generator():
    raise NotImplementedError()


if __name__ == '__main__':
    import random
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    db_np_data = np.random.randint(0, 5, (2000, 5))
    qry_np_data = np.random.randint(0, 5, (100, 5))
    exm_col_indices = [0, 4]
    BM25_topk_retrieval = BM25_topk_retrieval_v4
    res = BM25_topk_retrieval(
        db_np_data,
        qry_np_data,
        exact_match_col_indices=exm_col_indices,
        qry_batch_size=50,
        db_chunk_size=50,
        device='cpu',
        topK=10)
    def printf(i, res):
        print(qry_np_data[i])
        print(db_np_data[res.indices[i]])
        print(res.values[i])
        print(res.lens[i])
    for i in range(len(qry_np_data)):
        qry = qry_np_data[i][exm_col_indices]
        db = db_np_data[res.indices[i]][:, exm_col_indices]
        # printf(i, res)
        if (qry == db).all(-1).sum(-1) != res.lens[i]:
            print("ERROR for", i)
            printf(i, res)
            break
    from IPython import embed
    embed()
