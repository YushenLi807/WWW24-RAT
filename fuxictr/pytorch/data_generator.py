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


import numpy as np
from torch.utils import data
import fuxictr.datasets.data_utils as dataset_utils
from fuxictr.datasets.data_utils import load_hdf5, save_hdf5, \
    BM25_topk_retrieval_v4 as BM25_topk_retrieval, graph_collate_fn
import h5py
import re
from itertools import chain
import logging
import torch
import dgl
import os
import gc


class Dataset(data.Dataset):
    def __init__(self, darray, 
                 feature_map=None, 
                 graph_processor=None, 
                 retr_pool_darray=None,
                 retr_indices=None, 
                 retr_values=None, 
                 retr_lens=None):
        self.darray = darray # QxF
        self.graph_processor = graph_processor
        self.retr_pool_darray = None
        self.retr_indices = None
        self.retr_values = None
        self.retr_lens = None
        self.retrieval_augmented = False
        if retr_pool_darray is not None and retr_indices is not None and \
            retr_values is not None and retr_lens is not None:
            self.retr_pool_darray = retr_pool_darray  # QxF (pool=='self') or NdbxF (pool!='self')
            self.retr_indices = retr_indices  # QxK or Qx(2K)
            self.retr_values = retr_values  # QxK or Qx(2K)
            self.retr_lens = retr_lens  # Q or Qx2
            self.retrieval_augmented = True
            assert len(self.darray) == len(self.retr_indices) == \
                len(self.retr_values) == len(self.retr_lens), \
                f"darray.len = {len(self.darray)}, retr_indices.len = {len(self.retr_indices)}, retr_values.len = {len(self.retr_values)}, retr_lens.len = {len(self.retr_lens.shape)}"
            assert self.retr_indices.shape[-1] == self.retr_values.shape[-1]
        if self.graph_processor:
            self.darray = self.graph_processor.convert_indices(
                self.darray, feature_map.feature_specs)
            if self.retrieval_augmented and id(self.darray) != id(self.retr_pool_darray): # avoid repeated conversion
                self.retr_pool_darray = self.graph_processor.convert_indices(
                    self.retr_pool_darray, feature_map.feature_specs)

    def __getitem__(self, index):
        darray_i = self.darray[index] # (F+1)
        if self.retrieval_augmented:
            retrieved_darray_i = self.retr_pool_darray[self.retr_indices[index]] # Kx(F+1) or (2K)x(F+1)
            darray_i = np.expand_dims(darray_i, 0)  # 1x(F+1)
            darray_i = np.concatenate([darray_i, retrieved_darray_i])  # 1x(F+1), (Kx(F+1) or (2K)x(F+1)) => (1+K)x(F+1) or (1+2K)x(F+1)
        X_i = darray_i[..., :-1] # F or (1+K)xF or (1+2K)xF
        y_i = darray_i[..., -1] # () or (1+K) or (1+2K)
        if self.graph_processor:
            X_i = self.graph_processor.build_instance_graph(X_i, y_i)
        if self.retrieval_augmented:
            return X_i, y_i, self.retr_values[index], self.retr_lens[index] # the last two: (K) or (2K), () or (2)
        return X_i, y_i  # F, ()
    
    def __len__(self):
        return len(self.darray)


class DataGenerator(data.DataLoader):
    def __init__(self, data_path, batch_size=32, 
                 shuffle=False, num_workers=1, 
                 feature_map=None,
                 graph_processor=None,
                 retrieval_configs=None, 
                 retrieval_pool_fname=None, 
                 retrieval_augmented=False, 
                 **kwargs):
        if type(data_path) == list:
            data_path = data_path[0]
        data_array = load_hdf5(data_path) # QxF
        self.graph_processor = graph_processor
        if self.graph_processor:
            self.graph_processor = getattr(dataset_utils, graph_processor)
        
        if retrieval_configs is not None:
            assert retrieval_configs["pre_retrieval"], "we have only implemented the pre-retrieval strategy"
            if retrieval_configs["pre_retrieval"]:
                if retrieval_pool_fname != "self":
                    logging.info(f"{retrieval_configs['split_type']} retrieval, pool file: {retrieval_pool_fname}")
                    db_array = load_hdf5(retrieval_pool_fname)  # NdbxF
                data_root, data_fname = os.path.split(data_path)
                retrieval_save_path = os.path.join(
                    data_root, 
                    f'retrieval_{retrieval_configs["topK"]}_' + data_fname)
                if os.path.exists(retrieval_save_path):
                    retrieved_indices = load_hdf5(retrieval_save_path, "indices")  # QxK or Qx(2K)
                    retrieved_values = load_hdf5(retrieval_save_path, "values")  # QxK or Qx(2K)
                    retrieved_lens = load_hdf5(retrieval_save_path, "lens")  # Q or Qx2
                else:
                    if retrieval_pool_fname == 'self':
                        retrieval_data_array = data_array[:, retrieval_configs["used_col_indices"]].astype(int)
                        if retrieval_configs["label_wise"]:
                            retrieval_db_labels = data_array[:, -1].astype(int)
                        retrieved_indices = []
                        retrieved_values = []
                        retrieved_lens = []
                        fold_num = int(re.match("\d+-fold", retrieval_configs["split_type"]).group().split('-')[0])
                        fold_size = int(np.ceil(len(retrieval_data_array) / fold_num))
                        for fi in range(fold_num):
                            logging.info(f"{fold_num}-fold retrieval: process the {fi}-th fold")
                            fold_qry_data = retrieval_data_array[fi * fold_size: (fi + 1) * fold_size]
                            fold_db_data = np.concatenate(
                                [retrieval_data_array[: fi * fold_size], 
                                 retrieval_data_array[(fi + 1) * fold_size:]], 
                                axis=0)
                            fold_db_indices = np.concatenate(
                                [np.arange(fi * fold_size),
                                 np.arange((fi + 1) * fold_size, len(retrieval_data_array))],
                                axis=0)
                            if retrieval_configs["label_wise"]:
                                fold_db_labels = np.concatenate(
                                    [retrieval_db_labels[: fi * fold_size],
                                     retrieval_db_labels[(fi + 1) * fold_size:]],
                                    axis=0)
                                db_pos_indices = np.nonzero(fold_db_labels)[0]
                                fold_retrieved_pos_results = BM25_topk_retrieval(
                                    db_np_data=fold_db_data[db_pos_indices],
                                    qry_np_data=fold_qry_data, **retrieval_configs)
                                fold_retrieved_pos_indices = fold_db_indices[db_pos_indices[fold_retrieved_pos_results.indices]] # BxK
                                fold_retrieved_pos_values = fold_retrieved_pos_results.values # BxK
                                fold_retrieved_pos_lens = fold_retrieved_pos_results.lens # B
                                db_neg_indices = np.nonzero(1 - fold_db_labels)[0]
                                fold_retrieved_neg_results = BM25_topk_retrieval(
                                    db_np_data=fold_db_data[db_neg_indices],
                                    qry_np_data=fold_qry_data, **retrieval_configs)
                                fold_retrieved_neg_indices = fold_db_indices[db_neg_indices[fold_retrieved_neg_results.indices]] # BxK
                                fold_retrieved_neg_values = fold_retrieved_neg_results.values  # BxK
                                fold_retrieved_neg_lens = fold_retrieved_neg_results.lens  # B
                                retrieved_indices.append(
                                    np.concatenate([
                                        fold_retrieved_pos_indices,
                                        fold_retrieved_neg_indices], axis=-1))  # Bx(2K)
                                retrieved_values.append(
                                    np.concatenate([
                                        fold_retrieved_pos_values,
                                        fold_retrieved_neg_values], axis=-1))  # Bx(2K)
                                retrieved_lens.append(
                                    np.stack([
                                        fold_retrieved_pos_lens,
                                        fold_retrieved_neg_lens
                                    ], axis=-1))  # Bx2
                            else:
                                fold_retrieved_results = BM25_topk_retrieval(db_np_data=fold_db_data,
                                                                            qry_np_data=fold_qry_data,
                                                                            **retrieval_configs)
                                retrieved_indices.append(fold_db_indices[fold_retrieved_results.indices])  # BxK
                                retrieved_values.append(fold_retrieved_results.values)  # BxK
                                retrieved_lens.append(fold_retrieved_results.lens)  # B
                        retrieved_indices = np.concatenate(retrieved_indices)  # QxK or Qx(2K)
                        retrieved_values = np.concatenate(retrieved_values)  # QxK or Qx(2K)
                        retrieved_lens = np.concatenate(retrieved_lens)  # Q or Qx2
                    else:
                        db_data = db_array[:, retrieval_configs["used_col_indices"]].astype(int) # NdbxF'
                        qry_data = data_array[:, retrieval_configs["used_col_indices"]].astype(int) # NqxF'
                        if retrieval_configs["label_wise"]:
                            db_labels = db_array[:, -1].astype(int)
                            db_pos_indices = np.nonzero(db_labels)[0]
                            retrieved_pos_results = BM25_topk_retrieval(
                                db_np_data=db_data[db_pos_indices],
                                qry_np_data=qry_data, **retrieval_configs)
                            retrieved_pos_indices = db_pos_indices[retrieved_pos_results.indices] # QxK
                            retrieved_pos_values = retrieved_pos_results.values # QxK
                            retrieved_pos_lens = retrieved_pos_results.lens # Q
                            db_neg_indices = np.nonzero(1 - db_labels)[0]
                            retrieved_neg_results = BM25_topk_retrieval(
                                db_np_data=db_data[db_neg_indices],
                                qry_np_data=qry_data, **retrieval_configs)
                            retrieved_neg_indices = db_neg_indices[retrieved_neg_results.indices] # QxK
                            retrieved_neg_values = retrieved_neg_results.values # QxK
                            retrieved_neg_lens = retrieved_neg_results.lens # Q
                            retrieved_indices =  np.concatenate([
                                retrieved_pos_indices,
                                retrieved_neg_indices], axis=-1)  # Qx(2K)
                            retrieved_values =  np.concatenate([
                                retrieved_pos_values,
                                retrieved_neg_values], axis=-1)  # Qx(2K)
                            retrieved_lens =  np.stack([
                                retrieved_pos_lens,
                                retrieved_neg_lens
                            ], axis=-1)  # Qx2
                        else:
                            retrieved_results = BM25_topk_retrieval(db_np_data=db_data,
                                                                    qry_np_data=qry_data, 
                                                                    **retrieval_configs)
                            retrieved_indices = retrieved_results.indices
                            retrieved_values = retrieved_results.values
                            retrieved_lens = retrieved_results.lens
                    save_hdf5(retrieved_indices, retrieval_save_path, "indices")  # QxK or Qx(2K)
                    save_hdf5(retrieved_values, retrieval_save_path, "values")  # QxK or Qx(2K)
                    save_hdf5(retrieved_lens, retrieval_save_path, "lens")  # Q or Qx2
                if retrieval_augmented:
                    self.dataset = Dataset(
                        darray=data_array, 
                        feature_map=feature_map,
                        graph_processor=self.graph_processor,
                        retr_pool_darray=data_array if retrieval_pool_fname == 'self' else db_array, # QxF or NdbxF
                        retr_indices=retrieved_indices,  # QxK or Qx(2K)
                        retr_values=retrieved_values,  # QxK or Qx(2K)
                        retr_lens=retrieved_lens)  # Q or Qx2
                else:
                    logging.info("[[WARNING]] dataloader provided retrieved samples but the model doesn't enable retrieval-augmentated mode.")
                    self.dataset = Dataset(
                        darray=data_array[:, 0],  # Qx(1+K)xF => QxF
                        feature_map=feature_map,
                        graph_processor=self.graph_processor)
            else:
                raise NotImplementedError("we have only implemented the pre-retrieval strategy")
        else:
            assert not retrieval_augmented, "retrieval-augmented mode requires data_array like [Bx(1+K)x(F+1)]"
            self.dataset = Dataset(
                darray=data_array, 
                feature_map=feature_map,
                graph_processor=self.graph_processor)
        super(DataGenerator, self).__init__(dataset=self.dataset, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=num_workers, 
                                            collate_fn=graph_collate_fn if self.graph_processor else None)
        self.num_blocks = 1
        self.num_batches = int(np.ceil(len(self.dataset) * 1.0 / self.batch_size))
        self.num_samples = len(data_array)
        if data_array.ndim == 2:
            self.num_positives = data_array[:, -1].sum()
        elif data_array.ndim == 3:
            self.num_positives = data_array[:, 0, -1].sum()
        else:
            raise RuntimeError("data_array must be like [Nx(F+1)] or [Nx(K+1)x(F+1)]")
        self.num_negatives = self.num_samples - self.num_positives

    def __len__(self):
        return self.num_batches


class DataBlockGenerator(object):
    def __init__(self, data_block_list, batch_size=32, 
                 shuffle=False, 
                 feature_map=None, 
                 graph_processor=None, 
                 retrieval_configs=None, 
                 retrieval_pool_fname=None, 
                 retrieval_augmented=False, 
                 **kwargs):
        # data_block_list: path list of data blocks
        self.data_blocks = data_block_list
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_blocks = len(self.data_blocks)
        self.num_batches, self.num_samples, self.num_positives, self.num_negatives \
            = self.count_batches_and_samples()
        self.feature_map = feature_map
        self.graph_processor = graph_processor
        if self.graph_processor:
            self.graph_processor = getattr(dataset_utils, graph_processor)
        self.retrieval_configs = retrieval_configs
        self.retrieval_pool_fname = retrieval_pool_fname
        self.retrieval_augmented = retrieval_augmented

    def iter_block(self, data_block):
        darray = load_hdf5(data_block, verbose=False)
        if self.retrieval_configs is not None:
            assert self.retrieval_configs["pre_retrieval"], "we have only implemented the pre-retrieval strategy"
            if self.retrieval_configs["pre_retrieval"]:
                if self.retrieval_pool_fname != "self":
                    logging.info(f"{self.retrieval_configs['split_type']} retrieval, pool file: {self.retrieval_pool_fname}")
                    db_array = load_hdf5(self.retrieval_pool_fname)  # NdbxF
                data_root, data_fname = os.path.split(data_block)
                retrieval_save_path = os.path.join(
                    data_root, 
                    f'retrieval_{retrieval_configs["topK"]}_' + data_fname)
                if os.path.exists(retrieval_save_path):
                    retrieved_indices = load_hdf5(retrieval_save_path, "indices")
                    retrieved_values = load_hdf5(retrieval_save_path, "values")
                    retrieved_lens = load_hdf5(retrieval_save_path, "lens")
                else:
                    # retrieval within the same block. TODO: inter-block retrieval
                    if self.retrieval_pool_fname == 'self':
                        retrieval_data_array = darray[:, self.retrieval_configs["used_col_indices"]].astype(int)
                        if self.retrieval_configs["label_wise"]:
                            retrieval_db_labels = darray[:, -1].astype(int)
                        retrieved_indices = []
                        retrieved_values = []
                        retrieved_lens = []
                        fold_num = int(re.match("\d+-fold", self.retrieval_configs["split_type"]).group().split('-')[0])
                        fold_size = int(np.ceil(len(retrieval_data_array) / fold_num))
                        for fi in range(fold_num):
                            logging.info(f"{fold_num}-fold retrieval for {data_block}: process the {fi}-th fold")
                            fold_qry_data = retrieval_data_array[fi * fold_size: (fi + 1) * fold_size]
                            fold_db_data = np.concatenate(
                                [retrieval_data_array[: fi * fold_size], 
                                 retrieval_data_array[(fi + 1) * fold_size:]], 
                                axis=0)
                            fold_db_indices = np.concatenate(
                                [np.arange(fi * fold_size),
                                 np.arange((fi + 1) * fold_size, len(retrieval_data_array))],
                                axis=0)
                            if self.retrieval_configs["label_wise"]:
                                fold_db_labels = np.concatenate(
                                    [retrieval_db_labels[: fi * fold_size],
                                     retrieval_db_labels[(fi + 1) * fold_size:]],
                                    axis=0)
                                db_pos_indices = np.nonzero(fold_db_labels)[0]
                                fold_retrieved_pos_results = BM25_topk_retrieval(
                                    db_np_data=fold_db_data[db_pos_indices],
                                    qry_np_data=fold_qry_data, **(self.retrieval_configs))
                                fold_retrieved_pos_indices = fold_db_indices[db_pos_indices[fold_retrieved_pos_results.indices]] # BxK
                                fold_retrieved_pos_values = fold_retrieved_pos_results.values # BxK
                                fold_retrieved_pos_lens = fold_retrieved_pos_results.lens # B
                                db_neg_indices = np.nonzero(1 - fold_db_labels)[0]
                                fold_retrieved_neg_results = BM25_topk_retrieval(
                                    db_np_data=fold_db_data[db_neg_indices],
                                    qry_np_data=fold_qry_data, **(self.retrieval_configs))
                                fold_retrieved_neg_indices = fold_db_indices[db_neg_indices[fold_retrieved_neg_results.indices]] # BxK
                                fold_retrieved_neg_values = fold_retrieved_neg_results.values  # BxK
                                fold_retrieved_neg_lens = fold_retrieved_neg_results.lens  # B
                                retrieved_indices.append(
                                    np.concatenate([
                                        fold_retrieved_pos_indices,
                                        fold_retrieved_neg_indices], axis=-1))  # Bx(2K)
                                retrieved_values.append(
                                    np.concatenate([
                                        fold_retrieved_pos_values,
                                        fold_retrieved_neg_values], axis=-1))  # Bx(2K)
                                retrieved_lens.append(
                                    np.stack([
                                        fold_retrieved_pos_lens,
                                        fold_retrieved_neg_lens
                                    ], axis=-1))  # Bx2
                            else:
                                fold_retrieved_results = BM25_topk_retrieval(db_np_data=fold_db_data,
                                                                            qry_np_data=fold_qry_data,
                                                                            **(self.retrieval_configs))
                                retrieved_indices.append(fold_db_indices[fold_retrieved_results.indices])  # BxK
                                retrieved_values.append(fold_retrieved_results.values)  # BxK
                                retrieved_lens.append(fold_retrieved_results.lens)  # B
                        retrieved_indices = np.concatenate(retrieved_indices)  # QxK or Qx(2K)
                        retrieved_values = np.concatenate(retrieved_values)  # QxK or Qx(2K)
                        retrieved_lens = np.concatenate(retrieved_lens)  # Q or Qx2
                    else:
                        db_data = db_array[:, self.retrieval_configs["used_col_indices"]].astype(int) # NdbxF'
                        qry_data = darray[:, self.retrieval_configs["used_col_indices"]].astype(int) # QxF'
                        if self.retrieval_configs["label_wise"]:
                            db_labels = db_array[:, -1].astype(int)
                            db_pos_indices = np.nonzero(db_labels)[0]
                            retrieved_pos_results = BM25_topk_retrieval(
                                db_np_data=db_data[db_pos_indices],
                                qry_np_data=qry_data, **(self.retrieval_configs))
                            retrieved_pos_indices = db_pos_indices[retrieved_pos_results.indices] # QxK
                            retrieved_pos_values = retrieved_pos_results.values # QxK
                            retrieved_pos_lens = retrieved_pos_results.lens # Q
                            db_neg_indices = np.nonzero(1 - db_labels)[0]
                            retrieved_neg_results = BM25_topk_retrieval(
                                db_np_data=db_data[db_neg_indices],
                                qry_np_data=qry_data, **(self.retrieval_configs))
                            retrieved_neg_indices = db_neg_indices[retrieved_neg_results.indices] # QxK
                            retrieved_neg_values = retrieved_neg_results.values # QxK
                            retrieved_neg_lens = retrieved_neg_results.lens # Q
                            retrieved_indices =  np.concatenate([
                                retrieved_pos_indices,
                                retrieved_neg_indices], axis=-1)  # Qx(2K)
                            retrieved_values =  np.concatenate([
                                retrieved_pos_values,
                                retrieved_neg_values], axis=-1)  # Qx(2K)
                            retrieved_lens =  np.stack([
                                retrieved_pos_lens,
                                retrieved_neg_lens
                            ], axis=-1)  # Qx2
                        else:
                            retrieved_results = BM25_topk_retrieval(db_np_data=db_data,
                                                                    qry_np_data=qry_data, 
                                                                    **(self.retrieval_configs))
                            retrieved_indices = retrieved_results.indices
                            retrieved_values = retrieved_results.values
                            retrieved_lens = retrieved_results.lens
                    save_hdf5(retrieved_indices, retrieval_save_path, "indices")  # QxK or Qx(2K)
                    save_hdf5(retrieved_values, retrieval_save_path, "values")  # QxK or Qx(2K)
                    save_hdf5(retrieved_lens, retrieval_save_path, "lens")  # Q or Qx2
                # WARNINGS: possible OOM due to the large memory overhead of 'retrieved_samples'
                #  [solution] do not process the whole datablock at once; process it by mini-batches
                if self.retrieval_pool_fname == 'self':
                    retrieved_samples = darray[retrieved_indices]  # QxKxF or Qx(2K)xF
                else:
                    retrieved_samples = db_array[retrieved_indices]  # QxKxF or Qx(2K)xF
                del retrieved_indices
                gc.collect()
                darray = np.expand_dims(darray, 1) # Qx1xF
                darray = np.concatenate([darray, retrieved_samples], 1) # Qx(1+K)xF or Qx(1+2K)xF
                del retrieved_samples
                gc.collect()
            else:
                raise NotImplementedError("we have only implemented the pre-retrieval strategy")
            if not self.retrieval_augmented:
                logging.info("[[WARNING]] dataloader provided retrieved samples but the model doesn't enable retrieval-augmentated mode.")
                darray = darray[:, 0] # Bx(1+K)x(F+1) => Bx(F+1)

        if self.graph_processor:
            darray = self.self.graph_processor.convert_indices(
                darray, self.feature_map.feature_specs)
            graph_list = []
            for i in range(len(darray)):
                graph = self.graph_processor.build_instance_graph(
                    darray[i, ..., :-1], darray[i, ..., -1])
                graph_list.append(graph)
            X = np.array(graph_list)
        else:
            X = torch.from_numpy(darray[..., :-1])
        y = torch.from_numpy(darray[..., -1])
        if self.retrieval_configs is not None:
            retrieved_values = torch.from_numpy(retrieved_values)
            retrieved_lens = torch.from_numpy(retrieved_lens)
        block_size = len(y)
        indexes = list(range(block_size))
        if self.shuffle:
            np.random.shuffle(indexes)
        if self.retrieval_configs is not None and self.retrieval_augmented:
            for idx in range(0, block_size, self.batch_size):
                batch_index = indexes[idx:(idx + self.batch_size)]
                if self.graph_processor:
                    yield dgl.batch(X[batch_index]), y[batch_index], retrieved_values[batch_index], retrieved_lens[batch_index]
                else:
                    yield X[batch_index], y[batch_index], retrieved_values[batch_index], retrieved_lens[batch_index]
        else:
            for idx in range(0, block_size, self.batch_size):
                batch_index = indexes[idx:(idx + self.batch_size)]
                if self.graph_processor:
                    yield dgl.batch(X[batch_index]), y[batch_index]
                else:
                    yield X[batch_index], y[batch_index]

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.data_blocks)
        return chain.from_iterable(map(self.iter_block, self.data_blocks))

    def __len__(self):
        return self.num_batches

    def count_batches_and_samples(self):
        num_samples = 0
        num_positives = 0
        num_batches = 0
        for block_path in self.data_blocks:
            with h5py.File(block_path, 'r') as hf:
                data_array = hf[list(hf.keys())[0]][()]
                num_samples += len(data_array)
                if data_array.ndim == 2:
                    num_positives += np.sum(data_array[:, -1])
                elif data_array.ndim == 3: # Nqx(K+1)x(F+1)
                    num_positives += np.sum(data_array[:, 0, -1])
                else:
                    raise RuntimeError("data_array must be like [Nx(F+1)] or [Nx(K+1)x(F+1)]")
                num_batches += int(np.ceil(len(data_array) * 1.0 / self.batch_size))
        num_negatives = num_samples - num_positives
        return num_batches, num_samples, num_positives, num_negatives


def get_data_generator(
    data_path_list, 
    batch_size=32, 
    shuffle=False, 
    num_workers=1, 
    feature_map=None, 
    retrieval_configs=None,
    retrieval_pool_fname=None, 
    retrieval_augmented=False, 
    **kwargs):
    assert len(data_path_list) > 0, "invalid data files or paths."
    if len(data_path_list) == 1:
        return DataGenerator(data_path=data_path_list[0], 
                             batch_size=batch_size, 
                             shuffle=shuffle, 
                             num_workers=num_workers, 
                             feature_map=feature_map, 
                             retrieval_configs=retrieval_configs,
                             retrieval_pool_fname=retrieval_pool_fname,
                             retrieval_augmented=retrieval_augmented, 
                             **kwargs)
    else:
        return DataBlockGenerator(data_block_list=data_path_list, 
                                  batch_size=batch_size, 
                                  shuffle=shuffle, 
                                  feature_map=feature_map, 
                                  retrieval_configs=retrieval_configs,
                                  retrieval_pool_fname=retrieval_pool_fname,
                                  retrieval_augmented=retrieval_augmented, 
                                  **kwargs)
