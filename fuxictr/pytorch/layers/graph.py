# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (C) 2021. Tsinghua University. All rights reserved.
#
# Authors: Kelong Mao <Tsinghua University>
#          Jieming Zhu <Huawei Noah's Ark Lab>
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product

import dgl.function as fn
from dgl.nn.functional import edge_softmax


class FiGNN_Layer(nn.Module):
    def __init__(self, 
                 num_fields, 
                 embedding_dim,
                 gnn_layers=3,
                 reuse_graph_layer=False,
                 use_gru=True,
                 use_residual=True,
                 device=None):
        super(FiGNN_Layer, self).__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.gnn_layers = gnn_layers
        self.use_residual = use_residual
        self.reuse_graph_layer = reuse_graph_layer
        self.device = device
        if reuse_graph_layer:
            self.gnn = GraphLayer(num_fields, embedding_dim)
        else:
            self.gnn = nn.ModuleList([GraphLayer(num_fields, embedding_dim)
                                      for _ in range(gnn_layers)])
        self.gru = nn.GRUCell(embedding_dim, embedding_dim) if use_gru else None
        self.src_nodes, self.dst_nodes = zip(*list(product(range(num_fields), repeat=2)))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.W_attn = nn.Linear(embedding_dim * 2, 1, bias=False)

    def build_graph_with_attention(self, feature_emb):
        src_emb = feature_emb[:, self.src_nodes, :]
        dst_emb = feature_emb[:, self.dst_nodes, :]
        concat_emb = torch.cat([src_emb, dst_emb], dim=-1)
        alpha = self.leaky_relu(self.W_attn(concat_emb))
        alpha = alpha.view(-1, self.num_fields, self.num_fields)
        mask = torch.eye(self.num_fields).to(self.device)
        alpha = alpha.masked_fill(mask.bool(), float('-inf'))
        graph = F.softmax(alpha, dim=-1) # batch x field x field without self-loops
        return graph

    def forward(self, feature_emb):
        g = self.build_graph_with_attention(feature_emb)
        h = feature_emb
        for i in range(self.gnn_layers):
            if self.reuse_graph_layer:
                a = self.gnn(g, h)
            else:
                a = self.gnn[i](g, h)
            if self.gru is not None:
                a = a.view(-1, self.embedding_dim)
                h = h.view(-1, self.embedding_dim)
                h = self.gru(a, h)
                h = h.view(-1, self.num_fields, self.embedding_dim)
            else:
                h = a + h
            if self.use_residual:
                h += feature_emb
        return h


class GraphLayer(nn.Module):                        # input : num_fields,embedding_dim     output : [num_fields, embedding_dim]
    def __init__(self, num_fields, embedding_dim):
        super(GraphLayer, self).__init__()
        self.W_in = torch.nn.Parameter(torch.Tensor(num_fields, embedding_dim, embedding_dim))
        self.W_out = torch.nn.Parameter(torch.Tensor(num_fields, embedding_dim, embedding_dim))
        nn.init.xavier_normal_(self.W_in)
        nn.init.xavier_normal_(self.W_out)
        self.bias_p = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, g, h):
        h_out = torch.matmul(self.W_out, h.unsqueeze(-1)).squeeze(-1) # broadcast multiply
        aggr = torch.bmm(g, h_out)
        a = torch.matmul(self.W_in, aggr.unsqueeze(-1)).squeeze(-1) + self.bias_p
        return a


class HypergraphLayer(nn.Module):
    def __init__(self, in_feat, hidden_feat):
        super(HypergraphLayer, self).__init__()
        self.K = nn.Linear(2 * in_feat, hidden_feat)
        self.V = nn.Linear(2 * in_feat, hidden_feat)
        self.Q = nn.Linear(in_feat, hidden_feat)
        self.W = nn.Linear(in_feat + hidden_feat, hidden_feat)
        self.edge_W = nn.Linear(2 * hidden_feat + in_feat, hidden_feat)

    def forward(self, graph):
        edge_embds = graph.edata['h']
        src, dst = graph.edges()
        node_embds = graph.ndata['h']
        src_messages = node_embds[src] * edge_embds
        src_messages = torch.cat((src_messages, node_embds[src]), 1)

        graph.ndata['Q'] = self.Q(graph.ndata['h'])
        graph.edata['K'] = self.K(src_messages)
        graph.edata['V'] = self.V(src_messages)

        graph.apply_edges(fn.v_mul_e('Q', 'K', 'alpha'))
        graph.edata['alpha'] = edge_softmax(graph, graph.edata['alpha'])
        graph.edata['V'] = graph.edata['alpha'] * graph.edata['V']

        graph.update_all(fn.copy_e('V', 'h_n'), fn.sum('h_n', 'h_n'))
        graph.ndata['h'] = self.W(
            torch.cat((graph.ndata['h_n'], graph.ndata['h']), 1))

        edge_embds = torch.cat(
            (graph.ndata['h'][src], graph.ndata['h'][dst], graph.edata['h']), 1)
        edge_embds = self.edge_W(edge_embds)
        graph.edata['h'] = edge_embds

        return graph


class PET_Layer(nn.Module):
    def __init__(self, num_layers, in_feat, hidden_feat, dropout=0.1):
        super(PET_Layer, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(HypergraphLayer(in_feat=in_feat, hidden_feat=hidden_feat))
        for _ in range(self.num_layers - 1):
            self.layers.append(
                HypergraphLayer(in_feat=hidden_feat, hidden_feat=hidden_feat))
        # layernorm before each propogation
        self.layernorm = nn.LayerNorm(hidden_feat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph):
        for i in range(self.num_layers):
            graph = self.layers[i](graph)
            graph.ndata['h'] = F.relu_(graph.ndata['h'])
            graph.ndata['h'] = self.dropout(self.layernorm(graph.ndata['h']))
            graph.edata['h'] = F.relu_(graph.edata['h'])
            graph.edata['h'] = self.dropout(self.layernorm(graph.edata['h']))
        return graph

