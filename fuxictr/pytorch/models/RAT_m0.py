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

from torch import nn, einsum
import torch
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import MLP_Layer, EmbeddingLayer, LR_Layer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class RAT_m0(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="RAT_m0", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 dnn_hidden_units=[64, 64, 64], 
                 dnn_activations="ReLU", 
                 attention_layers=2,
                 num_heads=1,
                 attention_dim=8,
                 net_dropout=0, 
                 batch_norm=False,
                 layer_norm=False,
                 use_scale=False,
                 use_wide=False,
                 use_residual=True,
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 depth=4,
                 heads=4,
                 pool='cls',
                 dim_head=10,
                 dropout=0.,
                 emb_dropout = 0., 
                 scale_dim = 4,
                 **kwargs):
        super(RAT_m0, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer, 
                                      net_regularizer=net_regularizer,
                                      **kwargs) 
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.label_embedding_layer = nn.Embedding(num_embeddings=3,  # binary classification
                                                    embedding_dim=embedding_dim)
        self.query_proj = nn.Linear(embedding_dim * feature_map.num_fields, 
                                    embedding_dim * feature_map.num_fields)
        self.query_dropout = None
        if net_dropout > 0:
            self.query_dropout = nn.Dropout(net_dropout)

        #Cross-Intra Attention
        k = kwargs['retrieval_configs']['topK']
        self.encoder = Transformer(embedding_dim, depth, num_heads, dim_head, embedding_dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool
        ###
        self.lr_layer = LR_Layer(feature_map, output_activation=None, use_bias=False) \
                        if use_wide else None
    
        self.dnn = MLP_Layer(input_dim=embedding_dim * feature_map.num_fields,
                            output_dim=1, 
                            hidden_units=dnn_hidden_units,
                            hidden_activations=dnn_activations,
                            output_activation=None, 
                            dropout_rates=net_dropout, 
                            batch_norm=batch_norm, 
                            use_bias=True) \
                if dnn_hidden_units else None # in case no DNN used
        
        self.fc = nn.Linear(embedding_dim , 1)
        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        
        # Bx(K+1)xF, Bx(K+1), BxK, B
        X, y, retrieved_values, retrieved_lens = self.inputs_to_device(inputs)
        k = retrieved_values.shape[-1]
        assert retrieved_lens.ndim == 1, "RIM does not support label-wise retrieval-enhanced training"
        retrieved_X, retrieved_y = X[:, 1:], y[:, 1:]  # BxKxF, BxKx1
        X, y = X[:, 0], y[:, 0]  # BxF, Bx1
        token = torch.ones((X.shape[0], 1))*2  # Bx1 token=2
        token = token.to(self.device)
        retrieved_emb = self.embedding_layer(retrieved_X)  # BxKxFxd
        retrieved_y_emb = self.label_embedding_layer(retrieved_y.long())  # BxKx1xd
        X = torch.unsqueeze(X, 1)  # Bx1xF
        X_emb = self.embedding_layer(X)
        token = torch.unsqueeze(token, 1)  # Bx1x1
        x_emb = self.embedding_layer(X)  # Bx1xFxd
        y_emb = self.label_embedding_layer(token.long())  # B×1×1×d
        target_emb = torch.concat([y_emb, x_emb], dim=2)  #Bx1x(F+1)xd
        retrieved_feature_emb = torch.concat([retrieved_y_emb, retrieved_emb], dim=2)  #BxKx(F+1)xd
        feature_emb = torch.concat([target_emb, retrieved_feature_emb], dim=1)  #Bx(K+1)x(F+1)xd

        ###
        x = feature_emb
        b, t, n, _ = x.shape
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> b (t n) d')
        x = self.encoder(x)
        x = rearrange(x, 'b (t n) d -> b t n d',t=t)
        x = x[:,:,0]  # b t d
        x = x[:,0]    # b d
        x = torch.flatten(x, start_dim=1)  # b d

        ###             

        y_pred = self.fc(x)
        if self.dnn is not None:
            y_pred += self.dnn(X_emb.flatten(start_dim=1))
        if self.lr_layer is not None:
            y_pred += self.lr_layer(X)
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
