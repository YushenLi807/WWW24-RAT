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
from torch import nn
import torch
from .sequence import KMaxPooling
from ...pytorch.torch_utils import get_activation
from copy import deepcopy

class APG_Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, 
                 is_weight_generated=False, is_bias_generated=False, 
                 decompose_rank=None, overparam_dim=None):
        super(APG_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_weight_generated = is_weight_generated
        self.is_bias_generated = is_bias_generated
        self.decompose_rank = None
        if self.is_weight_generated:
            self.weight = None
            self.decompose_rank = decompose_rank
            if self.decompose_rank is not None:
                if self.decompose_rank >= self.in_features or self.decompose_rank >= self.out_features:
                    print(f"[[WARNING]] decomposed rank [{self.decompose_rank}] >= min(in_feat [{self.in_features}], out_feat [{self.out_features}])")
                if overparam_dim is None:
                    self.U_matrix = nn.Linear(in_features=self.in_features, 
                                              out_features=self.decompose_rank, 
                                              bias=False)
                    self.V_matrix = nn.Linear(in_features=self.decompose_rank, 
                                              out_features=self.out_features, 
                                              bias=False)
                else:  # over parameterization
                    assert overparam_dim > self.in_features and overparam_dim > self.out_features, \
                        "requires overparameterization dimension > max(in_feat, out_feat)"
                    self.U_matrix = nn.Sequential(
                        nn.Linear(in_features=self.in_features, 
                                  out_features=overparam_dim, 
                                  bias=False), 
                        nn.Linear(in_features=overparam_dim, 
                                  out_features=self.decompose_rank, 
                                  bias=False), 
                    )
                    self.V_matrix = nn.Sequential(
                        nn.Linear(in_features=self.decompose_rank, 
                                  out_features=overparam_dim, 
                                  bias=False), 
                        nn.Linear(in_features=overparam_dim, 
                                  out_features=self.out_features, 
                                  bias=False), 
                    )
        else:
            self.weight = nn.Parameter(data=torch.zeros(self.in_features, self.out_features))
            nn.init.xavier_normal_(self.weight)

        if bias:
            self.bias = None if self.is_bias_generated else nn.Parameter(data=torch.zeros(self.out_features))
        else:
            self.bias = None

    def set_parameters(self, weight_tensor=None, bias_tensor=None):
        if weight_tensor is not None:
            assert self.weight is None, "Invalid operation: the layer weight is not empty"
            assert weight_tensor.ndim in [2, 3], \
                f"Only support weight_tensor in shape [Bx(I*O)] or [BxKx(I*O)], get input ndim = {weight_tensor.ndim}."
            # weight_tensor: Bx(I*O) or BxKx(I*O) => BxIxO or BxKxIxO
            if self.decompose_rank is not None:
                self.weight = weight_tensor.reshape([*weight_tensor.shape[:-1], self.decompose_rank, self.decompose_rank])
            else:
                self.weight = weight_tensor.reshape([*weight_tensor.shape[:-1], self.in_features, self.out_features])
        if bias_tensor is not None:
            assert self.bias is None, "invalid operation: the layer bias is not empty"
            assert bias_tensor.ndim in [2, 3], "only support weight_tensor in shape [BxO] or [BxKxO]"
            # bias_tensor: BxO or BxKxO
            self.bias = bias_tensor

    def forward(self, inp):
        out = inp if self.decompose_rank is None else self.U_matrix(inp)    
        # ...xI, ...xIxO => ...xIx1, ...xIxO => ...xIxO => ...xO
        out = (out.unsqueeze(-1) * self.weight).sum(-2)
        if self.decompose_rank is not None:
            out = self.V_matrix(out)
        if self.bias is not None:
            out = out + self.bias
        # clean current states
        if self.is_weight_generated:
            self.weight = None
        if self.bias is not None and self.is_bias_generated:
            self.bias = None
        return out


class MLP_Layer(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim=None, 
                 hidden_units=[], 
                 hidden_activations="ReLU",
                 output_activation=None, 
                 dropout_rates=0.0, 
                 batch_norm=False, 
                 use_bias=True):
        super(MLP_Layer, self).__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = [get_activation(x) for x in hidden_activations]
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if batch_norm:
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        if output_activation is not None:
            dense_layers.append(get_activation(output_activation))
        self.dnn = nn.Sequential(*dense_layers) # * used to unpack list
    
    def forward(self, inputs):
        return self.dnn(inputs)


class APG_MLP_Layer(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim=None, 
                 hidden_units=[], 
                 hidden_activations="ReLU",
                 output_activation=None, 
                 dropout_rates=0.0, 
                 batch_norm=False, 
                 use_bias=True, 
                 condition_mode="none", 
                 decompose_ranks=None, 
                 overparam_dims=None, 
                 meta_net_configs=None,
                 ):
        super(APG_MLP_Layer, self).__init__()
        self.condition_mode = condition_mode
        assert self.condition_mode in ["none", 'single', 'moe', 'self']
        
        # hyper-parameters
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        if not isinstance(decompose_ranks, list):
            decompose_ranks = [decompose_ranks] * (len(hidden_units) + 1 if output_dim is not None else len(hidden_units))
        if not isinstance(overparam_dims, list):
            overparam_dims = [overparam_dims] * (len(hidden_units) + 1 if output_dim is not None else len(hidden_units))
        if self.condition_mode != "none":
            '''meta_net_configs requires:
                - hidden_units=[], 
                - hidden_activations="ReLU",
                - output_activation=None, 
                - dropout_rates=0.0, 
                - batch_norm=False, 
                - use_bias=True
                - input_dim # => iff for 'single' and 'moe' mode
                - num_experts=1 # => iff for 'moe' mode
                - aggregation='mean' # => iff for 'moe' mode
            '''
            assert meta_net_configs is not None, f"Conditioning mode '{self.condition_mode}' requires meta-network configurations."
            if not isinstance(meta_net_configs, list):
                meta_net_configs = [deepcopy(meta_net_configs) for i in range(len(hidden_units) + 1 if output_dim is not None else len(hidden_units))]
        hidden_activations = [get_activation(x) for x in hidden_activations]
        hidden_units = [input_dim] + hidden_units
        
        # define how to create linear layer
        if self.condition_mode == 'none':
            def create_linear_layer(in_feats, out_feats, bias, layer_idx=None):
                return APG_Linear(in_features=in_feats, out_features=out_feats, bias=bias)
        else: # ['single', 'moe', 'self']
            def create_linear_layer(in_feats, out_feats, bias, layer_idx=None):
                return APG_Linear(in_features=in_feats, out_features=out_feats, bias=bias, 
                                  is_weight_generated=True, is_bias_generated=False, 
                                  decompose_rank=decompose_ranks[layer_idx], 
                                  overparam_dim=overparam_dims[layer_idx])
            # define how to create a hyper-layer
            def create_hyperlayer(layer_idx, input_dim, output_dim=None):
                if self.condition_mode == 'self':
                    meta_net_configs[layer_idx]['input_dim'] = input_dim
                elif self.condition_mode == 'moe':
                    num_experts = meta_net_configs[layer_idx]["num_experts"]
                    del meta_net_configs[layer_idx]["num_experts"]
                    self.aggregation = meta_net_configs[layer_idx]["aggregation"]
                    assert self.aggregation in ['mean', 'sum', 'attention', 'max'], \
                        f"undefined expert aggregation type '{self.aggregation}'."
                    del meta_net_configs[layer_idx]["aggregation"]
                if decompose_ranks[layer_idx] is not None:
                    meta_net_configs[layer_idx]['output_dim'] = (decompose_ranks[layer_idx] ** 2)
                else:
                    meta_net_configs[layer_idx]['output_dim'] = output_dim
                if self.condition_mode == 'moe':
                    MLPs = [MLP_Layer(**meta_net_configs[layer_idx]) for _ in range(num_experts)]
                    if self.aggregation == 'attention': 
                        # add a projection layer for attention query
                        MLPs = [nn.Linear(input_dim, meta_net_configs[layer_idx]['output_dim'])] + MLPs
                    return nn.ModuleList(MLPs)
                else:  # 'self', 'single'
                    return MLP_Layer(**meta_net_configs[layer_idx])
        
        # define network layers
        self.layers = nn.ModuleList()
        if self.condition_mode != 'none':
            self.hyperlayers = nn.ModuleList()
        for idx in range(len(hidden_units) - 1):
            self.layers.append(
                create_linear_layer(in_feats=hidden_units[idx], 
                                    out_feats=hidden_units[idx + 1], 
                                    bias=use_bias, 
                                    layer_idx=idx))
            if self.condition_mode != 'none':
                self.hyperlayers.append(
                    create_hyperlayer(layer_idx=idx, 
                                      input_dim=hidden_units[idx], 
                                      output_dim=hidden_units[idx + 1] * hidden_units[idx]))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                self.layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                self.layers.append(nn.Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            self.layers.append(
                create_linear_layer(in_feats=hidden_units[-1], 
                                    out_feats=output_dim, 
                                    bias=use_bias, 
                                    layer_idx=len(hidden_units) - 1))
            if self.condition_mode != 'none':
                self.hyperlayers.append(
                    create_hyperlayer(layer_idx=-1, 
                                      input_dim=hidden_units[-1], 
                                      output_dim=output_dim * hidden_units[-1]))
        if output_activation is not None:
            self.layers.append(get_activation(output_activation))

    def forward(self, inputs, conditions=None, condition_lens=None):
        # conditions: Bxd' or BxNexd'
        # condition_lens: B
        # inputs: Bxd
        assert inputs.ndim == 2, "Require input shape like [Bxd]"
        if self.condition_mode in ['none', 'self']:
            assert conditions is None, f"Conditioning mode '{self.condition_mode}' doesn't support forward conditions."
        elif self.condition_mode == 'moe':
            assert conditions.ndim == 3, f"Require conditions in shape [BxNexd]"
        elif self.condition_mode == 'single':
            assert conditions.ndim == 2, "Require conditions in shape [Bxd]"
        else:
            raise NotImplementedError(f"Undefined condition mode '{self.condition_mode}'.")
        outputs = inputs
        linear_layer_idx = 0
        for module in self.layers:
            if self.condition_mode != 'none' and isinstance(module, APG_Linear):
                if isinstance(self.hyperlayers[linear_layer_idx], nn.ModuleList): # moe
                    assert self.condition_mode == 'moe'
                    if self.aggregation == 'attention':
                        moe_query = self.hyperlayers[linear_layer_idx][0](outputs) # BxDh => BxDl
                        # stack(Bxd' => BxDl, Ne) => BxNexDl
                        moe_tensors = torch.stack([expert(conditions[:, i]) for i, expert in enumerate(self.hyperlayers[linear_layer_idx][1:])], dim=-2) # Bx...xNexd
                        # BxDl, BxNexDl => Bx1xDl, BxNexDl => BxNe
                        moe_weights = (moe_query.unsqueeze(-2) * moe_tensors).sum(-1)
                        if condition_lens is None:
                            moe_weights = moe_weights.softmax(dim=-1)
                        else: # masked softmax
                            moe_weights_drop_mask = (torch.arange(conditions.shape[-2]).expand(conditions.shape[:-1]).to(self.device) >= condition_lens.unsqueeze(-1)) # BxNe
                            moe_weights[moe_weights_drop_mask] = -torch.inf # BxNe
                            moe_weights = moe_weights.softmax(dim=-1).nan_to_num(nan=0)  # BxNe
                        # BxNexDl, BxNe => BxNexDl, BxNex1 (or Nex1) => BxDl
                        weight_tensor = (moe_tensors * moe_weights.unsqueeze(-1)).sum(-2)
                    else:
                        # stack(Bxd' => BxDl, Ne) => BxNexDl
                        moe_tensors = torch.stack([expert(conditions[:, i]) for i, expert in enumerate(self.hyperlayers[linear_layer_idx])], dim=-2) # Bx...xNexd
                        # Bx...xNexd => Bx...xd
                        weight_tensor = getattr(torch, self.aggregation)(moe_tensors, dim=-2)
                        if self.aggregation == 'max':
                            weight_tensor = weight_tensor.values
                elif self.condition_mode == 'self':
                    weight_tensor = self.hyperlayers[linear_layer_idx](outputs) # BxDl
                else: # 'single'
                    weight_tensor = self.hyperlayers[linear_layer_idx](conditions) # BxDl
                module.set_parameters(weight_tensor=weight_tensor)
                linear_layer_idx += 1
            outputs = module(outputs)
        return outputs


if __name__ == '__main__':
    import random, os
    from torchinfo import summary
    
    def get_activation(activation):
        if isinstance(activation, str):
            if activation.lower() == "relu":
                return nn.ReLU()
            elif activation.lower() == "sigmoid":
                return nn.Sigmoid()
            elif activation.lower() == "tanh":
                return nn.Tanh()
            else:
                return getattr(nn, activation)()
        else:
            return activation

    configs = dict(
        input_dim=100, 
        output_dim=10, 
        hidden_units=[128,64], 
        hidden_activations="ReLU",
        output_activation=None, 
        dropout_rates=0.0, 
        batch_norm=False, 
        use_bias=False, 
        condition_mode="moe", 
        decompose_ranks=3, 
        overparam_dims=200, 
        # meta_net_configs=None,
        meta_net_configs=dict(
            hidden_units=[32,8],
            hidden_activations="ReLU",
            output_activation=None,
            dropout_rates=0.0,
            batch_norm=False,
            use_bias=True,
            input_dim=200,  # => iff for 'single' and 'moe' mode
            num_experts=3,  # => iff for 'moe' mode
            aggregation='attention',  # => iff for 'moe' mode
        ),
        batch_size=1024, 
        seed=2023, 
    )
    random.seed(configs['seed'])
    os.environ["PYTHONHASHSEED"] = str(configs['seed'])
    np.random.seed(configs['seed'])
    torch.manual_seed(configs['seed'])
    torch.cuda.manual_seed(configs['seed'])
    torch.backends.cudnn.deterministic = True
    
    apg_model = APG_MLP_Layer(
        input_dim=configs['input_dim'], 
        output_dim=configs['output_dim'], 
        hidden_units=configs['hidden_units'],
        hidden_activations=configs['hidden_activations'], 
        output_activation=configs['output_activation'],
        dropout_rates=configs['dropout_rates'], 
        batch_norm=configs['batch_norm'], 
        use_bias=configs['use_bias'], 
        condition_mode=configs['condition_mode'],
        decompose_ranks=configs['decompose_ranks'], 
        overparam_dims=configs['overparam_dims'], 
        meta_net_configs=configs['meta_net_configs'], 
    )
    
    # summary(apg_model, input_size=(configs['batch_size'], configs['input_dim']), device='cpu')
    summary(apg_model, 
            input_size=[(configs['batch_size'], configs['input_dim']), 
                        (configs['batch_size'], 
                         configs['meta_net_configs']['num_experts'],
                         configs['meta_net_configs']['input_dim'])], 
            device='cpu')
    
    # model = MLP_Layer(
    #     input_dim=configs['input_dim'],
    #     output_dim=configs['output_dim'],
    #     hidden_units=configs['hidden_units'],
    #     hidden_activations=configs['hidden_activations'],
    #     output_activation=configs['output_activation'],
    #     dropout_rates=configs['dropout_rates'],
    #     batch_norm=configs['batch_norm'],
    #     use_bias=configs['use_bias'], 
    # )
    
    # summary(model, input_size=(configs['batch_size'], configs['input_dim']), device='cpu')

