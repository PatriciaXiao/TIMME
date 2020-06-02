import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from model.attention import MultiheadAttention, ScaledDotProductSelfAttention

class GraphConvolution(Module):
    """
    Simple rGCN layer, similar to https://arxiv.org/abs/1703.06103
    :param num_relation: number of different relations in the data
    :param num_adjs: number of adjacency matrix in the dataset (not necessarily 2 * num_relation + 1)
    :param in_features: number of feature of the input
    :param out_features: number of feature of the ouput
    :param bias: if bias is added, default is True
    :type num_relation: int
    :type num_adjs: int
    :type num_neighbors: array-like object, must be 3 dimension
    :type in_features: int
    :type out_features: int
    :type bias: bool
    :type attention: string, options "none" / "self" / "naive"
    """
    def __init__(self, num_relation, num_entities, num_adjs, in_features, out_features, bias=True, \
                attention_mode="none"):
        super(GraphConvolution, self).__init__()
        self.num_relation = num_relation
        self.num_adjs = num_adjs
        self.in_features = in_features
        self.out_features = out_features
        self.adj_weight = Parameter(nn.init.kaiming_uniform_(torch.FloatTensor(num_adjs, in_features, out_features)))
        # naive attention, might or might not be used
        # always initialize to get rid of the influence of randomness
        if attention_mode not in ["naive", "none", "self"]:
            print("attention mode {} not recognized, treated as none".format(attention_mode))
            attention_mode = "none"
        self.attention_mode = attention_mode
        if self.attention_mode == "naive":
            self.attention = Parameter(nn.init.uniform_(torch.FloatTensor(num_adjs)))
        elif self.attention_mode == "self":
            self.attention = ScaledDotProductSelfAttention(out_features, num_entities)
        else:
            self.register_parameter('attention', None)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.adj_weight.size(2))
        self.adj_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_, adjs):
        outputs = []
        # CUDA = adjs[0].is_cuda
        for i in range(len(self.adj_weight)):
            support = torch.mm(input_, self.adj_weight[i])
            output = torch.spmm(adjs[i], support)
            outputs.append(output)

        outputs_raw = torch.stack(outputs)
        outputs = torch.stack(outputs, 2)

        if self.attention_mode == "naive":
            outputs = F.softmax(self.attention) * outputs
        elif self.attention_mode == "self":
            attention_weight = self.attention(outputs_raw)
            outputs = attention_weight * outputs
        else:
            # none attention
            outputs = outputs / self.num_adjs

        # attention to be added here
        output = torch.sum(outputs, 2)
        return output if self.bias is None else output + self.bias

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

    
    
    

