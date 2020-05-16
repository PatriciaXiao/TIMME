from torch.nn import MultiheadAttention
'''
sample usage of MultiheadAttention:
self_attention = MultiheadAttention(out_features, nhead=1, dropout=0.0)
self.attention = self_attention
outputs_extracted = torch.sum(torch.stack(outputs_raw, 0), 1) # n_adj * n_entities * embedding_size
# actually, the multi-head attention provided by PyTorch expects the input as:
#    https://github.com/pytorch/pytorch/blob/ec7bb9de1c5f46a9a6fef60c4c80aca126641b8c/torch/nn/modules/activation.py
# Q, K, V: all (n_length, batch_size, embedding_size)
# notice: there's no relation between batches, the actual matrix we want to attend to is (n_len, emb_size)
# outputs: (attended_output, attended_output_weights)
attended_output, attention_weight_raw = self.attention(outputs_extracted, outputs_extracted, outputs_extracted)
# print(attention_weight_raw.shape) # n_batch, n_adj, n_adj (n_adj = n_length)
# this way it works
# as another option of sum(attention_weight_raw), we can also attention_weight_raw[0] etc. for faster & comparible performance
attention_weight = torch.sum(sum(attention_weight_raw), 0)
outputs = F.softmax(attention_weight) * outputs
'''

# https://github.com/CyberZHG/torch-multi-head-attention/blob/66f6ae801a6d2aea8994ef00af06fdfc67ec2026/torch_multi_head_attention/multi_head_attention.py#L77
import torch.nn as nn
import torch.nn.functional as F
import math
import torch

class ScaledDotProductSelfAttention(nn.Module):
    def __init__(self, dim, n_entities):
        super().__init__()
        self.scale = math.sqrt(dim * n_entities) #  * n_entities no longer necessary since we are not concatenating
    def __call__(self, matrix):
        '''
        the matrix is turned into 2-D, concatenation of the embeddings (R * N * dim -> R * (N * dim))
        or sum-up (R * N * dim -> R * dim)
        return the attention of size R
        '''
        matrix = torch.mean(matrix, 1) # take average over the entities
        query = key = value = matrix # self-attention
        scores = query.matmul(key.transpose(0,1)) / self.scale
        attention = F.softmax(torch.sum(scores, 0), dim=-1)
        return attention