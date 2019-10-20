import math

import numpy
import torch
from torch import nn


MIN_VALUE = float(numpy.finfo(numpy.float32).min)

def diagonal_mask(size, width=0, device='cpu', dtype=torch.uint8):
    """Create mask for restrict attention (1, size, size)

    :param int size: size of mask
    :param int width: width of the restricted range around the diagonal
    :param str device: "cpu" or "cuda" or torch.Tensor.device
    :param torch.dtype dtype: result dtype
    :rtype: torch.Tensor
    >>> diagonal_mask(3, width=1)
    [[1, 1, 0, 0],
     [1, 1, 1, 0],
     [0, 1, 1, 1],
     [0, 0, 1, 1]]
    """
    ret = torch.ones(size, size, device=device, dtype=dtype)
    return ret.tril(diagonal=width).triu(diagonal=-width)


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    :param int restricted: attention restriction, -1 means no restriction
    """

    def __init__(self, n_head, n_feat, dropout_rate, restricted=-1):
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)
        self.restricted = restricted

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            scores = scores.masked_fill(mask, MIN_VALUE)
        
        if self.restricted > 0:
            max_dim = max(scores.size(2), scores.size(3))
            restricted_mask = diagonal_mask(max_dim, self.restricted, device=scores.device)[:scores.size(2), :scores.size(3)]
            restricted_mask = restricted_mask.unsqueeze(0).unsqueeze(1) # (batch, 1, time1, time1)
            scores = scores.masked_fill(restricted_mask == 0, MIN_VALUE)
            if mask is not None:
                    self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
            else:
                self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
        elif self.restricted < 0:
            if mask is not None:
                    self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
            else:
                self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
        elif self.restricted == 0:
            assert scores.size(2) >= scores.size(3), "does not make sense to have output length less than the input length while have diagonal attention"
            self.attn = torch.eye(scores.size(3), device=scores.device)
            last_rows = self.attn[-1].repeat(scores.size(2)-scores.size(3), 1)
            self.attn = torch.cat((self.attn, last_rows))
            self.attn = self.attn.repeat(scores.size(0), scores.size(1), 1, 1)      

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)
