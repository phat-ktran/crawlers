import torch
from torch import nn
from torch.nn import functional as F

class BahdanauAttention(nn.Module):
    def __init__(
        self,
        enc_hidden_size: int,
        dec_hidden_size: int,
        att_dim: int,
        drop_rate: float = 0.5,
    ):
        super().__init__()
        self.U = nn.Linear(dec_hidden_size, att_dim)
        self.K = nn.Linear(enc_hidden_size, att_dim)
        self.v = nn.Linear(att_dim, 1)
        self.dropout_u = nn.Dropout(p=drop_rate)
        self.dropout_v = nn.Dropout(p=drop_rate)

    def forward(self, queries, keys, values, mask):
        original_dim = queries.dim()
        if original_dim == 2:
            queries = queries.unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        keys = self.K(keys)
        keys_exp = keys.unsqueeze(1)  # (batch_size, 1, seq_enc, att_dim)
        
        u = self.dropout_u(self.U(queries))  # (batch_size, seq_len, att_dim)
        u_exp = u.unsqueeze(2)  # (batch_size, seq_len, 1, att_dim)
        
        tanh_in = torch.tanh(keys_exp + u_exp)  # (batch_size, seq_len, seq_enc, att_dim)
        a = self.dropout_v(self.v(tanh_in)).squeeze(3)  # (batch_size, seq_len, seq_enc)
        a = a.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        alpha = F.softmax(a, dim=-1)  # (batch_size, seq_len, seq_enc)
        
        context = torch.matmul(alpha, values)  # (batch_size, seq_len, transform_hidden_size)
        if original_dim == 2:
            context = context.squeeze(1)

        return context