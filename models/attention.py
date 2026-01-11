import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, input_dim=1024, num_heads=8, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.dropout = dropout

        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        
        self.out_proj = nn.Linear(input_dim, input_dim)

    def forward(self, q, kv):
        B, Lq, _ = q.shape
        B, Lkv, _ = kv.shape

        Q = self.q_proj(q)
        K = self.k_proj(kv)
        V = self.v_proj(kv)

        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lkv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lkv, self.num_heads, self.head_dim).transpose(1, 2)

        output = F.scaled_dot_product_attention(
            Q, K, V, 
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )

        output = output.transpose(1, 2).contiguous().view(B, Lq, self.input_dim)
        output = self.out_proj(output)

        return output

class SelfAttention(nn.Module):
    def __init__(self, input_dim=1024, num_heads=8, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.dropout = dropout

        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        B, L, _ = x.shape

        # 这里 q, k, v 来源都是 x
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        output = F.scaled_dot_product_attention(
            Q, K, V, 
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )

        output = output.transpose(1, 2).contiguous().view(B, L, self.input_dim)
        output = self.out_proj(output)

        return output
