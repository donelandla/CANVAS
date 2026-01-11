import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CrossAttention

def modulate(x, gamma, beta):
    return x * (1 + gamma) + beta


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FusionLayer(nn.Module):
    def __init__(self, hidden_size=1024, mlp_ratio=4.0, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, num_heads, dropout=0.1)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ffn = Mlp(hidden_size, int(hidden_size * mlp_ratio), drop=0.1)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, y):
        alpha_attn, beta_attn, gamma_attn, alpha_ffn, beta_ffn, gamma_ffn = self.adaLN_modulation(y).chunk(6, dim=2)
        alpha_attn, beta_attn, gamma_attn, alpha_ffn, beta_ffn, gamma_ffn = torch.tanh(alpha_attn), torch.tanh(beta_attn), torch.tanh(gamma_attn), torch.tanh(alpha_ffn), torch.tanh(beta_ffn), torch.tanh(gamma_ffn)
        x = x + torch.sigmoid(alpha_attn) * self.cross_attn(modulate(self.norm1(x), gamma_attn, beta_attn), y)
        x = x + torch.sigmoid(alpha_ffn) * self.ffn(modulate(self.norm2(x), gamma_ffn, beta_ffn))
        return x
