import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsort

class PositionEncoder(nn.Module):
    def __init__(self, device, n=224//14, embed_dim=512):
        super(PositionEncoder, self).__init__()
        self.n = n
        self.num_patches = n * n
        self.embed_dim = embed_dim
        self.device = device
        
        self.n_bins_H = 72
        self.n_bins_S = 256
        self.n_bins_V = 256
        self.eps = 1e-8
        self.linear = nn.Linear(3, 1)

        self.rank_embedding = nn.Parameter(torch.randn(self.num_patches, embed_dim, device=self.device))
        nn.init.normal_(self.rank_embedding, std=0.02)
        self.register_buffer('rank_indices', torch.arange(self.num_patches, dtype=torch.float32, device=device))
        self.spatial_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim, device=self.device))
        nn.init.trunc_normal_(self.spatial_embedding, std=0.02)
        
    def compute_ghe(self, patches):
        # patches: [B, N, 3, h, w]
        B, N, _, h, w = patches.shape
        
        # flatten each patch
        H = patches[:,:,0].reshape(B*N, -1)
        S = patches[:,:,1].reshape(B*N, -1)
        V = patches[:,:,2].reshape(B*N, -1)
        
        def compute_hist(vector, n_bins):
            vector = vector.contiguous()
            bins = torch.linspace(0, 1, n_bins + 1, device=vector.device)
            inds = torch.bucketize(vector, bins) - 1
            inds = torch.clamp(inds, 0, n_bins - 1)
            hist = torch.zeros((B*N, n_bins), device=vector.device)
            hist.scatter_add_(1, inds, torch.ones_like(vector))
            return hist
        
        hist_H = compute_hist(H, self.n_bins_H)
        kernel = torch.tensor([0.25, 0.5, 0.25], device=H.device).view(1,1,3)
        hist_H_pad = torch.cat([hist_H[:, -1:], hist_H, hist_H[:, :1]], dim=1).unsqueeze(1)
        hist_H = F.conv1d(hist_H_pad, kernel, padding=0).squeeze(1)
        
        hist_S = compute_hist(S, self.n_bins_S)
        hist_V = compute_hist(V, self.n_bins_V)
        
        def entropy(hist):
            p = hist / (hist.sum(dim=-1, keepdim=True) + self.eps)
            return -(p * torch.log(p + self.eps)).sum(dim=-1)
        
        ghe_H = entropy(hist_H)
        ghe_S = entropy(hist_S)
        ghe_V = entropy(hist_V)
        
        ghe_all = torch.stack([ghe_H, ghe_S, ghe_V], dim=1)  # [B*N, 3]
        ghe = self.linear(ghe_all)  # [B*N, 1]
        ghe = ghe.view(B, N)  # [B, N]
        
        return ghe

    def forward(self, img):
        B, C, H, W = img.shape
        n = self.n
        h, w = H // n, W // n
        N = n * n

        # patches: [B, N, C, h, w]
        patches = img.unfold(2, h, h).unfold(3, w, w)  # [B, C, n, h, n, w]
        patches = patches.permute(0,2,4,1,3,5).contiguous()   # [B, n, n, C, h, w]
        patches = patches.view(B, N, C, h, w)                 # [B, N, C, h, w]
        richness = self.compute_ghe(patches)         # [B, N]

        rank = torchsort.soft_rank(richness, regularization='l2')  # [B, N]
        dist = (rank.unsqueeze(-1) - self.rank_indices.unsqueeze(0)) ** 2
        rank_weights = F.softmax(-dist, dim=-1) # [B, N, N]
        rank_embeds = torch.matmul(rank_weights, self.rank_embedding)

        spatial_embeds = self.spatial_embedding
        pos_embeds = spatial_embeds + rank_embeds
        pos_embeds = pos_embeds.view(B, n, n, self.embed_dim).permute(0, 3, 1, 2).contiguous()

        return pos_embeds
