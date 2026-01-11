import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip_image import CLIPModel
from .hsv_encoder import HSVEncoder
from .position import PositionEncoder
from .fusion import FusionLayer
from .classifier import Classifier
from .attention import SelfAttention


class CANVAS(nn.Module):
    def __init__(self, 
                input_dim=1024, 
                num_heads=8, 
                hidden_dim=256,
                device='cuda'):
        super().__init__()

        self.clip = CLIPModel()
        self.hsv_encoder = HSVEncoder(output_dim=input_dim)
        self.pos_encoder = PositionEncoder(device=device, n=224//14, embed_dim=input_dim)

        self.img_norm = nn.LayerNorm(input_dim, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(input_dim, num_heads, dropout=0.1)
        self.hsv_norm = nn.LayerNorm(input_dim, elementwise_affine=False, eps=1e-6)
        self.hsv_attn = SelfAttention(input_dim, num_heads, dropout=0.1)

        self.fusion = FusionLayer(hidden_size=input_dim, num_heads=num_heads)
        self.classifier = Classifier(input_dim=input_dim, hidden_dim=hidden_dim)

    def forward(self, hsv_input, clip_input, targets, train=True, temperature=0.07):
        with torch.no_grad():
            x = self.clip(clip_input).float()
        y = self.hsv_encoder(hsv_input).float()
        p = self.pos_encoder(hsv_input).float()
        
        y = y + p 

        y = y.flatten(2).transpose(1, 2) # [B, H*W, C]
    
        x = x + self.img_attn(self.img_norm(x))
        y = y + self.hsv_attn(self.hsv_norm(y))

        fused = self.fusion(x, y)
        fused_global = fused.mean(dim=1)
        cla_logits = self.classifier(fused_global)

        if train:
            # contrastive loss
            clip_feat_global = x.mean(dim=1)  
            hsv_feat_global = y.mean(dim=1)
            clip_feat = F.normalize(clip_feat_global, dim=-1)   
            hsv_feat = F.normalize(hsv_feat_global, dim=-1)    
            con_logits = torch.matmul(hsv_feat, clip_feat.T) / temperature
            labels = torch.arange(hsv_feat.size(0)).to(hsv_feat.device)
            con_loss = F.cross_entropy(con_logits, labels)

            # classification loss
            cla_loss = F.binary_cross_entropy_with_logits(cla_logits, targets)
            return con_loss, cla_loss
        else:
            probs = torch.sigmoid(cla_logits)
            return probs
