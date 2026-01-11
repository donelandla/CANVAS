import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import math

class HSVEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1,1,3,3))
        self.register_buffer('sobel_y', sobel_y.view(1,1,3,3))

        backbone = resnet50(weights=None)
        backbone.conv1 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.proj = nn.Conv2d(2048, output_dim, kernel_size=1)

        self.norm_s = nn.BatchNorm2d(1, affine=True, eps=1e-6)
        self.norm_v = nn.BatchNorm2d(1, affine=True, eps=1e-6)
        self.input_bn = nn.BatchNorm2d(8, eps=1e-6)

        self._init_weights()

    def gradients(self, x):
        B, C, H, W = x.shape
        gx = F.conv2d(x, self.sobel_x.repeat(C,1,1,1), groups=C, padding=1)
        gy = F.conv2d(x, self.sobel_y.repeat(C,1,1,1), groups=C, padding=1)
        return torch.cat([gx, gy], dim=1)  # [B, 2*C, H, W]

    def forward(self, hsv):
        H = hsv[:, 0:1, :, :] * 2 * math.pi
        S = hsv[:, 1:2, :, :]
        V = hsv[:, 2:3, :, :]

        h_sin = torch.sin(H)
        h_cos = torch.cos(H)
        h_feat = torch.cat([h_sin, h_cos], dim=1)  # [B, 2, H, W]

        s_feat = self.norm_s(S)
        v_feat = self.norm_v(V)
        base_feats = torch.cat([h_feat, s_feat, v_feat], dim=1)  # [B,4,H,W]

        feats = self.gradients(base_feats)  # [B,8,H,W]
        feats = self.input_bn(feats)
        x = self.backbone(feats)  # [B,512,H/32,W/32]
        x = self.proj(x)  # [B,output_dim,H/32,W/32]

        x = F.interpolate(x, size=(16, 16), mode='bilinear', align_corners=False)
        
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
