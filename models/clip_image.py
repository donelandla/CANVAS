from .clip import clip 
import torch.nn as nn

clip_name = "ViT-L/14"

CHANNELS = {
    "ViT-L/14" : 768
}

class CLIPModel(nn.Module):
    def __init__(self):
        super(CLIPModel, self).__init__()
        self.model, self.preprocess = clip.load(name=clip_name, device="cuda")
        self.visual = self.model.visual

        for param in self.model.parameters():
            param.requires_grad = False
 
 
    def forward(self, x):
        input_dtype = self.visual.conv1.weight.dtype
        x = x.type(input_dtype)
        features = self.visual(x, return_full=True)
        return features

