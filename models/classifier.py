import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  
            nn.ReLU(),                         
            nn.Dropout(0.3),                   
            nn.Linear(hidden_dim, 1)           
        )


    def forward(self, x):
        logits = self.classifier(x)
        return logits
