import torch 
import torch.nn as nn


class LinearRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features=3)
        
    def forward(self, X: torch.Tensor):
        return self.linear(X)


class LR_ReLU(nn.Module):
    def __init__(self, in_features, hidden_size=16, out_features=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(inplace=False),          
            nn.Linear(hidden_size, out_features)
        )
        
    def forward(self, X: torch.Tensor):
        return self.model(X)