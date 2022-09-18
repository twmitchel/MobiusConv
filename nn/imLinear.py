import torch
import torch.nn as nn

class imLinear(nn.Module):
    
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
              
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):

        return torch.permute(self.lin(torch.permute(x, (0, 2, 3, 1))), (0, 3, 1, 2))