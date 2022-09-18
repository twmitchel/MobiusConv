import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import realSgn

from utils.rs_diff import rsOrder01, rsNorm2, rsDirichlet

EPSSigma = 1e-6;

###########################################################
### Filter Response Normalization with Dirichlet energy ###
### followed by a thresholded activation                ###
###########################################################

class FRDirichlet(nn.Module):
    
    def __init__(self, B, channels):
        super(FRDirichlet, self).__init__()

        ''' 
        Inputs:
            B: Spherical Harmonic bandlimit
            
            channels: # of channels
        '''
        self.B = B;
                
        self.channels = channels;
    
        self.scale = torch.nn.Parameter(torch.Tensor(1, channels));
        
        self.bias = torch.nn.Parameter(torch.Tensor(1, channels));
        
        self.eps = torch.nn.Parameter(torch.Tensor(1, channels));
        
        self.dirichlet = rsDirichlet(B)
        
        self.th = torch.nn.Parameter(torch.Tensor(1, channels));
        
        
        torch.nn.init.ones_(self.scale);        
        torch.nn.init.zeros_(self.bias)
        torch.nn.init.xavier_uniform_(self.th);
        torch.nn.init.constant_(self.eps, EPSSigma);

        
    def forward(self, x, xRes=0):
        
        ''' 
        Input: 
            x: (batch_size x in_channels x 2*B x 2*B) input float tensor
            xRes: (batch_size x channels x 2*B x 2*B) residual connection float tensor, defaults to zero
        
        Output:
            xOut: (batch_size x out_channels x 2*B x 2*B) normalized float tensor
            
        Both the input and output are signals on the Riemann sphere with values on a 2*B x 2*B (theta x phi)
        Driscoll-Healy spherical grid (see the documentation of TS2Kit and the gridDH function in ts2kit.py)
        '''
        b, C, B = x.size()[0], self.channels, self.B
        
        dxNorm = torch.sqrt(self.dirichlet(x) + realSgn(self.eps) + EPSSigma);
        
        ## Normalization
        xN = x * torch.reciprocal(dxNorm)[:, :, None, None] * self.scale[..., None, None] + self.bias[..., None, None];
        
        ## Thresholded nonlinearity
        return F.mish( xN + xRes - self.th[..., None, None]) + self.th[..., None, None];
      
        
