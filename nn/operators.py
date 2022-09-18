import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import os

from utils.rs_diff import rsOrder02
from utils.utils import isOrigin

## Bounds on log-magnitudes of a^2 and an terms
dilBound = 3.0;
transMin = np.log(0.075); # Don't change this one unless you really, really, really know what you're doing
transMax = 3.0;

EPSFrame = 1e-3

########################################################
### Module for computing frame and density operators ###
########################################################


class operators(nn.Module):
   
    def __init__(self, B, channels):
        super(operators, self).__init__()
        
        self.B = B
        
        self.diff = rsOrder02(B)
        
        ## Parameters for learnable frame offsets
        self.df = torch.nn.Parameter(torch.Tensor(1, channels, 2));

        self.d2f = torch.nn.Parameter(torch.Tensor(1, channels, 2));

        torch.nn.init.zeros_(self.df);
        torch.nn.init.zeros_(self.d2f);
                            
    def forward(self, x):
        
        #x: b x C x theta x phi
        b, C, B = x.size()[0], x.size()[1], self.B
        
        ## Set up tensors holding log of frame operator elements
        lnAlpha = torch.empty(b, C, 2*B, 2*B, device=x.device).float().fill_(0.0);
        
        Phi = torch.empty(b, C, 2*B, 2*B, device=x.device).float().fill_(0);
        
        lnTau = torch.log(torch.empty(b, C, 2*B, 2*B, device=x.device).float().fill_(EPSFrame));
         
        Psi = torch.empty(b, C, 2*B, 2*B, device=x.device).float().fill_(0);
        
        ## Compute differential and Hessian at origin
        dx0, d2x0 = self.diff(x);
         
        dx0 = torch.reshape(dx0, (b, C, 2*B, 2*B));
        
        d2x0 = torch.reshape(d2x0, (b, C, 2*B, 2*B));
                        
        vInd = torch.nonzero(torch.logical_not(isOrigin(dx0, EPSFrame)));
        
        df = torch.exp(torch.view_as_complex(self.df));
        d2f = torch.view_as_complex(self.d2f);
       
        ## a^2 term encoding
        a2 = df[0, vInd[:, 1]] / dx0[vInd[:, 0], vInd[:, 1], vInd[:, 2], vInd[:, 3]];
        
        an = 0.5 * (d2x0[vInd[:, 0], vInd[:, 1], vInd[:, 2], vInd[:, 3]] - torch.reciprocal(a2 * a2) * d2f[0, vInd[:, 1]]) * df[0, vInd[:, 1]] * torch.reciprocal( dx0[vInd[:, 0], vInd[:, 1], vInd[:, 2], vInd[:, 3]] * dx0[vInd[:, 0], vInd[:, 1], vInd[:, 2], vInd[:, 3]]);
        
        lnAlpha[vInd[:, 0], vInd[:, 1], vInd[:, 2], vInd[:, 3]] = F.hardtanh(torch.log(torch.abs(a2)), min_val = -1.0*dilBound, max_val = dilBound);
        
        Phi[vInd[:, 0], vInd[:, 1], vInd[:, 2], vInd[:, 3]] = torch.angle(a2);

        ## a * n term encoding
        transInd = torch.nonzero(torch.logical_not(isOrigin(an, EPSFrame)));
        
        tInd = vInd[transInd[:, 0], ...];
        
        lnTau[tInd[:, 0], tInd[:, 1], tInd[:, 2], tInd[:, 3]] = F.hardtanh(torch.log(torch.abs(an[transInd[:, 0]])), min_val = transMin, max_val = transMax);
        
        Psi[tInd[:, 0], tInd[:, 1], tInd[:, 2], tInd[:, 3]] = torch.angle(an[transInd[:, 0]]);
        
        ## Density operator
        h = dx0.real * dx0.real + dx0.imag * dx0.imag;
        
        return h, lnAlpha, Phi, lnTau, Psi;    
    

   
        