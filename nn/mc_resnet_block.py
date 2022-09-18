import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as tcp
from nn import MobiusConv, FRDirichlet, imLinear

################################################
###     Mobius Convolution ResNet block      ###
################################################

class MCResNetBlock(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, B, D1=1, D2=1, M=2, Q=30, mid_channels=None, checkpoint=False):
        super(MCResNetBlock, self).__init__()
        
        '''
        Inputs:
            in_channels: # of input channels
            
            out_channels: # of output channels
            
            B: Spherical harmonic bandlimit
            
            D1: Radial band-limit of log-polar filters
            
            D2: Angular band-limt of log-polar filters
            
            M: Angular band-limit of representation
            
            Q: # of radial quadrature samples for radial component of representation
            
            mid_channels: # of intermediate channels (default = out_channels)
            
            checkpoint: Flag to use checkpoining (trade computational speed for less memory overhead)
        '''
        
               
        iC1 = in_channels
        oC2 = out_channels;
        
        if (mid_channels is None):
            oC1 = out_channels;
            iC2 = out_channels;
        else:
            oC1 = mid_channels;
            iC2 = mid_channels;
        
        # Convolution blocks
        self.conv1 = MobiusConv(iC1, oC1, B, D1, D2, M, Q);
        self.conv2 = MobiusConv(iC2, oC2, B, D1, D2, M, Q);

        # Normalization blocks
        self.FR1 = FRDirichlet(B, oC1);
        self.FR2 = FRDirichlet(B, oC2);
        
        # Residual connection
        if (in_channels == out_channels):
            self.res = torch.nn.Identity();
        else:
            self.res = imLinear(in_channels, out_channels, bias=False);
            
        if checkpoint:
            self.wrapper = tcp.checkpoint
        else:
            self.wrapper = lambda f, x: f(x);
    
    def _forward(self, x):
        
        # Mobius Convolution        
        x_conv = self.conv1(x);
        
        # Normalization + Nonlinearity     
        x_conv = self.FR1(x_conv);
        
        # Mobius Convolution 
        x_conv = self.conv2(x_conv);
        
        # Normalization + Nonlinearity w/ residiual connection
        xOut = self.FR2(x_conv, self.res(x))
        
        return xOut;
    
    def forward(self, x):
        
        ''' 
        Input: 
            x: (batch_size x in_channels x 2*B x 2*B) float tensor
        
        Output:
            xOut: (batch_size x out_channels x 2*B x 2*B) float tensor
            
        Both the input and output are signals on the Riemann sphere with values on a 2*B x 2*B
        Driscoll-Healy spherical grid (see the documentation of TS2Kit and the gridDH function in ts2kit.py)
        '''
        
        return self.wrapper(self._forward, x)

       
  
