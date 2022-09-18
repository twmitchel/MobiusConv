import torch
import torch.nn as nn
import torch.nn.functional as F
from nn import MCResNetBlock, imLinear, FRDirichlet

######################################################
### An example of a UNet awith Mobius Convolutions ###
######################################################

## Down block -- pooling followed by an MCResNet block
class MCDown(nn.Module):

    def __init__(self, in_channels, out_channels, B, D1=1, D2=1, M=2, Q=30, checkpoint=True):
        super(MCDown).__init__()
        
        '''
        Inputs:
            in_channels: # of input channels
            
            out_channels: # of output channels
            
            B: Input spherical harmonic bandlimit, must be even
            
            D1: Radial band-limit of log-polar filters
            
            D2: Angular band-limt of log-polar filters
            
            M: Angular band-limit of representation
            
            Q: # of radial quadrature samples for radial component of representation
                                    
            checkpoint: Flag to use checkpoining (trade computational speed for less memory overhead)
        '''
        
        assert B % 2 == 0
        
        self.pool = torch.nn.AdaptiveMaxPool2d( (B+2, B) );
        
        self.conv = MCResNetBlock(in_channels, out_channels, B // 2, D1=D1, D2=D2, M=M, Q=Q, checkpoint=checkpoint);
        
        

    def forward(self, x):
        
        ''' 
        Input: 
            x: (batch_size x in_channels x 2*B x 2*B) float tensor
        
        Output:
            xOut: (batch_size x out_channels x B x B) float tensor
            
        '''
        
        return self.conv(self.pool(x));

    
## Up block -- upsampling followed by concat and convolution
class MCUp(nn.Module):
    
    def __init__(self, in_channels, out_channels, B, D1=1, D2=1, M=2, Q=30, checkpoint=True):
        super(MCUp).__init__()
        
        '''
        Inputs:
            in_channels: # of input channels
            
            out_channels: # of output channels
            
            B: Input spherical harmonic bandlimit
            
            D1: Radial band-limit of log-polar filters
            
            D2: Angular band-limt of log-polar filters
            
            M: Angular band-limit of representation
            
            Q: # of radial quadrature samples for radial component of representation
                                    
            checkpoint: Flag to use checkpoining (trade computational speed for less memory overhead)
        '''
        
        self.unpool = torch.nn.Upsample(size=(4*B, 4*B), mode='bilinear', align_corners=False)
        
        self.conv = MCResNetBlock(in_channels, out_channels, 2*B, D1=D1, D2=D2, M=M, Q=Q, checkpoint=checkpoint);


        
    def forward(self, x, xRes):
        
        ''' 
        Input: 
            x: (batch_size x in_channels / 2 x 2*B x 2*B) low-res input float tensor 
            xRes: (batch_size x in_channels / 2 x 4*B x 4*B) high-res residual float tensor
        
        Output:
            xOut: (batch_size x out_channels x 4*B x 4*B) float tensor

        '''
        
        return self.conv( torch.cat( (self.unpool(x), xRes), dim=1 ) );
        

class MCDownsample(nn.Module):
    
    def __init__(self, channels, B, nLayers, D1=1, D2=1, M=2, Q=30, checkpoint=True):
        super(MCDownsample).__init__()
        
        '''
        Inputs:
            in_dim: # of input channel dimensions
            
            channels: # of channels at the highest resolution
            
            B: Input spherical harmonic bandlimit, must be even and divisible by 2^(nLayers - 1)
            
            nLayers: Number of layers in the inverted pyramid, must be such that  2*B0 *  2^(1 - nLayers) >= 8
            
            D1: Radial band-limit of log-polar filters
            
            D2: Angular band-limt of log-polar filters
            
            M: Angular band-limit of representation
            
            Q: # of radial quadrature samples for radial component of representation
                                    
            checkpoint: Flag to use checkpointing (trade computational speed for less memory overhead)
        '''
        
        assert B % 2 == 0
        assert 2*B / (2 ** (nLayers - 1) ) >= 8 and 2*B % (2**(nLayers - 1))== 0
                                                           
                                                           
        ML = [];
        
        ML.append( MCResNetBlock(channels, channels, B, D1=D1, D2=D2, M=M, Q=Q, checkpoint=checkpoint) ) 
                
        for l in range(nLayers-1):
            
            factor = 2 ** l
            
            CIN = channels * factor                
            COUT = channels * factor * 2
            
            BL = B // factor;
                        
            if (l == nLayers-2):
                    COUT = COUT // 2;
                 
            ML.append(MCDown(CIN, COUT, BL, D1=D1, D2=D2, M=M, Q=Q, checkpoint=checkpoint));
        
        self.layers = torch.nn.ModuleList(ML);
        self.nLayers = nLayers
        

        
    def forward(self, x):
        
        ''' 
        Input: 
            x: (batch_size x channels x 2*B0 x 2*B0) input float tensor 
        
        Output:
            h: (batch_size x channels * 2^(nLayers - 1) x  2*B / (2^(nLayers - 1) x 2*B / (2^(nLayers-1) ) float tensor
            hL: list of outputs of each layer
        '''
        
        hL = [];
        h = self.layers[0](x);
                  
        for l in range(1, self.nLayers):
            hL.append(h);
            h = self.layers[l](h);
        
        return h, hL;

    
class MCUpsample(nn.Module):
    
    def __init__(self, channels, B, nLayers, D1=1, D2=1, M=2, Q=30, checkpoint=True):
        super(MCUpsample).__init__()
        
        '''
        Inputs:
            in_dim: # of input channel dimensions
            
            channels: # of channels at the lowest resolution
            
            B: Input spherical harmonic bandlimit
            
            nLayers: Number of layers in the pyramid, must be such that  channels %  2^(1 - nLayers) == 0
            
            D1: Radial band-limit of log-polar filters
            
            D2: Angular band-limt of log-polar filters
            
            M: Angular band-limit of representation
            
            Q: # of radial quadrature samples for radial component of representation
                                    
            checkpoint: Flag to use checkpointing (trade computational speed for less memory overhead)
        '''
        
        ML = [];
        
        
        for l in range(nLayers-1):
            
            factor = 2 ** l
            
            CIN = channels // factor                
            COUT = channels // (factor * 2)
            
            B = B0 * factor
                        
            if (l < nLayers-2):
                    COUT = COUT // 2;
                 
            ML.append(MCUp(CIN, COUT, B, D1=D1, D2=D2, M=M, Q=Q, checkpoint=checkpoint));
                  
        
        self.layers = torch.nn.ModuleList(ML);
        self.nLayers = nLayers
        

        
    def forward(self, x, xL):
        
        ''' 
        Input: 
            x: (batch_size x channels x 2*B x 2*B) input float tensor 
            xL: list of outputs of each downsampling layer

        Output:
            h: (batch_size x channels // 2^(nLayers - 1) x  2*B * (2^(nLayers - 1) x 2*B * (2^(nLayers-1) ) float tensor
        '''
        
        for l in range(self.nLayers-1):
            
            x = self.layers[l](x, xL.pop());

        
        return x;    
    

            
class MCUNet(nn.Module):
    
    def __init__(self, in_dim, channels, B, nLayers, D1=1, D2=1, M=2, Q=30, checkpoint=True):
        super(MCUNet).__init__()
        
        
        '''
        Inputs:
            in_dim: # of input channel dimensions
            
            channels: # of channels at the lowest resolution
            
            B: Input spherical harmonic bandlimit
            
            nLayers: Number of layers in the pyramid, must be such that  channels %  2^(1 - nLayers) == 0
            
            D1: Radial band-limit of log-polar filters
            
            D2: Angular band-limt of log-polar filters
            
            M: Angular band-limit of representation
            
            Q: # of radial quadrature samples for radial component of representation
                                    
            checkpoint: Flag to use checkpointing (trade computational speed for less memory overhead)
        '''
        
        self.inLayer = torch.nn.Sequential(imLinear(in_dim, channels), nn.Mish());
        
        self.down = MCDownsample(channels, B, nLayers, D1=D1, D2=D2, M=M, Q=Q, checkpoint=checkpoint);
        
        self.up = MCUpsample( channels * (2** (nLayers - 1) ), B // (2 ** (nLayers-1) ), nLayers, D1=D1, D2=D2, M=M, Q=Q, checkpoint=checkpoint);
        
    
    def forward(self, x):
        ''' 
        Input: 
            x: (batch_size x in_dim x 2*B x 2*B) input float tensor 

        Output:
            xOut: (batch_size x channels x  2*B x 2*B ) float tensor
        '''
        
        x = self.inLayer(x);
        
        x, xL = self.down(x);
        
        return self.up(x, xL);
        
        
        
    
                                               
                                                           


        
    