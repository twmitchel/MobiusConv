import torch
import torch.nn as nn
import torch.nn.functional as tFn
from math import pi as PI
import os

import numpy as np
import scipy

from cache.cache import cacheDir, quadDir

from TS2Kit.ts2kit import FTSHT, ITSHT

from utils.representation import logPolarFilters, logPolarFiltersLT, rsConvCoeffLT, linearizeLogPolar, mellinQuadrature, t, sigma1, sigma2

from utils.rs_diff import rsDirichlet
from nn import operators

#############################################################
###                    Utility modules                    ###
#############################################################


## The reduction (related to Equation (19) in the paper). Can create big intermediate tensors on GPU - if you're getting memory errors, this is likely the culprit. Scripting helps, but implementing this in JAX could potentially make this much less expensive
@torch.jit.script
def reduceBasesLTConv(x, lnAlpha, Phi, lnTau, Psi, pAlpha, pPhi, pTau, pPsi, F, W, b: int, B: int, Q: int, M: int, D1: int,  D2: int, iC: int):
    
    return  torch.matmul(torch.reshape( torch.mul(torch.mul(torch.mul(x[:,  None, None, ...], torch.exp(lnAlpha[:, None, None, ...] * pAlpha[None, :, :, None, None, None] + Phi[:, None, None, ...] * pPhi[None, :, :, None, None, None]))[..., None, None], torch.exp( lnTau[:, None, None, :, :, :, None, None] * pTau[None, :, :, None, None, None, ...] + Psi[:, None, None, :, :, :, None, None] * pPsi[None, :, :, None, None, None, ...])[:, ...]), F[None, :, :, None, None, None, ...]), ( b, Q, 2*(M+1), 2*B, 2*B, iC * (2*D1 + 1) * (2*D2+1)) ), W); 


def collectWeights(coeffs):
    
    iC, nCoeffs, oC = coeffs.size()[0], coeffs.size()[1], coeffs.size()[2]
        
    wZ = torch.complex(coeffs[:, 0, :], torch.zeros_like(coeffs[:, 0, :]));
    
    wS = torch.permute(torch.reshape(coeffs[:, 1:, :], (iC, (nCoeffs - 1)//2, 2, oC)), (0, 1, 3, 2)).contiguous()
    
    w = torch.complex(wS[..., 0], wS[..., 1]);
        
    return torch.reshape(torch.cat( (torch.flip(w, [1]), wZ[:, None, :], w), dim=1), (-1, oC));

   
class convBlockReduceLT(nn.Module):
    
    def __init__(self, B, M, Q, sigma=None, S=None):
        super(convBlockReduceLT, self).__init__()
        
        self.B = B;
        self.M = M;
        self.Q = Q;
        
        self.FST = FTSHT(B).float();
        self.IST = ITSHT(B).float();
        
        self.register_buffer('delta', rsConvCoeffLT(B, M));
        
        ## Create or load filter conv coefficent matrix
        fName = cacheDir + '/mobFilterCoeffs_{}_{}_{}'.format(B, M, Q) + '.pt';

        if (os.path.isfile(fName) == False):
                        
            F = logPolarFiltersLT(B, M, Q, sigma, S);
           
            Flm = self.FST(torch.reshape(F, ( Q*2*(M+1), 2*B, 2*B)));
            
            Flm = torch.reshape( Flm, (Q, 2*(M+1), (2*B-1), B) );
                                       
            Fm = torch.empty(Q, 2*(M+1), B).fill_(0).cfloat();
    
            for m in range(-M, M+2):

                if (m < (M + 1)):
                    mInd = m;
                    oInd = m + M;
                else:
                    mInd = 0;
                    oInd = 2*M + 1;

                for q in range(Q):
                    Fm[q, oInd, :] = Flm[q, oInd, (-mInd) + (B-1), :];
    
            torch.save(Fm, fName);
            
            print("Computed filter coeffs B = {}".format(B), flush=True);
        else:
            
            Fm = torch.load(fName);
             
        self.register_buffer('Fm', Fm);
        
    
    def forward(self, g):
        
        # g: b x Q x 2*(M+1)  X  theta x phi
        
        b, B, M, Q = g.size()[0], self.B, self.M, self.Q;
        
        # Forward SHT
        # b * 2 * (Q + 1) * P x m*l
        g = self.FST(torch.reshape(g, (b*Q*(2*(M + 1)), 2*B, 2*B)))
        
        ## Reshape and sparse matmul by conv coeffs
        # b * Q x (2*(M + 1)) * m * l
        g = torch.transpose(torch.mm(self.delta, torch.transpose(torch.reshape(g, (b*Q, 2*(M+1)*(2*B - 1)*B)), 0, 1)), 0, 1);
        
        ## Reshape, multiply by appropriate scale, then reduce
        # g: b x m x l
        g = torch.sum( torch.mul(torch.reshape(g, (b, Q, 2*(M + 1),  2*B - 1, B)), self.Fm[None, :, :, None, :]), dim=(1, 2));
                      
        # Inverse SHT and return
        return self.IST(g)
   
  
class linearBasisConvLT(nn.Module):
    
    def __init__(self, in_channels, out_channels, B, D1, D2, M=2, Q=30, sigma=None, S=None, W=None):
        super(linearBasisConvLT, self).__init__()
        
        self.B = B;
        self.D1 = D1;
        self.D2 = D2;
        self.M = M;
        self.Q = Q;
        self.iC = in_channels
        self.oC = out_channels
                        
        self.T = operators(B, in_channels);
        
        E = rsDirichlet(B).float()
        
        FW = torch.sqrt(torch.reshape(E(torch.reshape(logPolarFilters(B, D1, D2), ((2*D1 + 1)*(2*D2 + 1), 2*B, 2*B) )[None, ...]), ( 2*D1 + 1, 2*D2 + 1))); 
        
        F = linearizeLogPolar(B, D1, D2, M, Q, sigma=sigma, S=S, W=W);
        
        self.register_buffer('F', F / FW[None, None, ...]);
        
        pPhi = torch.empty(1, 2*(M+1)).fill_(0).cfloat();
        pAlpha = torch.empty(Q, 2*(M + 1)).fill_(0).cfloat();
        
        pPsi = torch.empty(1, 2*(M + 1), 1, 2*D2 + 1).fill_(0).cfloat();
        pTau = torch.empty(Q, 2*(M + 1), 2*D1 + 1, 1).fill_(0).cfloat();
        
        for m in range(-M, M+2):
        
            if (m < (M + 1)):
                
                if (sigma is not None):
                    sig = sigma[m+M];
                else:
                    sig = sigma1;
                
                mInd = m;
                oInd = m + M;
                
            else:
                
                if (sigma is not None):
                    sig = sigma[m + M];
                else:
                    sig = sigma2;
                    
                mInd = 0;
                oInd = 2*M + 1;
            
            if (S is not None):
                s = S[m + M, :];
            else:
                s, _ = mellinQuadrature(Q, mInd);

            pPhi[0, oInd] = -1.0*1j*mInd;

            for v in range(-D2, D2+1):

                pPsi[0, oInd, 0, v+D2] = -1.0*1j*(v - mInd);


            for q in range(Q):

                pAlpha[q, oInd] = -sig - 1j*s[q];

                for u in range(-D1, D1+1):

                    pTau[q, oInd, u, 0] = t + sig + 1j*(s[q] - u);

        self.register_buffer('pPhi', pPhi);
        self.register_buffer('pPsi', pPsi);
        
        self.register_buffer('pAlpha', pAlpha);
        self.register_buffer('pTau', pTau);
        
        # Number of real coeffs
        nCoeffs = (2*D1 + 1)*(2*D2 + 1);
        
        self.coeffs = torch.nn.Parameter(torch.Tensor(in_channels, nCoeffs, out_channels));
        
        torch.nn.init.xavier_uniform_(self.coeffs); 
        
        
    
    def forward(self, x):

        b, iC, oC, Q, M, B, D1, D2 = x.size()[0], x.size()[1], self.oC, self.Q, self.M, self.B, self.D1, self.D2
        
        # x: b x iC x theta x phi 
        
        W = collectWeights(self.coeffs)
        
        # Compute transformation and density operators
        h, lnAlpha, Phi, lnTau, Psi = self.T(x);
           
        xR = torch.permute(h, (0, 2, 3, 1));
               
        lnAlpha = torch.permute(lnAlpha, (0, 2, 3, 1));
        Phi = torch.permute(Phi, (0, 2, 3, 1));
        lnTau = torch.permute(lnTau, (0, 2, 3, 1));
        Psi = torch.permute(Psi, (0, 2, 3, 1));  
               
        # Fused reduction  
        xR = reduceBasesLTConv(xR, lnAlpha, Phi, lnTau, Psi, self.pAlpha, self.pPhi, self.pTau, self.pPsi, self.F, W, b, B, Q, M, D1, D2, iC)

        return torch.permute(xR, (0, 5, 1, 2, 3, 4));

###############################################################
###             Mobius Convolution via the SHT              ###
###############################################################

class MobiusConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, B, D1, D2, M=2, Q=30, optQuad=False):
        super(MobiusConv, self).__init__()
        
        '''
        Inputs:
            in_channels: # of input channels
            
            out_channels: # of output channels
            
            B: Spherical Harmonic Bandlimit
            
            D1: Radial band-limit of log-polar filters
            
            D2: Angular band-limt of log-polar filters
            
            M: Angular band-limit of representation
            
            Q: # of radial quadrature samples for radial component of representation
            
            optQuad: Flag to use the learned quadrature samples, will default to hueristic values if the former is not available

        '''
        
        self.B = B;
        self.D1 = D1;
        self.D2 = D2;
        self.M = M;
        self.Q = Q;
        self.oC = out_channels
        self.iC = in_channels
        
        sigma = None;
        S = None;
        W = None;
        
        if (optQuad == True):
            
            qID = '{}_{}_{}_{}_{}.pt'.format(D1, D2, M, Q, t)

            sigFile = os.path.join(quadDir, 'sig_' + qID)
            stenFile = os.path.join(quadDir, 'sten_'+qID)
            weightFile = os.path.join(quadDir, 'w_'+qID)
            
            if (os.path.isfile(sigFile) and os.path.isfile(stenFile) and os.path.isfile(weightFile)):
                
                sigma = torch.load(sigFile).detach().cpu()
                S = torch.load(stenFile).detach().cpu()
                W = torch.load(weightFile).detach().cpu()
                                
            else:
                print('Quadrature files do not exist in directory, reverting to default...', flush=True);
                
        self.linearize = linearBasisConvLT(in_channels, out_channels, B, D1, D2, M, Q, sigma, S, W);

        self.convSum = convBlockReduceLT(B, M, Q, sigma, S)
        
        
        
    def forward(self, x):
        
        ''' 
        Input: 
            x: (batch_size x in_channels x 2*B x 2*B) float tensor
        
        Output:
            xOut: (batch_size x out_channels x 2*B x 2*B) float tensor
            
        Both the input and output are signals on the Riemann sphere with values on a 2*B x 2*B (theta x phi)
        Driscoll-Healy spherical grid (see the documentation of TS2Kit and the gridDH function in ts2kit.py)
        '''
        b, oC, B, M, Q = x.size()[0], self.oC, self.B, self.M, self.Q
                
        xC = self.linearize(x)
        
        xC = self.convSum( torch.reshape(xC, (b*oC, Q, 2*(M+1), 2*B, 2*B)) ).real;
        
        return torch.reshape(xC, (b, oC, 2*B, 2*B)) 
        

        
