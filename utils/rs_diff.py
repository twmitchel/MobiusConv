import os
import os.path as osp
import glob
import numpy as np
import torch
import torch.nn as nn
from math import pi as PI

from cache import cacheDir
from TS2Kit import gridDH, FTSHT, ITSHT
sm0 = 1e-5;

################################################################
### Utilities for computing derivatives and other quantities ###
### on the Riemann Sphere via the SHT                        ###
################################################################


### Module for computing the differential of functions
### on RS in local frames
class rsDiff01(nn.Module):
    
    def __init__(self, B):
        super(rsDiff01, self).__init__()
        
        self.B = B;
        
        self.FSHT = FTSHT(B).float();
        self.ISHT = ITSHT(B).float();
        
        fName = cacheDir + '/diffCoeffs_{}'.format(B) + '.pt';
        
        if (os.path.isfile(fName) == False):
            
            c = torch.empty(2*B - 1, B, 3).fill_(0).cfloat();
            
            for m in range(-(B-1), B):
                
                for l in range(np.absolute(m), B):
                    
                    c[m+(B-1), l, 0] = 1j*m;
                                        
                    c[m+(B-1), l, 1] = l * np.sqrt( ( (l+1) * (l+1) - m * m) / ( (2*l+1)*(2*l+3)));
                    
                    c[m+(B-1), l, 2] = -1.0*(l+1)*np.sqrt( ( l*l - m * m) / ( (2*l + 1) * (2*l - 1)));
                                
            torch.save(c, fName);
            
        else:
            c = torch.load(fName);
            
        nullMask = torch.empty(2*B-1, B).fill_(0).long();
        
        for l in range(B):
            for m in range(-(B-1), B):

                if (np.absolute(m) > l):
                    
                    nullMask[m+(B-1), l] = 1;
                    
             
        self.register_buffer('c', c);
        self.register_buffer('nullMask', nullMask);
        
        theta, phi = gridDH(B);
                
        zInvH = 0.5*torch.polar(torch.reciprocal(torch.tan(0.5*phi)), theta);
                  
        self.sD1dT = 1j;
        self.sD1dP = 1.0;
        
        self.register_buffer('zInvH', zInvH);
        


    def forward(self, f):
        
        B, b, nInd = self.B, f.size()[0], self.nullMask;
        
        flm = self.FSHT(f)
                
        # dTheta
        dT = self.ISHT(torch.mul(flm, self.c[None, :, :, 0])).real;       
        
        # dPhi * sin(phi)
        fAlpha = torch.zeros_like(flm);
        fBeta = torch.zeros_like(flm);
        
        fAlpha[:, :, 1:] = flm[..., :(B-1)] * self.c[None, :, :(B-1), 1];
        fBeta[:, :, :(B-1)] = flm[..., 1:] * self.c[None, :, 1:, 2];
        
        fAlpha[:, nInd[:, 0], nInd[:, 1]] = torch.zeros_like(fAlpha[:, nInd[:, 0], nInd[:, 1]]);
        fBeta[:, nInd[:, 0], nInd[:, 1]] = torch.zeros_like( fBeta[:, nInd[:, 0], nInd[:, 1]]);
        
        dP = self.ISHT(fAlpha + fBeta).real;
        
        
        d1f = (self.sD1dT * dT + self.sD1dP * dP) * self.zInvH[None, ...];


        return d1f

    
    
    
### Module for computing the differential and hessian of functions
### on RS in local frames
class rsDiff02(nn.Module):
    
    def __init__(self, B):
        super(rsDiff02, self).__init__()
        
        self.B = B;
        
        self.FSHT = FTSHT(B).float();
        self.ISHT = ITSHT(B).float();
        
        fName = cacheDir + '/diffCoeffsO2_{}'.format(B) + '.pt';
        
        if (os.path.isfile(fName) == False):
            
            c = torch.empty(2*B - 1, B, 4).fill_(0).cfloat();
            
            for m in range(-(B-1), B):
                
                for l in range(np.absolute(m), B):
                    
                    c[m+(B-1), l, 0] = 1j*m;
                    
                    c[m+(B-1), l, 1] = -1.0 * m * m;
                    
                    c[m+(B-1), l, 2] = l * np.sqrt( ( (l+1) * (l+1) - m * m) / ( (2*l+1)*(2*l+3)));
                    
                    c[m+(B-1), l, 3] = -1.0*(l+1)*np.sqrt( ( l*l - m * m) / ( (2*l + 1) * (2*l - 1)));
                    
                         
            
            torch.save(c, fName);
            
        else:
            c = torch.load(fName);
            
        nullMask = torch.empty(2*B-1, B).fill_(0).long();
        
        for l in range(B):
            for m in range(-(B-1), B):

                if (np.absolute(m) > l):
                    
                    nullMask[m+(B-1), l] = 1;
                    
             
        self.register_buffer('c', c);
        self.register_buffer('nullMask', nullMask);
        
        theta, phi = gridDH(B);
                
        zInvH = 0.5*torch.polar(torch.reciprocal(torch.tan(0.5*phi)), theta);
        
        zInvH2 = 0.5*zInvH*zInvH;
          
        self.sD1dT = 1j;
        self.sD1dP = 1.0;
        
        self.sD2dT = -4.0*1j;
        self.sD2d2T = -2.0;
        
        sD2dP = (torch.sin(2.0*phi) - 4.0*torch.sin(phi))/torch.sin(phi);
        sD2d2P = (1.0 - torch.cos(2.0*phi))/(torch.sin(phi) * torch.sin(phi));
        self.sD2dTdP = 4.0 * 1j;
        

        self.register_buffer('zInvH', zInvH);
        self.register_buffer('zInvH2', zInvH2);
                
        self.register_buffer('sD2dP', sD2dP);
        self.register_buffer('sD2d2P', sD2d2P);
        



    def forward(self, f):
        
        B, b, nInd = self.B, f.size()[0], self.nullMask;
        
        
        flm = self.FSHT(f)
               
        
        # dTheta
        dT = torch.mul(flm, self.c[None, :, :, 0]);
        
        #d2Theta
        d2T = torch.mul(flm, self.c[None, :, :, 1]);
        
        
        # dPhi * sin(phi)
        
        fAlpha = torch.zeros_like(flm);
        fBeta = torch.zeros_like(flm);
        
        fAlphaT = torch.zeros_like(dT);
        fBetaT = torch.zeros_like(dT);
        
        fAlpha2 = torch.zeros_like(flm);
        fBeta2 = torch.zeros_like(flm);
        
        fAlpha[:, :, 1:] = flm[..., :(B-1)] * self.c[None, :, :(B-1), 2];
        fBeta[:, :, :(B-1)] = flm[..., 1:] * self.c[None, :, 1:, 3];
               
        fAlpha[:, nInd[:, 0], nInd[:, 1]] = torch.zeros_like(fAlpha[:, nInd[:, 0], nInd[:, 1]]);
        fBeta[:, nInd[:, 0], nInd[:, 1]] = torch.zeros_like( fBeta[:, nInd[:, 0], nInd[:, 1]]);
        
        fAlphaT[:, :, 1:] = dT[..., :(B-1)] * self.c[None, :, :(B-1), 2];
        fBetaT[:, :, :(B-1)] = dT[..., 1:] * self.c[None, :, 1:, 3];
        
        
        fAlphaT[:, nInd[:, 0], nInd[:, 1]] = torch.zeros_like(fAlphaT[:, nInd[:, 0], nInd[:, 1]]);
        fBetaT[:, nInd[:, 0], nInd[:, 1]] = torch.zeros_like( fBetaT[:, nInd[:, 0], nInd[:, 1]]);
        
        
        dP = fAlpha + fBeta;
        
        dPdT = fAlphaT + fBetaT;
        
        # d2Phi * sin(phi^2)
        fAlpha2[:, :, 1:] = dP[..., :(B-1)] * self.c[None, :, :(B-1), 2];
        fBeta2[:, :, :(B-1)] = dP[..., 1:] * self.c[None, :, 1:, 3];
        
        fAlpha2[:, nInd[:, 0], nInd[:, 1]] = torch.zeros_like(fAlpha2[:, nInd[:, 0], nInd[:, 1]]);
        fBeta2[:, nInd[:, 0], nInd[:, 1]] = torch.zeros_like( fBeta2[:, nInd[:, 0], nInd[:, 1]]);
        
        d2P =  0.5*(fAlpha2 + fBeta2)
                
        # Invert  
        dT, d2T, dP, d2P, dPdT = torch.split( self.ISHT(torch.cat( (dT, d2T, dP, d2P, dPdT), dim=0)).real,b, dim=0);
        
        d1f = (self.sD1dT * dT + self.sD1dP * dP) * self.zInvH[None, ...];
        
        d2f = (self.sD2dT * dT + self.sD2d2T * d2T + self.sD2dP[None, ...] * dP + self.sD2d2P[None, ...] * d2P + self.sD2dTdP * dPdT) * self.zInvH2[None, ...];

        return d1f, d2f


## Module for computing differentials in frame at origin via the chain rule
class rsOrder01(nn.Module):
    
    def __init__(self, B):
        super(rsOrder01, self).__init__()
        
        self.B = B;
        
        theta, phi = gridDH(B);
        
        theta = theta.float();
        phi = phi.float();
        
        self.diff = rsDiff01(B);

        transP = torch.exp(-1j*theta) / ( torch.cos(phi / 2.0) * torch.cos(phi / 2.0) );
                
        self.register_buffer('transP', transP);        
            
        
    def forward(self, x):
        
        b, C, B = x.size()[0], x.size()[1], self.B
        
        x0 = torch.reshape(x, ( b * C,  2*B, 2*B) );
                
        dx = self.diff(x0)
        
        dx0 = dx * self.transP[None, ...];
                
        return torch.reshape(dx0, (b, C, 2*B, 2*B)).cfloat()


## Module for computing differentials and hessians in frame at origin via the chain rule
class rsOrder02(nn.Module):
    
    def __init__(self, B):
        super(rsOrder02, self).__init__()
        
        self.B = B;
        
        theta, phi = gridDH(B);
        
        theta = theta.float()
        phi = phi.float()
        
        self.diff = rsDiff02(B);

        transP = torch.exp(-1j*theta) / ( torch.cos(phi / 2.0) * torch.cos(phi / 2.0) );
        
        transPP = 2.0 * torch.tan(phi / 2.0) * transP;
        
        self.register_buffer('transP', transP);
        self.register_buffer('transPP', transPP);
                        
        
    def forward(self, x):
        
        b, C, B = x.size()[0], x.size()[1], self.B
        
        x0 = torch.reshape(x, ( b * C,  2*B, 2*B) );
                
        dx, d2x = self.diff(x0)
        
        dx0 = dx * self.transP[None, ...];
        
        d2x0 = d2x * self.transP[None, ...] * self.transP[None, ...] + dx * self.transPP[None, ...];
        
        return torch.reshape(dx0, (b, C, 2*B, 2*B)).cfloat(), torch.reshape(d2x0, (b, C, 2*B, 2*B)).cfloat(); 
    
    
## L^2 norm of a scalar-valued function on the Riemann sphere
class rsNorm2(nn.Module):
    
    def __init__(self, B):
        super(rsNorm2, self).__init__()
        
        self.B = B;
        
        self.FSHT = FTSHT(B).float();

    def forward(self, f):
                    
        flm = self.FSHT(f);
                
        return torch.sum( ( flm.real * flm.real + flm.imag * flm.imag), dim=(1, 2));

    
## Module for computing the Dirichlet energy of a function on RS
class rsDirichlet(nn.Module):
    
    def __init__(self, B):
        super(rsDirichlet, self).__init__()
        
        self.B = B;
        
        self.diff = rsOrder01(B);
        
        self.norm2 = rsNorm2(B).float();
         
        
    def forward(self, x):
        
        b, C, B  = x.size()[0], x.size()[1], self.B
        
        dx = self.diff(x);
        
        return torch.reshape(self.norm2(torch.reshape(dx, (b*C, 2*B, 2*B))), (b, C));


 ## Smooth scalar functions on the RS
class rsSmooth(nn.Module):
    
    def __init__(self, B, channels, smFact=None):
        super(rsSmooth, self).__init__()
        
        self.B = B;
        
        self.SHT = FTSHT(B).float();
        self.ISHT = ITSHT(B).float();
        
        lap = -1.0 * torch.arange(B) * (torch.arange(B) + 1);

        self.register_buffer('lap', lap);
        
        if (smFact is None):
            self.smooth = torch.nn.Parameter(torch.Tensor(1, channels));      
            torch.nn.init.constant_(self.smooth, np.log(sm0));
        else:
            self.register_buffer('smooth', np.log(smFact)*torch.ones(1, channels).float());
            
            
           
    def forward(self, f):
        
        b, C, B = f.size()[0], f.size()[1], self.B;
    
        flm = self.SHT(torch.reshape(f, (b*C, 2*B, 2*B)));
        
        smooth = torch.reshape(torch.exp(self.smooth).repeat(b, 1), (b*C, ));
        
        return torch.reshape(self.ISHT( torch.exp(smooth[:, None, None] * self.lap[None, None, :]) * flm ), (b, C, 2*B, 2*B));
    


## Computes quadrature weights for integration
def sphereQuadratureDH(B):
    
    fName = cacheDir + '/intQuad_{}'.format(B) + '.pt';
    
    if (os.path.isfile(fName) == False):
    
        theta, phi = gridDH(B);

        theta = theta.float()
        phi = phi.float()
        
        x = torch.cos(theta) * torch.sin(phi);
        y = torch.sin(theta) * torch.sin(phi);
        z = torch.cos(phi);

        P = torch.cat( (x[..., None], y[..., None], z[..., None]), dim=-1);

        Q = torch.empty(2*B, 2*B).float().fill_(0);

        for i in range(0, 2*B):
            for j in range(0, 2*B):

                if (i == 0):

                    indP = [ 0, i+1, i+1, i+1, 0];
                    indT = [(j-1)% (2*B), (j-1)%(2*B), j, (j+1)% (2*B), (j+1)%(2*B)];

                elif (i == (2*B - 1)):

                    indP = [ i, (i-1), (i-1), (i-1), i ];
                    indT = [(j+1)%(2*B), (j+1)%(2*B), j, (j-1) % (2*B), (j-1)% (2*B)];

                else:

                    indP = [ i, (i-1), (i-1), (i-1), i, i+1, i+1, i+1, i ];
                    indT = [(j+1)%(2*B), (j+1)%(2*B), j, (j-1) % (2*B), (j-1)% (2*B), (j-1)%(2*B), j, (j+1)% (2*B), (j+1)%(2*B) ];

                A = 0;
                for k in range(len(indP)-1):

                    a = torch.linalg.vector_norm(torch.squeeze(P[indT[k], indP[k], :] - P[j, i, :]))
                    b = torch.linalg.vector_norm(torch.squeeze(P[indT[k+1], indP[k+1], :] - P[indT[k], indP[k], :]));

                    c = torch.linalg.vector_norm(torch.squeeze(P[j, i, :] - P[indT[k+1], indP[k+1], :]));

                    s = (a + b + c)/2;

                    A += torch.sqrt( s * (s - a) * (s - b) * (s - c)).item()/4.0;

                Q[j, i] = A;
                
        torch.save(Q, fName)
        
        print('Computed intQuad_{}'.format(B), flush=True);
        
    else:
        Q = torch.load(fName);
    
    return Q;
