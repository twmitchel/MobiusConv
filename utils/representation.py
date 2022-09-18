import torch
import torch.nn.functional as F
import numpy as np
from math import pi as PI
import os
from scipy.special import gamma, factorial, gammaln, j0, jn_zeros, digamma;
from mpmath import meijerg, mp
from cache.cache import cacheDir, quadDir
from TS2Kit.ts2kit import gridDH

import subprocess
import shlex

import progressbar

mp.dps = 25

EPS = 1e-7

sqrt2 = np.sqrt(2.0);


## Value of t, controls filter-drop off.
t = 0.15;

## Heuristic 'good' values of sigma1, sigma2 for t = 0.15
sigma1 = -0.35
sigma2 = 0.15


#########################################
############# Filter Banks ##############
#########################################

## Heuristic quadrature samples (not learned)
## Inputs:
# Q: (int) Num quadrature samples
# m: (int) Angular frequency
## Outputs:
# S: (Q, ) torch float tensor of quadrature points
# W: (Q, ) torch float tensor of quadrature weights

def mellinQuadrature(Q, m):
    
    rng = [7, 5, 8, 10, 15];

    
    if (m == 0):
        r = rng[0];
    elif ( np.absolute(m) == 1 ):
        r = rng[1];
    elif ( np.absolute(m) == 2 ):
        r = rng[2];
    elif (np.absolute(m) == 3):
        r = rng[3];
    else:
        r = rng[4];
    
    S = ((torch.arange(Q).float() / (Q - 1)) * 2.0 * r) - r;
    
    W = torch.ones_like(S) * (S[1] - S[0]);
    
    W[0] = W[0] / 2.0;
    W[-1] = W[-1] / 2.0;
    
    return S, W;


def rsConvCoeffLT(B, M):
    
    fName = cacheDir + '/convCoeffLT_{}_{}'.format(B, M) + '.pt';
    rawName = cacheDir + '/convCoeff_{}_{}'.format(B, M) + '.txt';
    
    sPath = os.path.abspath(cacheDir);
    
    ## Compute S2 convolution coefficents
    if (os.path.isfile(fName) == False):
        
        if (os.path.isfile(rawName) == False):

            script = './precomp/build/bin/convcoeff --savePath "{}" --B {} --M {}'.format(sPath, B, M);

            subprocess.call(shlex.split(script))

            print('Computed conv coeffs...', flush=True);

        delVec = np.loadtxt(rawName);

        ind = torch.from_numpy(delVec[:, :4]).long();

        delta = torch.empty(B, B, 2*B - 1, 2*(M+1)).fill_(0).cfloat();

        delta[ind[:, 0], ind[:, 1], ind[:, 2] + (B-1), ind[:, 3] + M] = torch.view_as_complex(torch.from_numpy(delVec[:, 4:]).float())*torch.pow(torch.ones_like(ind[:, 3])*1j, -1.0*ind[:, 3]);

        print('Loaded conv coeffs, converting to sparse matrix...', flush=True);

        ## Convert to sparse matrix

        widgets = [progressbar.Percentage(), progressbar.Bar(), 
                  progressbar.AdaptiveETA()]    

        bar = progressbar.ProgressBar(max_value= 2*(M + 1)*(2*B-1), widgets=widgets)


        H = 0;
        W = 0;

        indH = [];
        indW = [];
        val = [];

        barCount = 0;
        for m in range(-M, M+2):

            if (m < (M + 1)):
                mInd = m;
                oInd = m + M;
            else:
                mInd = 0;
                oInd = 2*M + 1;

            # for each q, for each k, compute B x B matrix (L |--> S)

            for k in range( -(B-1), B):

                barCount = barCount + 1;
                bar.update(barCount);

                # (q, p, k)-th matrix

                for s in range( max(np.absolute(k), np.absolute(mInd)), B):

                    iH = H + s; 

                    for l in range(np.absolute(k), B):

                        iW = W + l;

                        fc = delta[l, s, k+(B-1), (-mInd) + M];

                        if ( torch.abs(fc).item() > 1.0e-12 ):

                            indH.append(iH);
                            indW.append(iW);
                            val.append(fc);



                H += B;
                W += B;

            # End k loop

        # End q loop

        # Cat indices, turn into sparse matrix
        ind = torch.cat( (torch.tensor(indH).long()[None, :], torch.tensor(indW).long()[None, :]), dim=0);
        val = torch.tensor( val ).cfloat();

        DC = torch.sparse_coo_tensor(ind, val, [H, W], dtype=torch.cfloat)
        
        torch.save(DC, fName);
        
    else:
        
        DC = torch.load(fName);
    
    return DC;


def logPolarFilters(B, D1, D2, t=t):
    
    F = torch.empty(2*D1 + 1, 2*D2 + 1, 2*B, 2*B).fill_(0).cfloat();

    theta, phi = gridDH(B);
    
    lnr = torch.log(torch.tan(phi/2));
    
    for u in range(-D1, D1+1):
        for v in range(-D2, D2+1):  
            
            F[u+D1, v+D2, ...] = torch.mul(torch.exp( torch.complex(-1.0 * lnr * t, lnr * u)), torch.exp( torch.complex(torch.zeros_like(theta), -1.0*v*(theta + PI))) );
         
        
    return F;

def logPolarFiltersLT(B, M, Q, sigma=None, S=None):
        
    F = torch.empty(Q, 2*(M + 1), 2*B, 2*B).fill_(0).cfloat();

    theta, phi = gridDH(B);
    
    lnr = torch.log(torch.tan(phi/2));
    
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
                sig = sigma[m+M];
            else:
                sig = sigma2;
            mInd = 0;
            oInd = 2*M + 1;
        
        if (S is not None):            
            s = S[m+M, :];
        else:         
            s, _ = mellinQuadrature(Q, mInd);

        for q in range(Q):
                       
            F[q, oInd, ...] = torch.mul(torch.exp( torch.complex(lnr * sig, lnr * s[q])), torch.exp(  -1j*mInd*(theta + PI)) );
            
    return F;



    


#################################################################
########## Bessel decomp of radial filter component ############# 
#################################################################


## Meijer-G Function Mellin Coefficents
def MeijerMellinS1(t, sigma1, s, m, u, w):
    
    if ( (u == 0) and (m == 0) ):
       
        M = (1.0/(1 - 0.5*(2 - 1j*s + t)))* gamma(0.5*(2 + sigma1 + 1j*w))*gamma(0.5*(1j*s - sigma1 - 1j*w - t))/ (2*gamma(0.5*(2 - sigma1 - 1j*w))*gamma(0.5*(2 - 1j*s + sigma1 + 1j*w + t)));
        
    elif ( u == 0 ):
        
        M = 0.0;
        
    elif ( (u >= m) and (u < 0) ):
        
        M = gamma(0.5*(-u + sigma1 + 1j*w))*gamma(0.5*(u + 1j*s - m - sigma1 - 1j*w - t))/(2 * gamma(0.5*(2 - u - sigma1 - 1j*w))*gamma(0.5*(2 + u -1j*s - m + sigma1 + 1j*w + t)));
        
    elif ( (u >= m) and (u > 0) ):
        
        M = np.power(-1.0, np.absolute(u))*gamma(0.5*(u + sigma1 + 1j*w))*gamma(0.5*(u + 1j*s - m - sigma1 - 1j*w - t))/(2 * gamma(0.5*(2 + u - sigma1 - 1j*w))*gamma(0.5*(2 + u -1j*s - m + sigma1 + 1j*w + t)));
        
    elif ( (u < m) and (u < 0) ):
        
        M = np.power(-1.0, np.absolute(u - m))*gamma(0.5*(-u + sigma1 + 1j*w))*gamma(0.5*(-u + 1j*s + m - sigma1 - 1j*w - t))/(2 * gamma(0.5*(2 - u - sigma1 - 1j*w))*gamma(0.5*(2 - u -1j*s + m + sigma1 + 1j*w + t)));
        
    elif ( (u < m) and (u > 0) ):
        
        M = np.power(-1.0, np.absolute(m))*gamma(0.5*(u + sigma1 + 1j*w))*gamma(0.5*(-u + 1j*s + m - sigma1 - 1j*w - t))/(2 * gamma(0.5*(2 + u - sigma1 - 1j*w))*gamma(0.5*(2 - u -1j*s + m + sigma1 + 1j*w + t)));
        
    
    return M;


def MeijerMellinS2(t, sigma2, s, m, u, w):
    
    if ( u != 0 ):
        
        M = 0;
        
    elif ( (u == 0) and (m < 0) ):
        
        M = gamma(0.5*(sigma2 + 1j*w))*gamma(0.5*(1j*s - m - sigma2 - 1j*w - t))/(2*gamma(0.5*(2 - sigma2 - 1j*w))*gamma(0.5*(2 - 1j*s - m + sigma2 + 1j*w + t)));
        
    elif ( (u == 0) and (m > 0) ):
        
        M = np.power(-1.0, np.absolute(m))*gamma(0.5*(sigma2 + 1j*w))*gamma(0.5*(1j*s + m - sigma2 - 1j*w - t))/(2*gamma(0.5*(2 - sigma2 - 1j*w))*gamma(0.5*(2 - 1j*s + m + sigma2 + 1j*w + t)));
        
    elif ( (u == 0) and (m == 0) ):
        
        M = (1.0/(1 - 0.5*(2 - 1j*s + t)))*gamma(0.5*(sigma2 + 1j*w))*gamma(0.5*(2 + 1j*s - sigma2 - 1j*w -t))/(2*gamma(0.5*(2 - sigma2 - 1j*w))*gamma(0.5*(2 - 1j*s + sigma2 + 1j*w + t)));
    
    return M;


def radBesselCoeff(s, m, t):
    
    if (m <= 0):
    
        R = gamma(0.5*(2 - 1j*s - m + t))/gamma(0.5*(1j*s - m - t));
    
    else:
        
        R = np.power(-1.0, np.absolute(m))*gamma(0.5*(2 - 1j*s + m + t))/gamma(0.5*(1j*s + m - t));
        
    return R;

def linearizeLogPolar(B, D1, D2, M, Q, sigma=None, S=None, W=None, t=t):        
    
    ## Create or load filter linearization
    fName = cacheDir + '/linearFilters_{}_{}_{}_{}_{}_{}'.format(B, D1, D2, M, Q, t) + '.pt';

    if (os.path.isfile(fName) == False):

        F = torch.empty(Q, 2*(M+1), 2*D1 + 1, 2*D2+1).fill_(0).cfloat();
        

        for m in range(-M, M+2):

            if (m < (M + 1)):
                mInd = m;
                oInd = m + M;
                MFn  = MeijerMellinS1
                
                if (sigma is not None):
                    sig = float(sigma[oInd]);
                else:
                    sig = sigma1;
            else:
                mInd = 0;
                oInd = 2*M + 1;
                MFn = MeijerMellinS2
                
                if (sigma is not None):
                    sig = float(sigma[oInd]);
                else:
                    sig = sigma2;

            if (S is not None):
                s = S[oInd, :];
                w = W[oInd, :];
            else:                
                s, w = mellinQuadrature(Q, mInd);

            for q in range(Q):
                
                for u in range(-D1, D1+1):
                    for v in range(-D2, D2+1):
                        
                        F[q, oInd, u + D1, v + D2] = w[q]*radBesselCoeff(u, v, t) * MFn(t, sig, u, v, mInd, s[q]);
                        
        
        torch.save(F, fName);
        
    
    else:
    
        F = torch.load(fName);
        
    
    return F;
    
    
    
###########################################################
################ Quadrature Analysis ######################
########################################################### 


def aMMeijer(s, m, u, t):
    
    return [[0.5*(2 - u - 1j*s + m + t)], [0.5*(2 + u - 1j*s - m + t)]];

def bMMeijer(u):
    return [[-0.5*u], [0.5*u]];


def meijerM(s, m, u, t, r):
    
    r2 = r*r;
    
    b = bMMeijer(u);
    
    c = 1;
    
    if ( u >= m):
        a = aMMeijer(s, m, u, t);
    else:
        a = aMMeijer(s, -m, -u, t);
        
        if ( (u - m) % 2 == 1):
            c = -1;
    
    out = c * meijerg(a, b, r2);
    
    return float(out.real) + 1j*float(out.imag);


def sampleMellinCoeffs(U, V, M, t, S, sigma1, sigma2):
    
    
    
    nS = S.size()[0];
    nSig = sigma1.size()[-1];
    
    # Out: U X V X M x S x nSigma
    
    fName = quadDir + '/mellinCoeffSample_{}_{}_{}_{}_{}_{}_{}.pt'.format(U, V, M, t, nS, nSig, int(S[-1]))

        
    if (os.path.isfile(fName) == False):
        
        MC = torch.empty(2*U + 1, 2*V + 1, 2*M + 2, nS, nSig).fill_(0).cdouble();

        print('Sampling Mellin Coeffs', flush=True);

        widgets = [progressbar.Percentage(), progressbar.Bar(), 
                progressbar.AdaptiveETA()]    

        bar = progressbar.ProgressBar(max_value=(2*V + 1)*(2*M + 2), widgets=widgets)

        barCount = 0;
        
        uA = torch.arange(-U, U+1)[:, None, None].numpy();
        sA = S[None, :, None].numpy();
        
        for m in range(-M, M+2):
                    
            if (m < M + 1):
                mInd = m;
                MFn = MeijerMellinS1
                
                if (sigma1.dim() > 1):                   
                    sig = sigma1[m+M, :];
                else:
                    sig = sigma1;
                    
            else:
                mInd = 0;
                MFn = MeijerMellinS2
                sig = sigma2;
            
            sigA = sig[None, None, :].numpy();
            
            
            for v in range(-V, V+1):
                #print(MFn(t, sigA, uA, v, qInd, sA), flush=True)
                
                MC[:, v+V, m + M, :, :] = torch.tensor(MFn(t, sigA, uA, v, mInd, sA)).cdouble();
                
                barCount = barCount + 1;
                bar.update(barCount);
                
          
        print('Done!');
        
        torch.save( MC, fName);
        
    else:
        
        MC = torch.load(fName);
                
    return MC


     
                        
def sampleMeijerM(U, V, M, t, r):
    
    fName = quadDir + '/meijerMSample_{}_{}_{}_{}_{}_{}'.format(U, V, M, t, r.size()[0], int(r[-1])) + '.pt';

    if (os.path.isfile(fName) == False):
        
        print('No file {}'.format(fName), flush=True);
        print('Sampling Meijer M', flush=True);

        widgets = [progressbar.Percentage(), progressbar.Bar(), 
            progressbar.AdaptiveETA()]    

        bar = progressbar.ProgressBar(max_value=r.size()[0], widgets=widgets)
        
        nR = r.size()[0];

        MM = torch.empty(2*U + 1, 2*V + 1, 2*M + 1, nR).fill_(0).cdouble();

        
        for q in range(nR):
            bar.update(q);
            for u in range(-U, U+1):
                for v in range(-V, V+1):
                    for m in range(-M, M+1):

                        MM[u+U, v+V, m+M, q] = meijerM(u, v, m, t, r[q].item());
                        
                        

        print('Done!', flush=True)
        
        torch.save(MM, fName)
        
    else:
        
        MM = torch.load(fName);
    
    return MM
