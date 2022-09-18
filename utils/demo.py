import torch
import torch.nn.functional as F
import numpy as np
from math import pi as PI
from scipy.spatial.transform import Rotation as R
from TS2Kit.ts2kit import FTSHT, ITSHT, gridDH



def randSignal(batch_size, B, C):
    
    slm = 2*(torch.rand(batch_size, C, 2*B - 1, B, 2) - 0.5).float();
    
    slm = torch.view_as_complex(slm);
    
    for l in range(B):
        for m in range(-(B-1), B):
            
            if (m == 0):
                slm[..., m + (B-1), l] = slm[..., m + (B-1), l].clone().real;
                
            if (m*m > l*l):
                slm[..., m + (B-1), l] = 0.0;   
                
            if (m*m <= l * l and m < 0):
                slm[..., m + (B-1), l] = np.power(-1.0, -m) * torch.conj(slm[..., (-m)+(B-1), l].clone());
                
              
                
    N = torch.sqrt( torch.sum(slm.real * slm.real + slm.imag * slm.imag, dim=(2, 3)));
    
    
    return torch.reshape(ITSHT(B).float()(torch.reshape(slm / N[..., None, None], (batch_size*C, 2*B -1, B))), (batch_size, C, 2*B, 2*B))

   
class bilinearInterpolant(object):


    def __init__(self, theta, phi):
        
        B = theta.size()[0] // 2
    
        theta = torch.reshape( theta, (4*B*B, ) );
        phi = torch.reshape( phi, (4*B*B, ) );

        phi[phi < 0] = 0;
        phi[phi > PI] = PI-1e-7;

        theta[theta < 0] = 2 * PI + theta[theta < 0];

        px = torch.mul( theta, ( 2 * B) / (2 * PI)  );
        py = torch.mul(phi, (2*B - 1) / PI);

        py[py > 2*B - 1] = 2*B - 1;

        pyF = torch.floor(py).long();
        pyC = pyF + 1;

        pxF = torch.floor(px).long()
        pxC = pxF + 1;

        w00 = torch.mul(pxC - px, pyC - py);
        w10 = torch.mul(px - pxF, pyC - py);
        w01 = torch.mul(pxC - px, py - pyF);
        w11 = torch.mul(px - pxF, py - pyF);

        pxC = torch.remainder(pxC, (2*B));
        pxF = torch.remainder(pxF, (2*B));

        pyC[pyC > 2*B - 1] = 2*B - 1;
        pyF[pyF > 2*B - 1] = 2*B - 1;

        M, N = torch.meshgrid( torch.arange(0, 2*B), torch.arange(0, 2*B), indexing='ij');

        M = torch.reshape(M, (4*B*B, ));
        N = torch.reshape(N, (4*B*B, ));

        self.M = M;
        self.N = N;
        self.pxF = pxF;
        self.pyF = pyF;
        self.pxC = pxC;
        self.pyC = pyC;
        self.w00 = w00;
        self.w01 = w01;
        self.w10 = w10;
        self.w11 = w11;

    def __call__(self, im):
        
        M, N, pxF, pyF, pxC, pyC = self.M, self.N, self.pxF, self.pyF, self.pxC, self.pyC
        w00, w10, w01, w11 =  self.w00, self.w10, self.w01, self.w11
        
        imW = torch.zeros_like(im);
    
        imW[:, :, M, N] = (im[..., pxF, pyF] * w00[None, None, :] + im[..., pxC, pyF] * w10[None, None, :] 
                           + im[..., pxF, pyC] * w01[None, None, :] + im[..., pxC, pyC] * w11[None, None, :])
    
    
        return imW;

    

class randMobius(object):


    def __init__(self, B):
        
        self.B = B;
                
        theta, phi = gridDH(B)
        
        
        x0 = torch.cos(theta) * torch.sin(phi);
        y0 = torch.sin(theta) * torch.sin(phi);
        z0 = torch.cos(phi);
        
        self.X = torch.cat((x0[..., None], y0[..., None], z0[..., None]), dim=-1)
                

    def __call__(self, shift):
        
        B = self.B
                                           
        p = torch.rand(3, 1).float();
        
        
        
        p[0] = 2 * PI * p[0];
        p[1] = PI * p[1];
        p[2] = 2 * PI * p[2];
        
        if (shift > 1e-4):
            
            cA = torch.rand(2, 1).float()

            cA[0] = 2 * PI * cA[0];
            cA[1] = PI * cA[1];


            c = torch.zeros(3, 1).float();

            c[0] = np.cos(cA[0, 0].item()) * np.sin(cA[1, 0].item());
            c[1] = np.sin(cA[0, 0].item()) * np.sin(cA[1, 0].item());
            c[2] = np.cos(cA[1, 0].item());
            
            c = c * shift; 

            CT1 = 1.0 - c[0]*c[0] - c[1]*c[1] - c[2]*c[2];

            XC = self.X + c[None, None, ..., 0]

            XP = torch.div(CT1 * XC, XC[..., 0, None] * XC[..., 0, None] + XC[..., 1, None] * XC[..., 1, None] + XC[..., 2, None] * XC[..., 2, None]) + c[None, None, :, 0];


            XP = XP / torch.sqrt(XP[..., 0, None] * XP[..., 0, None] + XP[..., 1, None] * XP[..., 1, None] + XP[..., 2, None] * XP[..., 2, None]);
            
        else:
            
            XP = self.X;
                   
                                                          
        M = torch.tensor(R.from_euler('zyz', p.squeeze().cpu().numpy()).as_matrix());

        
        XP = torch.matmul(M[None, None, ...], XP[..., None]).squeeze(-1);
                             
        
        thetaM = torch.atan2(XP[..., 1], XP[..., 0])
        phiM = torch.acos(XP[..., 2]);     
        
        thetaM[thetaM < 0] = thetaM[thetaM < 0] + 2 * PI;
                
        return thetaM.float(), phiM.float();
