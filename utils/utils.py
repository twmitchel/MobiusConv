import torch

EPS = 1e-7


## Complex Utils

def mappedToNull(x):
    
    return torch.logical_or(torch.isnan(x), torch.isinf(x))

def mappedToInf(X):
    
    return torch.logical_or(mappedToNull(X.real), mappedToNull(X.imag))
   
    
## Check if tensors are zero or origin
def isZero(x, eps=EPS):
    
    return torch.logical_and( torch.lt(x, eps), torch.gt(x, -eps) );

def isOrigin(z, eps=EPS):
    
    return torch.logical_and( isZero(z.real, eps), isZero(z.imag, eps) );


def realSgn(z):
    
    zOut = z.clone();
    
    lInd = torch.nonzero(z < 0, as_tuple=True);
    
    zOut[lInd] = -1.0 * (z[lInd].clone());
    
    return zOut;








    
    