import glob
import os

cacheDir = 'cache/files'
quadDir = 'quadrature/files'


def clearCache(cacheDir=cacheDir):
    
    cFiles  = glob.glob(cacheDir + '/*.pt')
    
    for l in range(len(cFiles)):
        
        os.remove(cFiles[l]);
        
    
    return 1; 
