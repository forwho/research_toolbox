import numpy as np
from scipy.stats import binom, norm

def signtest(x,y=None,method='exact',direction='right'):
    if y==None:
        y=np.zeros(x.shape)
    diff=x-y
    npos=diff[diff>0].shape[0]
    nneg=x.shape[0]-npos
    if method=='exact':
        if direction=='both':
            sgn=min(nneg,npos)
            p=min(1,2*binom.cdf(sgn,x.shape[0],0.5))
        elif direction=='right':
            p=binom.cdf(nneg,x.shape[0],0.5)
        else:
            p=binom.cdf(npos,x.shape[0],0.5)
        zval=np.nan
    else:
        if direction=='both':
            z=(npos-nneg-np.sign(npos-nneg))/np.sqrt(x.shape[0])
            p=2*norm.cdf(-np.abs(z),0,1)
        elif direction=='right':
            z=(npos-nneg-1)/np.sqrt(x.shape[0])
            p=norm.cdf(-z,0,1)
        else:
            z=(npos-nneg+1)/np.sqrt(x.shape[0])
            p=norm.cdf(z,0,1)
        zval=z
    return zval, p
