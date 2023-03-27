import numpy as np
from scipy.spatial.distance import pdist

class distance():
    def __init__(self, X):
        self.X=X
    def fit(self, mode='pmd'):
        if mode=='pmd':
            return self.penalized_mahalanobis_distance()
        elif mode=='mahalanobis':
            return self.mahalanobis_distance()
    
    def penalized_mahalanobis_distance(self,pSI=None):
        X=self.X-np.mean(self.X,axis=0)
        if pSI==None:
            XT=X.T
            u, sigma, vt=np.linalg.svd(XT)
            S=np.diag(sigma)
            D=np.dot(S,S.T)
            DD=np.diag(D)
            lam=np.min(DD)
            # regD=np.diag(1/(np.diag(D)+lam))
            regD=np.zeros(u.shape)
            for i in range(regD.shape[0]):
                if i<DD.shape[0]:
                    regD[i,i]=1/(np.diag(D)[i]+lam)
                else:
                    regD[i,i]=1/lam 
            pSI=X.shape[0]*np.dot(np.dot(u,regD),u.T)
        dis=np.zeros((X.shape[0],X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(i+1,X.shape[0]):
                delta=X[i]-X[j]
                dis[i,j]=np.sqrt(np.dot(np.dot(delta,pSI),delta.T))
        dis+=dis.T
        return dis, pSI

    def mahalanobis_distance(self):
        return pdist(self.X,'mahalanobis')

