import os
from brainspace.null_models import MoranRandomization
import scipy
import nibabel as nib
import numpy as np

def nii_center(nifti_file):
    nif=nib.load(nifti_file)
    data=np.asanyarray(nif.dataobj)
    center=np.zeros((int(np.max(data)),4))
    for i in range(center.shape[0]):
        center[i,0:3]=np.mean(np.where(data==i+1),axis=1)
    center[:,3]=1
    center=np.matmul(center,nif.affine.T)
    return center[:,0:3]

def calc_pro_mat(center):
    mat=np.zeros((center.shape[0],center.shape[0]))
    for i in range(center.shape[0]):
        for j in range(i+1,center.shape[0]):
            mat[i,j]=np.linalg.norm(center[i]-center[j])*(-1)
    mat=mat+mat.T
    return mat

def moran_random(nifti_file,map,repeatn=1000):
    center=nii_center(nifti_file)
    pro_mat=calc_pro_mat(center)
    msr = MoranRandomization(n_rep=repeatn, procedure='singleton', tol=1e-6,random_state=0)
    msr.fit(pro_mat)
    xrand=msr.randomize(map)
    return xrand


def autocorr(xmap,ymap,pro_mat,repeatn=1000,method='pearson'):
    msr = MoranRandomization(n_rep=repeatn, procedure='singleton', tol=1e-6,random_state=0)
    msr.fit(pro_mat)
    xrand=msr.randomize(xmap)
    if method=='pearson':
        r_obs, p_obs=scipy.stats.pearsonr(xmap,ymap)
    else:
        r_obs, p_obs=scipy.stats.spearmanr(xmap,ymap)
    if method=='pearson':
        r_rand=np.asarray([scipy.stats.pearsonr(d,ymap)[0] for d in xrand])
    else:
        r_rand=np.asarray([scipy.stats.spearmanr(d,ymap)[0] for d in xrand])
    p_rand=np.mean(np.abs(r_rand) >= np.abs(r_obs))
    return r_obs, p_rand, xrand

def spatial_autocorrelation_correction(xmap,ymap,xrand=None,repeatn=1000,method='pearson',atlas='bna'):
    if atlas=='bna':
        atlas_file='%s/../data/atlas/bna/BN_Atlas_246_1mm.nii.gz' % os.path.dirname(os.path.abspath(__file__))
    if np.any(xrand==None):
        center=nii_center(atlas_file)
        pro_mat=calc_pro_mat(center)
        msr = MoranRandomization(n_rep=repeatn, procedure='singleton', tol=1e-6,random_state=0)
        msr.fit(pro_mat)
        xrand=msr.randomize(xmap)
    if method=='pearson':
        r_obs, p_obs=scipy.stats.pearsonr(xmap,ymap)
    else:
        r_obs, p_obs=scipy.stats.spearmanr(xmap,ymap)
    if method=='pearson':
        r_rand=np.asarray([scipy.stats.pearsonr(d,ymap)[0] for d in xrand])
    else:
        r_rand=np.asarray([scipy.stats.spearmanr(d,ymap)[0] for d in xrand])
    p_rand=np.mean(np.abs(r_rand) >= np.abs(r_obs))
    return r_obs, p_rand, xrand
