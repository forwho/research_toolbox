import nibabel as nib
import numpy as np
import os


def array2dscalar(data,filename,mode='bna'):
    if mode=='bna':
        atlas_file='%s/../../data/surface_data/parcellation/fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii' % os.path.dirname(os.path.abspath(__file__))
        atlas=nib.load(atlas_file)
        atlas_data=atlas.get_fdata()
        new_data=np.zeros(atlas_data.shape)
        new_data[new_data==0]=np.nan
        for i in range(1,211,2):
            new_data[atlas_data==i]=data[i-1]
        for i in range(212,421,2):
            new_data[atlas_data==i]=data[i-211]
    if mode=='aparc':
        atlas_file='%s/../../data/surface_data/parcellation/fsaverage.aparc.32k_fs_LR.dlabel.nii' % os.path.dirname(os.path.abspath(__file__))
        atlas=nib.load(atlas_file)
        atlas_data=atlas.get_fdata()
        new_data=np.zeros(atlas_data.shape)
        new_data[new_data==0]=np.nan
        data=np.insert(data,3,np.nan)
        data=np.insert(data,38,np.nan)
        for i in range(1,71):
            new_data[atlas_data==i]=data[i-1]
    scalar_axis=nib.cifti2.cifti2_axes.ScalarAxis(['val'])
    brain_model_axis=atlas.header.get_axis(1)
    val_head=nib.cifti2.Cifti2Header.from_axes((scalar_axis,brain_model_axis))
    cifti=nib.cifti2.cifti2.Cifti2Image(new_data,val_head,atlas.nifti_header,atlas.extra,atlas.file_map)
    nib.save(cifti,filename)

def save_subcortical(data,filename,mode='bna'):
    if mode=='bna':
        bna=nib.load('%s/../../data/atlas/bna/BN_Atlas_246_1mm.nii.gz' % os.path.dirname(os.path.abspath(__file__)))
        bna_data=bna.get_fdata()
        bna_data[bna_data<=210]=0
        for i in range(210,247):
            bna_data[bna_data==i]=data[i-1]
        new_nifti=nib.Nifti2Image(bna_data,bna.affine)
    nib.save(new_nifti,filename)
