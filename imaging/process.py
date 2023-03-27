import numpy as np
import nibabel as nib
import os
import pandas as pd

def roi_value(map,atlas_name,res,method=np.mean):
    map_data=nib.load(map).get_fdata()
    if atlas_name=='bna':
        atlas_data=nib.load('%s/../data/atlas/bna/BN_Atlas_246_%1dmm.nii' % (os.path.dirname(os.path.abspath(__file__)),res)).get_fdata()
        atlas_index=np.unique(atlas_data)
    elif atlas_name=='desikan-killiany':
        atlas_data=nib.load('%s/../data/atlas/desikan-killiany/Desikan-Killiany_MNI_SPM12_%1dmm.nii' % (os.path.dirname(os.path.abspath(__file__)),res)).get_fdata()
        atlas_index=np.unique(atlas_data)
        atlas_index=atlas_index[np.bitwise_and(atlas_index>1000,atlas_index<2036)]
        atlas_index=atlas_index[atlas_index!=2000]
        atlas_index=atlas_index[atlas_index!=2004]
        atlas_index=atlas_index[atlas_index!=1004]
    elif atlas_name=='jhu':
        atlas_data=nib.load('%s/../data/atlas/jhu/JHU-ICBM-labels-%1dmm.nii.gz' % (os.path.dirname(os.path.abspath(__file__)),res)).get_fdata()
        atlas_index=np.unique(atlas_data)
    roi_values=[]
    sort_index=np.sort(atlas_index)
    for index in sort_index:
        roi_values.append(method(map_data[atlas_data==index]))
    return roi_values
    



