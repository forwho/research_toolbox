import numpy as np
import os
from research_toolbox.rt_stat import space_correction as sc

from research_toolbox.network import graphic_metrics as gm
from scipy import stats
from nibabel.freesurfer.io import read_annot
import nibabel as nib

from sklearn.cross_decomposition import PLSRegression
import pandas as pd
from pygam import LinearGAM, s
from research_toolbox import rt_stat

path='%s/../' % os.path.dirname(os.path.abspath(__file__))

def yeo_stat(dv,is_print=True):
    yeo_name=['Visual', 'Somatomotor', 'Dorsal attention', 'Ventral attention', 'Limbic', 'Frontoparietal', 'Default','Subcortical']
    yeo_data=pd.read_csv(path+'/data/atlas/bna/subregion_func_network_Yeo_updated.csv')
    yeo_data=yeo_data.iloc[:,3].to_numpy()
    yeo_data=yeo_data[0:210]
    dv_conn=dv[0:210]
    dv_yeo=np.zeros(8)
    for i in range(1,8):
        dv_yeo[i-1]=np.nanmean(dv_conn[yeo_data==i])
    dv_yeo[7]=np.nanmean(dv[211:246])
    sort_index=np.argsort(dv_yeo)
    if is_print:
        print('---------------Results of individual difference in different subnetwork based on yeo template-------------------')
        for i in range(8):
            print('The individual differences in %s is %f' % (yeo_name[sort_index[i]], dv_yeo[sort_index[i]]))
            if i==7:
                print('\n')
    return dv_yeo

def corr_degree(dv,nets,mode='network'):
    degree, strength=gm.nodal_metrics_calc(nets)
    nifile=path+'/data/atlas/bna/BN_Atlas_246_1mm.nii.gz'
    center=sc.nii_center(nifile)
    pro_mat=sc.calc_pro_mat(center)

    if mode=='network':
        xmap=np.mean(degree,axis=0)
        degree_r, degree_p=spatial_corr(xmap,dv,pro_mat,1000)
        xmap=np.mean(strength,axis=0)
        strength_r, strength_p=spatial_corr(xmap,dv,pro_mat,1000)
        print('---------------Results of correlation between individual difference with degree and strength-------------------')
        print('The correlation between individual difference with degree is %f(%f)' % (degree_r, degree_p))
        print('The correlation between individual difference with strength is %f(%f)\n' % (strength_r, strength_p))
    elif mode=='subnetwork':
        print('---------------Results of correlation between individual difference with degree and strength in subnetwork-------------------')
        degree_r=[]
        degree_p=[]
        strength_r=[]
        strength_p=[]
        yeo_name=['Visual', 'Somatomotor', 'Dorsal attention', 'Ventral attention', 'Limbic', 'Frontoparietal', 'Default','Subcortical']
        yeo_data=pd.read_csv(path+'/data/atlas/bna/subregion_func_network_Yeo_updated.csv')
        yeo_data=yeo_data.iloc[:,3].to_numpy()
        yeo_data=yeo_data[0:210]
        yeo_data=np.concatenate((yeo_data,np.ones(36)*8))
        for i in range(8):
            xmap=np.mean(degree,axis=0)[yeo_data==i+1]
            ymap=dv[yeo_data==i+1]
            tmp_mat=pro_mat[np.ix_(np.where(yeo_data==i+1)[0],np.where(yeo_data==i+1)[0])]
            tmp_r, tmp_p=spatial_corr(xmap,ymap,tmp_mat,1000)
            print('The correlation between individual difference with degree is %f(%f) in %s' % (tmp_r, tmp_p, yeo_name[i]))
            degree_r.append(tmp_r)
            degree_p.append(tmp_p)
            xmap=np.mean(strength,axis=0)[yeo_data==i+1]
            ymap=dv[yeo_data==i+1]
            tmp_r, tmp_p=spatial_corr(xmap,ymap,tmp_mat,1000)
            print('The correlation between individual difference with strength is %f(%f) in %s' % (tmp_r, tmp_p, yeo_name[i]))
            strength_r.append(tmp_r)
            strength_p.append(tmp_p)
            if i==7:
                print('\n')    
    return degree_r, degree_p, strength_r, strength_p

def corr_other_maps(dv):
    nifile=path+'/data/atlas/bna/BN_Atlas_246_1mm.nii.gz'
    center=sc.nii_center(nifile)
    pro_mat=sc.calc_pro_mat(center)
    results={}
    # Evolution map
    annot_file=path+'/data/atlas/bna/fsaverage5/label/rh.BN_Atlas.annot'
    
    gifti_file=path+'/data/Brain_Organization/EvolutionaryExpansion/rh.Hill2010_evo_fsaverage5.func.gii'
    labels,ctab,names=read_annot(annot_file)
    evo_map=nib.load(gifti_file).agg_data()
    val_data=np.zeros(210)
    for i in range(0,210):
        if evo_map[labels==(i+1)].shape[0]>0:
            val_data[i]=np.nanmean(evo_map[labels==(i+1)])
        else:
            val_data[i]=0
    new_labels=np.zeros(labels.shape)
    for i in range(0,210):
        new_labels[labels==i+1]=val_data[i]

    rindex=np.arange(1,210,2)
    rdv_conn=dv[rindex]
    revo_data=val_data[rindex]
    tmp_mat=pro_mat[1:210:2,1:210:2]
    rval, pval, xrand=spatial_corr(revo_data,rdv_conn,tmp_mat,1000)
    results['Evo']=[rval,pval]
    print('---------------Results of correlation between individual difference with othermaps-------------------')
    print('The correlation between individual difference with evolution map is %f(%f)' % (rval, pval))
    # CBF map, Myelin map and gradients map
    map_names=['CBF', 'Myelin', 'Gradients', 'HistGradients_G1','HistGradients_G2','L1Thickness','L2Thickness','L3Thickness','L4Thickness','L5Thickness','L6Thickness']
    # map_names=['L1Thickness']
    # val_files=['L1Thickness/L1Thickness']
    k=0
    val_files=['MeanCBF/MeanCBF', 'Myelin/MyelinMap','PrincipleGradient/Gradients','HistGradients_G1/HistGradients_G1','HistGradients_G2/HistGradients_G2','L1Thickness/L1Thickness','L2Thickness/L2Thickness','L3Thickness/L3Thickness','L4Thickness/L4Thickness','L5Thickness/L5Thickness','L6Thickness/L6Thickness']
    tmp_mat=pro_mat[0:210,0:210]
    val_datas=[]
    for val_file in val_files:
        val_file=path+'/data/Brain_Organization/'+val_file
        # f, axs = plt.subplots(2, 2, figsize=(8, 8),subplot_kw={'projection': '3d'})
        if val_file[-16:-3]=='HistGradients' or val_file[-9:]=='Thickness':
            annot_file=path+'/data/atlas/bna/fsaverage/label/lh.BN_Atlas.annot'
            gifti_file=val_file+'.lh.fsaverage.func.gii'
        else:
            annot_file=path+'/data/atlas/bna/fsaverage5/label/lh.BN_Atlas.annot'
            gifti_file=val_file+'.lh.fsaverage5.func.gii'
        labels,ctab,names=read_annot(annot_file)
        evo_map=nib.load(gifti_file).agg_data()
        if val_file== path+'/data/Brain_Organization/PrincipleGradient/Gradients':
            evo_map=evo_map[0]
        val_data=np.zeros(210)
        for i in range(0,210,2):
            if evo_map[labels==(i+1)].shape[0]>0:
                val_data[i]=np.nanmean(evo_map[labels==(i+1)])
            else:
                val_data[i]=0
        new_labels=np.zeros(labels.shape)
        for i in range(0,210,2):
            new_labels[labels==i+1]=val_data[i]  
        
        if val_file[-16:-3]=='HistGradients' or val_file[-9:]=='Thickness':
            annot_file=path+'/data/atlas/bna/fsaverage/label/rh.BN_Atlas.annot'
            gifti_file=val_file+'.rh.fsaverage.func.gii'
        else:
            annot_file=path+'/data/atlas/bna/fsaverage5/label/rh.BN_Atlas.annot'
            gifti_file=val_file+'.rh.fsaverage5.func.gii'
        labels,ctab,names=read_annot(annot_file)
        evo_map=nib.load(gifti_file).agg_data()
        if val_file== path+'/data/Brain_Organization/PrincipleGradient/Gradients':
            evo_map=evo_map[0]
        for i in range(1,210,2):
            if evo_map[labels==(i+1)].shape[0]>0:
                val_data[i]=np.nanmean(evo_map[labels==(i+1)])
            else:
                val_data[i]=0

        new_labels=np.zeros(labels.shape)
        for i in range(1,210,2):
            new_labels[labels==(i+1)]=val_data[i]
        
        rval, pval, xrand=spatial_corr(val_data,dv[0:210],tmp_mat,1000)
        val_datas.append(val_data)
        print('The correlation between individual difference with %s map is %f(%f)' % (map_names[k], rval, pval))
        if k==2:
            print('\n')
        results[map_names[k]]=[rval, pval]
        
        # np.save(data_path+'indi-data/results/batch_results/maps/'+map_names[k]+'.npy',val_data)
        k+=1
    return results, revo_data, val_datas

def len_nets_stat(dv,nets,len_nets,threshold,method='thres',mode='network',edge_thre=0,is_print=True):
    nifile=path+'/data/atlas/bna/BN_Atlas_246_1mm.nii.gz'
    center=sc.nii_center(nifile)
    pro_mat=sc.calc_pro_mat(center)
    nodes_num=246
    dv=dv[0:nodes_num]
    if method=='thres':
        degree=np.zeros((2,len_nets.shape[0],nodes_num))
        strength=np.zeros((2,len_nets.shape[0],nodes_num))
        for i in range(len_nets.shape[0]):
            len_net=np.copy(len_nets[i])
            weight_net=np.copy(nets[i])
            len_net[np.diag_indices(246)]=0
            weight_net[np.diag_indices(246)]=0
            weight_net[weight_net<edge_thre]=0
            for j in range(nodes_num):
                len_row=len_net[j,:]
                weight_row=weight_net[j,:]
                strength[0,i,j]=np.nansum(weight_row[np.bitwise_and(len_row>0,len_row<threshold)])
                strength[1,i,j]=np.nansum(weight_row[len_row>=threshold])
                degree_row=np.copy(weight_row)
                degree_row[degree_row>0]=1
                degree[0,i,j]=np.nansum(degree_row[np.bitwise_and(len_row>0,len_row<threshold)])
                degree[1,i,j]=np.nansum(degree_row[len_row>=threshold])
        mean_degree=np.mean(degree,axis=1)
        mean_strength=np.mean(strength,axis=1)
        long_percent=np.nansum(degree[1],axis=0)/246#np.nansum(np.nansum(degree,axis=0),axis=0)
        short_percent=np.nansum(degree[0],axis=0)/246#np.nansum(np.nansum(degree,axis=0),axis=0)
        results={}
        if mode=='network':
            if is_print:
                print('---------------Results of correlation between individual difference with edges of different  length-------------------')
            for i in range(mean_degree.shape[0]):
                rval, pval=spatial_corr(mean_degree[i],dv,pro_mat,1000)
                results['degree-%d' % i]=[rval, pval]
                if is_print:
                    print('The correlation between individual difference and %d degree is %f(%f)' % (i, rval, pval))
            for i in range(mean_strength.shape[0]):
                rval, pval=spatial_corr(mean_strength[i],dv,pro_mat,1000)
                results['strength-%d' % i]=[rval, pval]
                if is_print:
                    print('The correlation between individual difference and %d strength is %f(%f)' % (i, rval, pval))
            rval, pval=spatial_corr(long_percent,dv,pro_mat,1000)
            results['long-percent']=[rval, pval]
            if is_print:
                print('The correlation between individual difference and percent of long edges is %f(%f)\n' % (rval,    pval))
            rval, pval=spatial_corr(short_percent,dv,pro_mat,1000)
            results['short-percent']=[rval, pval]
            if is_print:
                print('The correlation between individual difference and percent of short edges is %f(%f)\n' % (rval,    pval))
        elif mode=='subnetwork':
            if is_print:
                print('---------------Results of correlation between individual difference with edges of different  length in subnetwork-------------------')
            yeo_name=['Visual', 'Somatomotor', 'Dorsal attention', 'Ventral attention', 'Limbic', 'Frontoparietal', 'Default','Subcortical']
            yeo_data=pd.read_csv(path+'/data/atlas/bna/subregion_func_network_Yeo_updated.csv')
            yeo_data=yeo_data.iloc[:,3].to_numpy()
            yeo_data=yeo_data[0:210]
            yeo_data=np.concatenate((yeo_data,np.ones(36)*8))
            for i in range(mean_degree.shape[0]):
                for j in range(8):
                    tmp_mat=pro_mat[np.where(yeo_data==j+1)[0]][:,np.where(yeo_data==j+1)[0]]
                    rval, pval=spatial_corr(mean_degree[i][yeo_data==j+1],dv[yeo_data==j+1],tmp_mat,1000)
                    results['degree-%d-%d' % (i,j)]=[rval, pval]
                    if is_print:
                        print('The correlation between individual difference and %d degree is %f(%f) in %s' % (i, rval, pval, yeo_name[j]))
            for i in range(mean_strength.shape[0]):
                for j in range(8):
                    tmp_mat=pro_mat[np.where(yeo_data==j+1)[0]][:,np.where(yeo_data==j+1)[0]]
                    rval, pval=spatial_corr(mean_strength[i][yeo_data==j+1],dv[yeo_data==j+1],tmp_mat,1000)
                    results['strength-%d-%d' % (i,j)]=[rval, pval]
                    if is_print:
                        print('The correlation between individual difference and %d strength is %f(%f) in %s' % (i, rval, pval, yeo_name[j]))
            for j in range(8):
                tmp_mat=pro_mat[np.where(yeo_data==j+1)[0]][:,np.where(yeo_data==j+1)[0]]
                rval, pval=spatial_corr(long_percent[yeo_data==j+1],dv[yeo_data==j+1],tmp_mat,1000)
                results['percent-%d' % j]=[rval, pval]
                if is_print:
                    print('The correlation between individual difference and percent of long edges is %f(%f) in %s' % (rval,    pval, yeo_name[j]))
                if j==7:
                    if is_print:
                        print('\n')

    else:
        degree=np.zeros((threshold,len_nets.shape[0],nodes_num))
        strength=np.zeros((threshold,len_nets.shape[0],nodes_num))
        edges=np.array([])
        for net in len_nets:
            net=net.flatten()
            edges=np.concatenate((edges, net),axis=0)
        edges=edges[edges>0]
        thres=[]
        for i in range(threshold-1):
            thres.append(np.percentile(edges,100*(i+1)/threshold))
        print(thres)
        for i in range(len_nets.shape[0]):
            len_net=np.copy(len_nets[i])
            weight_net=np.copy(nets[i])
            len_net[np.diag_indices(246)]=0
            weight_net[np.diag_indices(246)]=0
            weight_net[weight_net<edge_thre]=0
            for j in range(nodes_num):
                len_row=len_net[j,:]
                weight_row=weight_net[j,:]
                for k in range(threshold):
                    if k==0:
                        strength[k,i,j]=np.nansum(weight_row[np.bitwise_and(len_row>0,len_row<thres[k])])
                    elif k<threshold-1:
                        strength[k,i,j]=np.nansum(weight_row[np.bitwise_and(len_row<thres[k],len_row>=thres[k-1])])
                    else:
                        strength[k,i,j]=np.nansum(weight_row[len_row>=thres[k-1]])
                degree_row=np.copy(weight_row)
                degree_row[degree_row>0]=1
                for k in range(threshold):
                    if k==0:
                        degree[k,i,j]=np.nansum(degree_row[np.bitwise_and(len_row>0,len_row<thres[k])])
                    elif k<threshold-1:
                        degree[k,i,j]=np.nansum(degree_row[np.bitwise_and(len_row<thres[k],len_row>=thres[k-1])])
                    else:
                        degree[k,i,j]=np.nansum(degree_row[len_row>=thres[k-1]])
        mean_degree=np.mean(degree,axis=1)
        mean_strength=np.mean(strength,axis=1)
        long_percent=np.nansum(degree[-1],axis=0)/246#np.nansum(np.nansum(degree,axis=0),axis=0)
        short_percent=np.nansum(degree[0],axis=0)/246#np.nansum(np.nansum(degree,axis=0),axis=0)
        results={}
        results['thres']=thres
        if mode=='network':
            if is_print:
                print('---------------Results of correlation between individual difference with edges of different  length-------------------')
            for i in range(mean_degree.shape[0]):
                rval, pval,xrand=spatial_corr(mean_degree[i],dv,pro_mat,1000)
                results['degree-%d' % i]=[rval, pval]
                if is_print:
                    print('The correlation between individual difference and %d degree is %f(%f)' % (i, rval, pval))
            for i in range(mean_strength.shape[0]):
                rval, pval,xrand=spatial_corr(mean_strength[i],dv,pro_mat,1000)
                results['strength-%d' % i]=[rval, pval]
                if is_print:
                    print('The correlation between individual difference and %d strength is %f(%f)' % (i, rval, pval))
            rval, pval,xrand=spatial_corr(long_percent,dv,pro_mat,1000)
            results['percent']=[rval, pval]
            if is_print:
                print('The correlation between individual difference and percent of long edges is %f(%f)\n' % (rval, pval))
            rval, pval,xrand=spatial_corr(short_percent,dv,pro_mat,1000)
            results['percent']=[rval, pval]
            if is_print:
                print('The correlation between individual difference and percent of short edges is %f(%f)\n' % (rval, pval))
        elif mode=='subnetwork':
            if is_print:
                print('---------------Results of correlation between individual difference with edges of different  length in subnetwork-------------------')
            yeo_name=['Visual', 'Somatomotor', 'Dorsal attention', 'Ventral attention', 'Limbic', 'Frontoparietal', 'Default','Subcortical']
            yeo_data=pd.read_csv(path+'/data/atlas/bna/subregion_func_network_Yeo_updated.csv')
            yeo_data=yeo_data.iloc[:,3].to_numpy()
            yeo_data=yeo_data[0:210]
            yeo_data=np.concatenate((yeo_data,np.ones(36)*8))
            for i in range(mean_degree.shape[0]):
                for j in range(8):
                    tmp_mat=pro_mat[np.where(yeo_data==j+1)[0]][:,np.where(yeo_data==j+1)[0]]
                    rval, pval,xrand=spatial_corr(mean_degree[i][yeo_data==j+1],dv[yeo_data==j+1],tmp_mat,1000)
                    results['degree-%d-%d' % (i,j)]=[rval, pval]
                    if is_print:
                        print('The correlation between individual difference and %d degree is %f(%f) in %s' % (i, rval, pval, yeo_name[j]))
            for i in range(mean_strength.shape[0]):
                for j in range(8):
                    tmp_mat=pro_mat[np.where(yeo_data==j+1)[0]][:,np.where(yeo_data==j+1)[0]]
                    rval, pval=stats.pearsonr(mean_strength[i][yeo_data==j+1],dv[yeo_data==j+1],tmp_mat,1000)
                    results['strength-%d-%d' % (i,j)]=[rval, pval]
                    if is_print:
                        print('The correlation between individual difference and %d strength is %f(%f) in %s' % (i, rval, pval, yeo_name[j]))
            for j in range(8):
                tmp_mat=pro_mat[np.where(yeo_data==j+1)[0]][:,np.where(yeo_data==j+1)[0]]
                rval, pval=stats.pearsonr(long_percent[yeo_data==j+1],dv[yeo_data==j+1],tmp_mat,1000)
                results['percent-%d' % j]=[rval, pval]
                if is_print:
                    print('The correlation between individual difference and percent of long edges is %f(%f) in %s' % (rval, pval, yeo_name[j]))
                if j==7:
                    if is_print:
                        print('\n')
    return results, mean_strength, short_percent

    
   

def degree_dis_batch(nets,len_nets,threshold):
    nodes_num=246
    degree=np.zeros((threshold,len_nets.shape[0],nodes_num))
    edges=np.array([])
    for net in len_nets:
        net=net.flatten()
        edges=np.concatenate((edges, net),axis=0)
    edges=edges[edges>0]
    thres=[]
    for i in range(threshold-1):
        thres.append(np.percentile(edges,100*(i+1)/threshold))
    for i in range(len_nets.shape[0]):
        len_net=len_nets[i]
        weight_net=nets[i]
        for j in range(nodes_num):
            len_row=len_net[j,:]
            weight_row=weight_net[j,:]
            degree_row=np.copy(weight_row)
            degree_row[degree_row>0]=1
            for k in range(threshold):
                if k==0:
                    degree[k,i,j]=np.nansum(degree_row[len_row<thres[k]])
                elif k<threshold-1:
                    degree[k,i,j]=np.nansum(degree_row[np.bitwise_and(len_row<thres[k],len_row>=thres[k-1])])
                else:
                    degree[k,i,j]=np.nansum(degree_row[len_row>=thres[k-1]])
    mean_degree=np.mean(degree,axis=1)
    long_percent=np.nansum(degree[-1],axis=0)/np.nansum(np.nansum(degree,axis=0),axis=0)
    short_percent=1-long_percent
    return mean_degree, short_percent, long_percent

def spatial_corr(xmap,ymap,pro_mat,repeatn=1000):
    rval, pval, xrand=sc.autocorr(xmap,ymap,pro_mat,repeatn=repeatn,method='spearman')
    return rval, pval, xrand

def gene_corr(genexp,dv):
    nifile=path+'/data/atlas/bna/BN_Atlas_246_1mm.nii.gz'
    center=sc.nii_center(nifile)
    pro_mat=sc.calc_pro_mat(center)
    # pro_mat=pro_mat[0:246:2,0:246:2]
    rval, pval, xrand=spatial_corr(dv,genexp,pro_mat)
    return rval, pval, xrand

def dev_gene_corr(dev_genexp,dev_dv,dev_dv_rand):
    dev_genexp=dev_genexp[:,dev_dv>0]
    dev_dv_rand=dev_dv_rand[:,dev_dv>0]
    dev_dv=dev_dv[dev_dv>0]
    dev_dv=dev_dv[dev_dv>0]
    for i in range(dev_genexp.shape[0]):
        gene_score=np.squeeze(dev_genexp[i])
        true_r=stats.spearmanr(gene_score,dev_dv)[0]
        null_r=[]
        for j in range(1000):
            null_r.append(stats.spearmanr(gene_score,dev_dv_rand[j])[0])
        null_r=np.asarray(null_r)
        p=np.where(true_r<null_r)[0].shape[0]/1000
        print(true_r, p)

def dv_age_wind(dvs,ages, model):
    rvals=np.zeros(dvs.shape[1])
    pvals=np.zeros(dvs.shape[1])
    for i in range(dvs.shape[1]):
        yval=dvs[:,i]
        xval=ages
        nanindex=np.isnan(yval)
        xval=xval[np.logical_not(nanindex)]
        yval=yval[np.logical_not(nanindex)]
        if model=='gam':
            gam=LinearGAM(s(0)).fit(xval,yval)
            pvals[i]=gam.statistics_['p_values'][0]
            rvals[i]=gam.statistics_['pseudo_r2']['explained_deviance']
        elif model=='spearman':
            rvals[i],pvals[i]=stats.spearmanr(xval,yval)
        elif model=='pearson':
            rvals[i],pvals[i]=stats.pearsonr(xval,yval)
    thre=rt_stat.multiple_comparison_correction.fdr_bh(pvals)
    return rvals, pvals, thre

def yeo_stat_wind(dvs):
    dv_yeos=np.zeros((dvs.shape[0],8))
    for i in range(dvs.shape[0]):
        dv_yeos[i]=yeo_stat(dvs[i],is_print=False)
    return dv_yeos