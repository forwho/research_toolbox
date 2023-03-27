import os
import scipy
import pandas as pd
from scipy.io import loadmat
import numpy as np
from scipy.stats import spearmanr
from abagen.correct import normalize_expression
import abagen

path='%s/../' % os.path.dirname(os.path.abspath(__file__))

def gene_extract(atlas, probe_selection='rnaseq', sample_norm=None):
    if atlas=='bna':
        img='%s/data/atlas/bna/BN_Atlas_246_1mm_left.nii.gz' % path
        info='%s/data/atlas/bna/bna_info_left.csv' % path
    expression=abagen.get_expression_data(img, info, probe_selection=probe_selection,sample_norm=sample_norm,return_donors=True)
    return expression

def label2entrez(labels):
    gene_data=pd.read_csv(path+'/data/gene/reannotated.csv')
    gene_info=gene_data[['gene_symbol','entrez_id']][gene_data['gene_symbol'].isin(labels)].drop_duplicates(subset='gene_symbol')
    gene_info['entrez_id']=gene_info['entrez_id'].astype('int32')
    return gene_info


def gene_differential_stability(gene_exp):
    sub_num=gene_exp.shape[0]
    gene_num=gene_exp.shape[2]
    ds=[]
    for i in range(gene_num):
        rvals=[]
        for m in range(sub_num):
            for n in range(m+1,sub_num):
                vec1=np.squeeze(gene_exp[m,:,i])
                vec2=np.squeeze(gene_exp[n,:,i])
                nan_index=np.bitwise_or(np.isnan(vec1),np.isnan(vec2))
                vec1=vec1[np.bitwise_not(nan_index)]
                vec2=vec2[np.bitwise_not(nan_index)]
                result=spearmanr(vec1,vec2)
                rvals.append(result[0])
        rvals=np.asarray(rvals)
        ds.append(np.mean(rvals))
    return np.asarray(ds)

def develop_stand():
    exp=pd.read_csv(path+'/data/gene/development/expression_matrix.csv',header=None)
    exp=exp.loc[:,1:].to_numpy()
    col_data=pd.read_csv(path+'/data/gene/development/columns_metadata.csv')
    donors=np.unique(col_data['donor_id'].to_numpy())
    exp[exp==0]=np.nan
    for donor in donors:
        data=exp[:,col_data['donor_id']==donor].T
        data=normalize_expression(pd.DataFrame(data)).to_numpy()
        data=data.T
        exp[:,col_data['donor_id']==donor]=data
    return exp

def develop_extract():
    exp=develop_stand()
    col_data=pd.read_csv(path+'/data/gene/development/columns_metadata.csv')
    group=np.concatenate((np.ones(237),np.ones(59)*2,np.ones(71)*3,np.ones(64)*4,np.ones(93)*5))
    strucs=np.load(path+'/data/gene/development/strucs.npy',allow_pickle=True)
    # strucs=np.unique(col_data['structure_acronym'].to_numpy())
    exp_group=np.zeros((5,strucs.shape[0],len(exp)))
    for i in range(1,6):
        k=0
        for struc in strucs:
            data=exp[:,np.bitwise_and(group==i,col_data['structure_acronym']==struc)]
            data[data==0]=np.nan
            exp_group[i-1,k,:]=np.nanmean(data,axis=1)
            k+=1
    np.save(path+'/data/gene/development/exp_group.npy',exp_group)
    # np.save(path+'/data/gene/development/strucs.npy',strucs)
    return exp_group

def develop_gene_score(xrot_pd):
    '''
    xrot_pd has two columns, one is xrot, another is gene_symbol
    '''
    data=np.load(path+'/data/gene/development/exp_group.npy')
    data=np.delete(data,[2,4,6,10,12,15,17,18,22,23],axis=1)
    row_data=pd.read_csv(path+'/data/gene/development/rows_metadata.csv')
    dev_abagen=data[:,:,row_data['gene_symbol'].isin(xrot_pd['gene_symbol'].to_numpy())]
    columns_name=row_data['gene_symbol'][row_data['gene_symbol'].isin(xrot_pd['gene_symbol'].to_numpy())]

    nanindex=[np.any(np.isnan(dev_abagen[:,:,i])) for i in range(dev_abagen.shape[2])]

    columns_name=columns_name.loc[np.bitwise_not(np.asarray(nanindex))]
    dev_abagen=dev_abagen[:,:,np.bitwise_not(np.asarray(nanindex))]

    dev_loadings=[xrot_pd['xrot'][xrot_pd['gene_symbol']==symbol] for symbol in columns_name]
    gene_score=scipy.stats.zscore(dev_abagen,axis=1).dot(dev_loadings)
    return gene_score

def check_develop():
    loadings=pd.read_csv('G:/indi-data/results/batch_results/gene_stat/genes_weight_comp_1.csv')
    dev_data=np.load(path+'/data/gene/development/exp_group.npy')
    row_data=pd.read_csv(path+'/data/gene/development/rows_metadata.csv')
    dev_abagen=dev_data[:,:,row_data['gene_symbol'].isin(loadings['gene_symbol'].to_numpy())]
    strucs=np.load(path+'/data/gene/development/strucs.npy',allow_pickle=True)
    columns_name=row_data['gene_symbol'][row_data['gene_symbol'].isin(loadings['gene_symbol'].to_numpy())]
    for i in range(dev_abagen.shape[0]):
        tmp_data=pd.DataFrame(dev_abagen[i],index=strucs,columns=columns_name)
        tmp_data.to_csv('G:/indi-data/results/batch_results/gene_stat/development/develop_%d.csv' % i)

def check_expression_genes():
    exp=pd.read_csv(path+'/data/gene/development/abagen_expression_bna-left_rnaseq_filter.csv')
    gene_list=exp.columns[1:]
    data=loadmat(path+'/data/gene/hie_go_anno.mat',variable_names=['hie_go_anno'])
    go_table=pd.DataFrame({'IDLabel':[i[0] for i in data['hie_go_anno'][:,0]],'Name':[i[0] for i in data['hie_go_anno'][:,1]], 'ID':[i[0,0] for i in data['hie_go_anno'][:,2]],'Size':[i[0,0] for i in data['hie_go_anno'][:,3]],'Annotations':[np.squeeze(i) for i in data['hie_go_anno'][:,4]]})



if __name__=='__main__':
    #labels=['A1BG','A1BG-AS1','A2M','A2ML1','A3GALT2','A4GALT','AAAS']
    #print(label2entrez(labels))
    # subs=['9861','15697','15496','14380','12876','10021']
    # gene_exp=[]
    # for sub in subs:
    #     data=pd.read_csv(data_path+'/indi-data/results/batch_results/gene_stat/abagen_expression_bna-left_rnaseq_%s.csv' % sub)
    #     gene_exp.append(data.iloc[:,1:].to_numpy())
    # gene_exp=np.asarray(gene_exp)
    # ds=gene_differential_stability(gene_exp)
    # data=pd.read_csv(data_path+'/indi-data/results/batch_results/gene_stat/abagen_expression_bna-left_rnaseq.csv')
    # gene_list=data.columns[1:]
    # data=data.iloc[:,1:].to_numpy()
    # data=data[:,ds>=0.1]
    # # print(gene_list.shape)
    # gene_list=gene_list[ds>=0.1]
    # # print(data.shape,gene_list.shape)
    # data=pd.DataFrame(data,columns=gene_list)
    # data.to_csv(data_path+'/indi-data/results/batch_results/gene_stat/abagen_expression_bna-left_rnaseq_filter.csv')
    # data=pd.read_csv('D:\\indi-data\\results\\batch_results\\data_preprocess\\abagen_expression_filter.csv')
    # develop_extract()
    check_develop()
    pass

