import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy.stats import ttest_ind
from rt_stat.multiple_comparison_correction import fdr_bh

path='%s/../' % os.path.dirname(os.path.abspath(__file__))

def go_preprocess1():
    data=pd.read_csv(path+'/data/gene/goa_human_20220702.gaf',delimiter='\t',header=None,index_col=False,names=['DB','DB Object ID', 'DB Object Symbol',   'Qualifier','GO ID','DB:Reference','Evidence Code', 'With(or)From','Aspect','DB Object Name','DB Object Synonym','DB Object Type','Taxon','Date','Assigned By','Annotation Extension'])
    data=data[~(data['Qualifier'].str.contains('NOT')) & (data['Evidence Code']!='ND') & (data['Taxon'].str.contains('taxon:9606')) & (~(data['DB Object Symbol'].isna()))]

    unique_id=data['GO ID'].unique()
    aspect=[data['Aspect'][data['GO ID']==x].iloc[0] for x in unique_id]
    genes=[set(data['DB Object Symbol'][data['GO ID']==x].to_list()) for x in unique_id]
    go_data=pd.DataFrame({'GO ID': unique_id,'Aspect': aspect,'Genes':genes})
    # with open('/shared/su_group/User/md1weih/indi_difference/go_data_2022_03_23.txt','wb') as f:
    #     pickle.dump(go_data,f)
    go_data.to_csv(path+'/data/gene/go_data_2022_07_02.csv',index=False)

def go_preprocess2():
    go_data=pd.read_csv(path+'/data/gene/go_data_2022_07_02.csv',index_col=False)
    go_data_p=go_data[go_data['Aspect']=='P']
    go_data_f=go_data[go_data['Aspect']=='F']
    go_data_c=go_data[go_data['Aspect']=='C']
    go_data_p.to_csv(path+'/data/gene/go_data_2022_07_02_p.csv',index=False)
    go_data_f.to_csv(path+'/data/gene/go_data_2022_07_02_f.csv',index=False)
    go_data_c.to_csv(path+'/data/gene/go_data_2022_07_02_c.csv',index=False)


def go_preprocess3():
    anno=pd.read_csv(path+'/data/gene/go_data_2022_07_02_p.csv',index_col=False)
    data=loadmat(path+'/data/gene/GOTerms_BP.mat',variable_names=['gotable'])
    goname=[data['gotable'][:,1][data['gotable'][:,0]==anno['GO ID'][i]] for i in range(len(anno))]
    anno['GO Name']=goname
    # go_table=pd.DataFrame({'IDLabel':[i[0] for i in data['hie_go_anno'][:,0]],'Name':[i[0] for i in data['hie_go_anno'][:,1]], 'ID':[i[0,0] for i in data['hie_go_anno'][:,2]],'Size':[i[0,0] for i in data['hie_go_anno'][:,3]],'Annotations':[np.squeeze(i) for i in data['hie_go_anno'][:,4]]})
    # go_name=[go_table['Name'][go_table['IDLabel']==x].str.split(',,').iloc[0][0] if len(go_table['Name'][go_table['IDLabel']==x])>0 else '' for x in anno[:,0]]
    # go_name=np.asarray(go_name)
    # go_name=go_name[:,np.newaxis]
    # anno=np.concatenate((anno,go_name),axis=1)
    # np.save(path+'/data/gene/go_data_2022_03_23_p.npy',anno)
    # anno=pd.DataFrame(anno)
    anno.to_csv(path+'/data/gene/go_data_2022_07_02_name_p.csv',index=False)

def go_preprocess4():
    anno=pd.read_csv(path+'/data/gene/go_data_2022_07_02_name_p.csv',index_col=False)
    size=[]
    for i in range(len(anno)):
        genes=anno.iloc[i,3][2:-1].replace('\'','').split(', ')
        size.append(len(genes))
    anno['Size']=size
    anno.to_csv(path+'/data/gene/go_data_2022_07_02_name_p.csv',index=False)


def gtex_stat():
    data=pd.read_csv(path+'/data/gene/GTEx_Analysis_v6p_RNA-seq_RNA-SeQCv1.1.8_gene_median_rpkm.gct',delimiter='\t',index_col=False)
    nonBrainVal=data.loc[:,~data.columns.str.contains('Brain')].iloc[:,2:].to_numpy()
    BrainVal=data.loc[:,data.columns.str.contains('Brain')].to_numpy()
    # nonBrainVal[nonBrainVal==0]=np.nan
    # BrainVal[BrainVal==0]=np.nan
    rvals, pvals=ttest_ind(BrainVal,nonBrainVal,axis=1,nan_policy='omit')
    data['pvals']=pvals
    pvals=pvals[np.bitwise_not(np.isnan(pvals))]
    pthre=fdr_bh(pvals,thre=0.025)
    print(pthre)
    data.to_csv(path+'/data/gene/GTEx_Analysis_v6p_RNA-seq_RNA-SeQCv1.1.8_gene_median_rpkm_p.csv',index=False)
    index=np.bitwise_and(data['pvals']<pthre,rvals>0)
    index[index.isnull()]=False
    data=data[index]
    data.to_csv(path+'/data/gene/GTEx_Analysis_v6p_RNA-seq_RNA-SeQCv1.1.8_gene_median_rpkm_brain_p.csv',index=False)