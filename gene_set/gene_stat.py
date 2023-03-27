import os
from research_toolbox.gene_set.gene_preprocess import label2entrez
from scipy.io import loadmat
import pandas as pd
import numpy as np
import scipy
from sklearn.cross_decomposition import PLSRegression
import pickle
import functools
from research_toolbox.rt_stat.multiple_comparison_correction import fdr_bh
path='%s/../' % os.path.dirname(os.path.abspath(__file__))

def extract_exp_tissue(gene_set,exp,result):
    if gene_set=='brain':
        brain_genes=pd.read_csv(path+'/data/gene/Background_genes_null_brain.csv')
        tissue_genes=brain_genes['symbols'][brain_genes['brain > body (default)']==1]
    elif gene_set=='har-brain':
        brain_genes=pd.read_csv(path+'/data/gene/har-brain.txt',index_col=False)
        tissue_genes=brain_genes['Description']
    new_exp=exp.loc[:,exp.columns.isin(tissue_genes)]
    new_exp.to_csv(result)

def go_cate_weights(gene_weights,lthre,uthre,gene_thre):
    '''
    gene_weights: DataFrame n*2, the 1st column is the gene symbols, the 2nd column is the weight
    '''
    gene_weights.columns=['gene_symbol','weight']
    anno=pd.read_csv(path+'/data/gene/go_data_2022_07_02_name_p.csv',index_col=False)
    go_table=anno
    go_table=go_table[np.bitwise_and(go_table['Size']>=lthre,go_table['Size']<=uthre)]
    pos_neg_gene_weights=go_genes_filter(gene_weights,gene_thre)
    colname=['pos_weight','neg_weight']
    k=0
    for gene_weights in pos_neg_gene_weights:
        cate_weights=[]
        for i in range(len(go_table)):
            genes=go_table.iloc[i,3][2:-1].replace('\'','').split(', ')
            if len(genes)==0:
                weights=0
            elif np.any(gene_weights['gene_symbol'].isin(genes)):
                weights=gene_weights['weight'][gene_weights['gene_symbol'].isin(genes)]
            else:
                weights=0
            cate_weights.append(np.mean(weights))
        go_table[colname[k]]=cate_weights
        k+=1
    return go_table

def cate_pvalues(go_weights, perm_pos_weights, perm_neg_weights):
    pos_pval=[(np.where(perm_pos_weights[:,k]>=go_weights['pos_weight'].iloc[k])[0].shape[0]+1)/perm_pos_weights.shape[0] for k in range(len(go_weights))]
    neg_pval=[(np.where(perm_neg_weights[:,k]<=go_weights['neg_weight'].iloc[k])[0].shape[0]+1)/perm_neg_weights.shape[0] for k in range(len(go_weights))]
    go_weights['pos_pval']=pos_pval
    pos_thre=fdr_bh(np.asarray(pos_pval),0.05)
    go_weights['neg_pval']=neg_pval
    neg_thre=fdr_bh(np.asarray(neg_pval),0.05)
    return go_weights, pos_thre, neg_thre

def gene_cell(gene_weights,gene_thre,repeat_num=10000):
    cell_data=pd.read_csv(path+'/data/gene/celltypes_PSP.csv')
    cells=np.unique(cell_data['class'].to_numpy())
    cells.sort()
    gene_weights.columns=['gene_symbol','weight']
    pos_neg_gene_weights=go_genes_filter(gene_weights,gene_thre)
    colname=['pos_ratio','neg_ratio']
    k=0
    cell_table=pd.DataFrame({'class':cells})
    gene_nums=[]
    for gene_weight in pos_neg_gene_weights:
        cells_ratio=[]
        for i in range(len(cells)):
            intersect_genes_num=np.where(cell_data['gene'][cell_data['class']==cells[i]].isin(gene_weight['gene_symbol']).to_numpy())[0].shape[0]
            ratio=intersect_genes_num/len(gene_weight['weight'])
            cells_ratio.append(ratio)
        cell_table[colname[k]]=cells_ratio
        gene_nums.append(len(gene_weight['weight']))
        k+=1
    if repeat_num>0:
        pos_genes=np.zeros((repeat_num,cells.shape[0]))
        neg_genes=np.zeros((repeat_num,cells.shape[0]))
        for i in range(repeat_num):
            for j in range(len(cells)):
                pos_sample_genes=np.random.choice(gene_weights['gene_symbol'].to_numpy(), size=gene_nums[0], replace=False, p=None)
                neg_sample_genes=np.random.choice(gene_weights['gene_symbol'].to_numpy(), size=gene_nums[1], replace=False, p=None)
                pos_intersect_genes_num=np.where(cell_data['gene'][cell_data['class']==cells[j]].isin(pos_sample_genes).to_numpy())[0].shape[0]
                pos_ratio=pos_intersect_genes_num/gene_nums[0]
                neg_intersect_genes_num=np.where(cell_data['gene'][cell_data['class']==cells[j]].isin(neg_sample_genes).to_numpy())[0].shape[0]
                neg_ratio=neg_intersect_genes_num/gene_nums[1]
                pos_genes[i,j]=pos_ratio
                neg_genes[i,j]=neg_ratio
        pos_pval=[(np.where(pos_genes[:,k]>cell_table['pos_ratio'].iloc[k])[0].shape[0]+1)/pos_genes.shape[0] for k in range(len(cell_table))]
        neg_pval=[(np.where(neg_genes[:,k]>cell_table['neg_ratio'].iloc[k])[0].shape[0]+1)/pos_genes.shape[0] for k in range(len(cell_table))]
        cell_table['pos_pval']=pos_pval
        cell_table['neg_pval']=neg_pval

    return cell_table


def go_genes_filter(gene_weights,thre):
    gene_weights.columns=['gene_symbol','weight']
    pos_gene_weights=gene_weights[gene_weights['weight']>0].sort_values(by='weight',ascending=False)
    neg_gene_weights=gene_weights[gene_weights['weight']<0].sort_values(by='weight')
    pos_gene_weights=pos_gene_weights[pos_gene_weights['weight']>pos_gene_weights['weight'].iloc[int((len(pos_gene_weights)-1)*thre)]]
    neg_gene_weights=neg_gene_weights[neg_gene_weights['weight']<neg_gene_weights['weight'].iloc[int((len(neg_gene_weights)-1)*thre)]]
    return pos_gene_weights, neg_gene_weights

def gene_pls(pheno,genexp,n_components=1):
    pls1 = PLSRegression(n_components=n_components,scale=False)
    genexp=scipy.stats.zscore(genexp,axis=0)
    pheno=scipy.stats.zscore(pheno)
    pls1.fit(genexp, pheno)
    r2=pls1.score(genexp,pheno)
    return pls1.x_rotations_, pls1.x_scores_, r2, pls1.y_loadings_

def gene_pls_boot(pheno,genexp,repeatn,samples=None,n_components=1):
    xrot, xscores, r2, y_loadings=gene_pls(pheno,genexp,n_components)
    index=np.arange(0,pheno.shape[0])
    boot_xrot=[]
    if np.all(samples!=None):
        for sample in samples:
            tmp_pheno=pheno[sample]
            tmp_exp=genexp[sample]
            tmp_xrot, tmp_xscores, tmp_r2, tmp_y_loadings=gene_pls(tmp_pheno,tmp_exp,n_components=1)
            boot_xrot.append(tmp_xrot)
        std_rot=np.std(np.asarray(boot_xrot),axis=0)
    elif repeatn>0:
        samples=[]
        for i in range(repeatn):
            sample=np.random.choice(index,pheno.shape[0])
            samples.append(sample)
            tmp_pheno=pheno[sample]
            tmp_exp=genexp[sample]
            tmp_xrot, tmp_xscores, tmp_r2, tmp_y_loadings=gene_pls(tmp_pheno,tmp_exp,n_components=1)
            boot_xrot.append(tmp_xrot)
        std_rot=np.std(np.asarray(boot_xrot),axis=0)
    else:
        std_rot=1
    zxrot=xrot/std_rot
    return zxrot, xscores, r2, np.asarray(samples), xrot, y_loadings

def gene_corr_boot(pheno,genexp,repeatn,samples=None):
    true_weight=list(map(lambda x: scipy.stats.spearmanr(pheno,x)[0],genexp.T))
    index=np.arange(0,pheno.shape[0])
    boot_weight=[]
    if np.all(samples!=None):
        for sample in samples:
            tmp_pheno=pheno[sample]
            tmp_exp=genexp[sample]
            tmp_weight=list(map(lambda x: scipy.stats.spearmanr(tmp_pheno,x)[0],tmp_exp.T))
            boot_weight.append(tmp_weight)
    else:
        samples=[]
        for i in range(repeatn):
            sample=np.random.choice(index,pheno.shape[0])
            samples.append(sample)
            tmp_pheno=pheno[sample]
            tmp_exp=genexp[sample]
            tmp_weight=list(map(lambda x: scipy.stats.spearmanr(tmp_pheno,x)[0],tmp_exp.T))
            boot_weight.append(tmp_weight)
    boot_weight=np.asarray(boot_weight)
    zweight=true_weight/np.std(boot_weight,axis=0)
    return zweight

def gene_corr_boot2(pheno,genexp):
    true_weight=np.asarray(list(map(lambda x: scipy.stats.spearmanr(pheno,x)[0],genexp.T)))
    zweight=0.5*np.log((1+true_weight)/(1-true_weight))
    return zweight

def gene_filter_set(genexp):
    data=np.load(path+'/data/gene/go_data_2022_03_23_p.npy',allow_pickle=True)
    gene_list=genexp.columns[1:]
    all_genes=functools.reduce(set.union,list(data[:,2]))
    isin_flag=[gene in all_genes for gene in gene_list]
    labels=genexp.iloc[:,0]
    exp_data=genexp.iloc[:,1:]
    exp_data=exp_data.iloc[:,np.asarray(isin_flag)]
    genexp=pd.concat([labels,exp_data],axis=1)

    gene_list=genexp.columns[1:]
    cate_flag=np.zeros((data.shape[0],1))
    for i in range(data.shape[0]):
        set1=data[i,2]
        set2=set(gene_list)
        if len(set1 & set2)>0:
            cate_flag=1
    data=data[cate_flag==1,:]
    data=np.squeeze(data)
    return genexp,data


def gene_stat_run(gene_data,dv,samples,weight_method='pls'):
    gene_exp=gene_data.to_numpy()[:,0:]
    dv=dv[0:dv.shape[0]:2]
    if weight_method=='pls':
        zxrot, xscores, r2, samples, xrot, yloadings=gene_pls_boot(dv,gene_exp,10000, samples)
    elif weight_method=='corr':
        zxrot, xscores, r2, samples, xrot, yloadings=gene_pls_boot(dv,gene_exp,0)
        weights=gene_corr_boot(dv,gene_exp,10000, samples)
    flag=1
    if yloadings[np.abs(yloadings)==np.max(np.abs(yloadings))][0] < 0:
        flag=-1
    xscores=flag*xscores
    xrot=flag*np.squeeze(xrot[:,np.where(np.abs(yloadings)==np.max(np.abs(yloadings)))[0]])
    gene_list=gene_data.columns[0:]
    if weight_method=='pls':
        gene_weights=pd.DataFrame({'gene_symbol':gene_list,'weight_1':flag*np.squeeze(zxrot[:,np.where(np.abs(yloadings)==np.max(np.abs(yloadings)))[0]])})
    elif weight_method=='corr':
        gene_weights=pd.DataFrame({'gene_symbol':gene_list,'weight_1':weights})
    return gene_weights, xscores, r2, xrot, yloadings

    



if __name__=='__main__':
    # gene_data=pd.read_csv(data_path+'/indi-data/results/batch_results/gene_stat/abagen_expression_filter.csv')
    # dv=np.load(data_path+'/indi-data/results/batch_results/dvs/count_30_68_nofilter_combat_reg_dv.npy')
    # dv=dv[0:246:2]
    # x_rotations_, x_scores_, r2, y_loadings=gene_pls(dv,gene_data)
    # print(r2,x_rotations_.shape, np.square(y_loadings)/np.sum(np.square(y_loadings))*r2,np.max(np.square(y_loadings)/np.sum(np.square(y_loadings))*r2))
    # gene_filter_set(gene_data)
    # data=pd.read_csv(r'G:\indi-data\results\batch_results\gene_stat\gene_weights_0.csv')
    # go_cate_weights2(data,0.5)
    pass