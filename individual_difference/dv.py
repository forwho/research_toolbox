import numpy as np
from math import ceil
import multiprocessing as mp
from functools import reduce
from operator import add
import os

class indv_sp():
    def __init__(self,mats):
        self.mats=mats
    def indv_sp_run(self,i):
        '''
        mats: 3 dimensional array. The 1st dimension index networks
        return:
        dv_conn: 1 dimensional array. It's the individual difference of these networks.
        '''
        mats=self.mats
        num=mats.shape[0]
        sum_sim_vec=np.zeros((num-i-1,mats.shape[1]))
        k=0
        for j in range(i+1,num):
            mat1=mats[i]
            mat2=mats[j]
            sim_vec=_sim_conn(mat1,mat2)
            sum_sim_vec[k,:]=sim_vec
            k+=1
        return sum_sim_vec


def _sim_conn(mat1,mat2):
    '''
    mat1: 2 dimensional array
    mat2: 2 dimensional array

    return:
    sim_vec: 1 dimensional array, it's pearson correlation coefficients for between 2 networks for each nodes
    '''
    num=mat1.shape[0]
    sim_vec=np.zeros(num)
    for i in range(num):
        sim_vec[i]=np.corrcoef(mat1[i,:],mat2[i,:])[0,1]
    return sim_vec
    
def indv_conn_mp(mats):
    num=mats.shape[0]
    p=mp.Pool(os.cpu_count()-10)
    indv_sp_obj=indv_sp(mats)
    result=p.map(indv_sp_obj.indv_sp_run,range(num-1))
    sim_vecs=result
    dv_conn=1-np.nanmean(np.concatenate(sim_vecs,axis=0),axis=0)
    return dv_conn

def indv_conn(mats):
    '''
    mats: 3 dimensional array. The 1st dimension index networks

    return:
    dv_conn: 1 dimensional array. It's the individual difference of these networks.
    '''
    new_mats=np.copy(mats)
    num=mats.shape[0]
    sum_sim_vec=np.zeros((ceil(num*(num-1)/2),mats.shape[1]))
    k=0
    for i in range(num):
        for j in range(i+1,num):
            mat1=new_mats[i]
            mat2=new_mats[j]
            sim_vec=_sim_conn(mat1,mat2)
            sum_sim_vec[k,:]=sim_vec
            k+=1
    dv_conn=1-np.nanmean(sum_sim_vec,axis=0)
    return dv_conn

def intra_dv_conn(mats_pair,threshold=0):
    mats_pair[mats_pair<threshold]=0
    num=len(mats_pair)
    # sum_sim_vec=np.zeros(mats_pair[0][0].shape[0])
    sum_sim_vec=np.zeros((mats_pair.shape[0],mats_pair.shape[2]))
    for i in range(num):
        sim_vec=_sim_conn(mats_pair[i][0],mats_pair[i][1])
        sum_sim_vec[i]=sim_vec
    dv_conn=1-np.nanmean(sum_sim_vec,axis=0)
    return dv_conn

def batch_dv(mats,threshold=0,is_parallel=1):
    mats[mats<threshold]=0
    if is_parallel==1:
        dv_conn=indv_conn_mp(mats)
    else:
        dv_conn=indv_conn(mats)
    return dv_conn

def wind_data_gen(ages,mats,win_len,ith):
    '''
    ages: 1 dimensional array. The ages have been sorted.
    mats: 3 dimeansional array. The 1st dimension index networks.
    win_len: int scalar. The length of windows.
    ith: int scalar. The index of windows

    return:
    group_mats: 3 dimensional array. The mats of ith window
    group_ages: 1 dimensional array. The ages of ith window
    '''
    mats_shape=list(mats.shape)
    mats_shape[0]=win_len
    group_mats=np.zeros(mats_shape)
    group_ages=np.zeros(win_len)
    for i in range(ith,ith+win_len):
        group_mats[i-ith,:,:]=mats[i,:,:]
        group_ages[i-ith]=ages[i]
    return group_mats,group_ages

def wind_indv_conn(ages,mats,win_len,interval):
    '''
    ages: 1 dimensional array.
    mats: 3 dimeansiona array. The 1st dimension index networks.
    win_len: int scalar. The length of windows.

    return:
    all_dv_conn: 2 dimensional array. All the individual differences of networks of all the windows.
    mean_ages: 1 dimensional array. All the mean ages of all the windows.
    '''
    age_sort_index=np.argsort(ages)
    mats=mats[age_sort_index]
    ages=ages[age_sort_index]
    num=ages.shape[0]
    win_num=np.ceil((num-win_len)/interval)+1
    win_num=int(win_num)
    all_dv_conn=np.zeros((win_num,mats.shape[1]))
    mean_ages=np.zeros(win_num)
    for i in range(win_num):
        if num-i*interval>=win_len:
            start=i*interval
        else:
            start=num-win_len
        group_mats,group_ages=wind_data_gen(ages,mats,win_len,start)
        dv_conn=indv_conn(group_mats)
        all_dv_conn[i,:]=dv_conn
        mean_ages[i]=np.mean(group_ages)
    return all_dv_conn,mean_ages

if __name__=='__main__':
    mats=np.random.rand(3,4,4)
    # indv_sp_run(mats,0)
    indv_conn_mp(mats)