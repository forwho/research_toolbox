import numpy as np
from research_toolbox.rt_stat.group_tests import signtest
from research_toolbox.rt_stat.multiple_comparison_correction import fdr_bh

def nodal_metrics_calc(nets):
    net_num=nets.shape[0]
    nodal_num=nets.shape[1]
    degree=np.zeros((net_num,nodal_num))
    strength=np.zeros((net_num,nodal_num))
    k=0
    for i in range(net_num):
        net=np.copy(nets[i])
        strength[i,]=np.nansum(net,axis=1)
        net[net>0]=1
        degree[k,]=np.nansum(net,axis=1)
        k+=1
    return degree, strength

def get_spar(nets):
    spar=np.zeros(nets.shape[0])
    for i in range(nets.shape[0]):
        net=nets[i]
        spar[i]=net[net!=0].shape[0]/net.shape[0]/(net.shape[0]-1)
    return np.mean(spar)

def backbone_net(nets,pval,cor_method='fdr'):
    sign_test_h=np.zeros((nets.shape[1],nets.shape[2]))
    sign_test_p=np.ones((nets.shape[1],nets.shape[2]))
    sign_test_p=np.triu(sign_test_p)
    p_array=[]
    for i in range(nets.shape[1]):
        for j in range(i+1,nets.shape[2]):
            edges=nets[:,i,j].copy()
            # non-parameter sign test to get the p matrix
            z,sign_test_p[i,j]=signtest(edges,method='approximate',direction='right')
            p_array.append(sign_test_p[i,j])
    if cor_method=='fdr':
        pthre=fdr_bh(np.array(p_array),pval)
    else:
        pthre=pval/nets.shape[0]
    sign_test_p+=sign_test_p.T
    sign_test_h[sign_test_p<=pthre]=1
    ave_nets=np.nanmean(nets,axis=0)
    ave_nets[sign_test_h==0]=0
    return ave_nets, sign_test_h, sign_test_p,pthre