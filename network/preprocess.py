import numpy as np

def filter_nets(nets,thre):
    new_nets=np.copy(nets)
    new_nets[new_nets<thre]=0
    return new_nets

def vecs2mats(vecs,nodes_num):
    mats=[]
    for vec in vecs:
        mat=np.zeros((nodes_num,nodes_num))
        mat[np.triu_indices_from(mat,k=1)]=vec
        mat=mat+mat.T
        mats.append(mat)
    mats=np.asarray(mats)
    return mats

def fisher_z(nets):
    nets=0.5*np.log((1+nets)/(1-nets))
    return nets

def z_score(nets):
    new_nets=[]
    for net in nets:
        tmp_net=np.copy(net)
        mean_value=np.mean(tmp_net[tmp_net>0])
        std_value=np.std(tmp_net[tmp_net>0])
        new_net=np.zeros(tmp_net.shape)
        nonzero_index=tmp_net>0
        tmp_array=tmp_net[nonzero_index]
        tmp_array=(tmp_array-mean_value)/std_value
        new_net[nonzero_index]=tmp_array
        new_nets.append(new_net)
    new_nets=np.array(new_nets)
    return new_nets