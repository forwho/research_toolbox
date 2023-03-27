import numpy as np

def fdr_bh(p,thre=0.05):
    '''
    Control false discovery rate using Benjamini-Hochberg Procedure
    p: 1-dimensional p values array
    thre: false discovery rate

    return:
    thre: the threhold for p values. If thre equals 1 means no p value suvives after corrections.
    '''
    sort_p=p.copy()
    sort_p.sort()
    pthre=0
    for i in range(sort_p.shape[0]-1,-1,-1):
        if sort_p[i]<=thre*(i+1)/sort_p.shape[0]:
            pthre=sort_p[i]
            break
    return pthre
