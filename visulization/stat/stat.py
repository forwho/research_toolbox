import numpy as np
import pandas as pd
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

def roc_plot(labels, values, pos_label, sample_rate):
    fpr, tpr, thresholds = metrics.roc_curve(labels,values,pos_label=pos_label)
    interval=int(1/sample_rate)
    index=np.arange(0,fpr.shape[0],interval)
    if index[-1]!=fpr.shape[0]-1:
        np.concatenate((index,np.array([fpr.shape[0]-1])))
    custom_params = {"axes.spines.right": False, "axes.spines.top": False, "xtick.direction": 'in', "ytick.direction":'in'}
    sns.set_theme(style="ticks", font="Arial", font_scale=2,rc=custom_params)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_xlim(-0.01,1.01)
    ax.set_ylim(-0.01,1.01)
    plt.plot(fpr[index],tpr[index],linewidth=2,marker='s',markersize=5,markeredgewidth=2,markerfacecolor='white',color=plt.get_cmap('Set1')(0),alpha=1)
    plt.plot([0,1],[0,1],color='black',alpha=1)

def corr_plot(xval,yval,xlabel,ylabel,order,scatter_kws, line_kws,filename,xlim=None,ylim=None,aspect=1.2):
    sns.set_theme(style="ticks", font="Arial")
    facet_kws={}
    if xlim!=None:
        facet_kws['xlim']=xlim
    if ylim!=None:
        facet_kws['ylim']=ylim
    data=pd.DataFrame({xlabel:xval,ylabel:yval})
    fig=sns.lmplot(x=xlabel, y=ylabel,data=data,order=order,scatter_kws=scatter_kws,line_kws=line_kws,facet_kws=facet_kws,aspect=aspect,x_jitter=0)
    if filename!='':
        fig.savefig(filename,dpi=300)

def hex_corr_plot(xval,yval,color,linewidth,ticksize,tickfamily,filename):
    b, a = np.polyfit(xval, yval, deg=1)
    df=pd.DataFrame({'x':xval,'y':yval})
    # df.plot.hexbin(x='x',y='y',gridsize=20)
    fig, ax=plt.subplots(figsize=(10,10))
    plt.hexbin(xval,yval,gridsize=20,cmap='Greys')
    plt.plot(xval,a+b*xval,color=color,lw=linewidth)
    locs, labels=plt.xticks(fontsize=ticksize,fontfamily=tickfamily)
    locs, labels=plt.yticks(fontsize=ticksize,fontfamily=tickfamily)
    if filename!='':
        fig.savefig(filename,dpi=300)

def bar_plot(x, y, color,is_mean,filename):
    sns.set_theme(style="ticks", font="Arial")
    fig, ax = plt.subplots()
    ax.bar(x,y,color=color)
    if is_mean:
        plt.axline((0,np.mean(y)),slope=0,linestyle='--')
    fig.savefig(filename,dpi=300)

def bibar_plot(x,y1,y2,color1,color2,filename):
    fig, ax = plt.subplots(figsize=(8,6),ncols=2)
    ax[0].invert_xaxis()
    ax[0].barh(x,y1,color=color1,height=0.5)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[1].barh(x,y2,color=color2,height=0.5)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    plt.yticks([])
    plt.subplots_adjust(wspace=0, top=0.85, bottom=0.1, left=0.18, right=0.95)
    if filename!='':
       fig.savefig(filename,dpi=300)


def multi_corr(xvals,yvals,labels,order,filename,scatter_kws, line_kws,palette=None, xlim=None,ylim=None,aspect=1.2,x_jitter=0,y_jitter=0):
    sns.set_theme(style="ticks", font="Arial")
    all_xval=np.array([])
    all_yval=np.array([])
    all_subnet=np.array([])
    facet_kws={}
    if xlim!=None:
        facet_kws['xlim']=xlim
    if ylim!=None:
        facet_kws['ylim']=ylim
    for i in range(len(labels)):
        nanindex=np.isnan(xvals[i])
        xval=xvals[i][np.logical_not(nanindex)]
        yval=yvals[i][np.logical_not(nanindex)]
        all_xval=np.concatenate((all_xval,xval))
        all_yval=np.concatenate((all_yval,yval))
        subnet=np.ones(xval.shape)*i
        all_subnet=np.concatenate((all_subnet,subnet))
    plot_data=pd.DataFrame({'xval':all_xval,'yval':all_yval,'subnet':all_subnet})
    fig=sns.lmplot(x='xval',y='yval',data=plot_data,order=order,hue='subnet',scatter_kws=scatter_kws,line_kws=line_kws,palette=palette,facet_kws=facet_kws,aspect=aspect,x_jitter=x_jitter,y_jitter=y_jitter)
    if filename!='':
        fig.savefig(filename,dpi=300)

def _label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)

def over_den(xval,yval,filename, palette=None, aspect=15, height=0.5):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)},font_scale=2)
    new_y=[]
    for i in range(len(xval)):
        new_y=new_y+[yval[i]]*xval[i].shape[0]
    xval=np.concatenate(xval,axis=0)
    df=pd.DataFrame({'x':xval,'y':new_y})
    g = sns.FacetGrid(df, row="y", hue="y", aspect=aspect, height=height, palette=palette)
    g.map(sns.kdeplot, "x",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=.5)

    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
    g.map(_label, "x")
    g.figure.subplots_adjust(hspace=-.25)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    if filename!='':
        g.savefig(filename,dpi=300)

def gene_visu(gene_names,gene_weights):
    pass

def heat_plot(rvals,filename):
    sns.set_theme(style="white")
    mask = np.triu(np.ones_like(rvals, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap=mpl.colormaps['turbo']
    sns.heatmap(rvals, mask=mask, cmap=cmap, vmax=1, 
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    f.savefig(filename,dpi=300)




