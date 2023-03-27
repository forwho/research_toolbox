from nilearn import plotting
import matplotlib.pyplot as plt
import numpy as np

def network_vis(edges,node_coords,fname,node_color='auto',node_size=10,edge_threshold=0.01):
    f=plt.figure(figsize=(6,6),dpi=300)
    plotting.plot_connectome(edges,node_coords,node_color,node_size,edge_threshold=edge_threshold,figure=f)
    f.savefig(fname,dpi=300)

