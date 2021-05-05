#!/usr/bin/env python
# -*- coding: utf-8 -*-

from astropy.table import Table
from glob import glob
import os
from pathlib import Path
import matplotlib.pylab as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pandas as pd
import sys

from astropy.io import fits


def read_cluster(fpath=None):
    with fits.open(fpath) as data:
        cluster_df = pd.DataFrame(data[0].data)
    cluster_ar = cluster_df.to_numpy()
    return cluster_ar

def plot_cluster(fpath=None,save_dir=None,addBackground=True,circle=True,instrument='eROSITA'):
    cluster_ar = read_cluster(fpath=fpath)
    
    plt.figure(figsize=(4,4))
    file_name = Path(fpath).stem
    cluster_row = clusterList[clusterList['id']==int(file_name)]
    log_m = np.log10(cluster_row['M500_msolh'])[0]
    
    if circle:
        r_pixel = int(cluster_row['R500_pixel'][0])
        circle = plt.Circle((mid_pixel, mid_pixel), r_pixel, color="red",fill=False,zorder=2000)
        ax.add_patch(circle)
    plt.title(r'$\log\left(\frac{M_\mathrm{500c}}{h^{-1}\,M_\odot}\right) = $'+'{:.2f}'.format(log_m))
    
    cmap = mpl.cm.viridis
    lam = 0.1133929878
    noise = np.random.poisson(lam=lam, size=cluster.shape)

    plt.imshow(noise,cmap=cmap)
    im = plt.imshow(np.log10(cluster),cmap=cmap,interpolation='none')

    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.12)

    plt.colorbar(im, cax=cax)
    ax.set_xticks(np.linspace(0,,7))
    ax.set_yticks(np.linspace(0,384,7))

    ax.set_xlabel("x"), ax.set_ylabel("y")
    ax.set_facecolor('#DADADA')
    plt.tight_layout()
    #plt.savefig('../figs/eROSITA/' + file_name + '.png',dpi=250,bbox_inches='tight')

    ax.invert_yaxis()

    plt.close()
    
    return None

def plot_cluster_dir(source_dir=None,save_dir=None):
    fpath_arr = glob(source_dir + "*.fits")
    
    print("\nplotting " + source_dir + " --> " + save_dir)
    dir_size = str(len(fpath_arr))
    print(dir_size + " total object(s).\n")
    
    for idx, fpath in enumerate(fpath_arr):
        print("{:.2f} complete...".format(idx*100/dir_size), end='\r')
        plot_cluster(fpath=fpath,save_dir=save_dir)
    print("Done!\n")
    return None

def shift_cluster(cluster):
    r = int(cluster['R500_pixel'][0])
    mid = 192 #center pixel index of 384x384 image
    
    x = np.random.randint(low=-1*(mid+1*r)/2,high=(mid+r)/2,size=1)[0]
    y = np.random.randint(low=-1*r,high=r,size=1)[0]

    xi = mid + x
    yi = mid + y
    
    cluster_y_shift = np.roll(cluster_row,shift=y_shift,axis=0)
    cluster_xy_shift = np.roll(cluster_y_shift,shift=x_shift,axis=1)
    
    return cluster_xy_shift
    
    


def plot_cluster_trans():

if __name__ == "__main__":
    
    source_dir = sys.argv[1]
    save_dir = sys.argv[2]
    if (not source_dir.endswith('/')) or (not save_dir.endswith('/')):
        raise NotADirectoryError()
    else:
        plot_cluster_dir(source_dir=source_dir,save_dir=save_dir)

