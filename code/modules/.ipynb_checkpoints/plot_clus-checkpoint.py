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

def plot_cluster(fpath=None,save_dir=None,addBackground=True,instrument='eROSITA'):
    cluster_ar = read_cluster(fpath=fpath)

    plt.figure(figsize=(5,5))
    file_name = Path(fpath).stem

    plt.title(file_name)

    cmap = mpl.cm.viridis
    vmin,vmax = 0,3
    if instrument == 'eROSITA':
        #inputdir = '/n/home01/mntampaka/scratch/Magneticum_no_background/'
        detectorL = 1.03*60. #arcmin
        pixelL = 0.1609 #arcmin
        lam = 0.1133929878
        
    if addbackground == True:
        cluster_ar += np.random.poisson(lam=lam, size=cluster_ar.shape)
    im = plt.imshow(np.log10(cluster_ar), vmin=vmin,vmax=vmax,cmap=cmap,interpolation='None')

    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)

    plt.colorbar(im, cax=cax)

    ax.set_xlabel("pixel x"), ax.set_ylabel("pixel y")
    ax.set_facecolor('#DADADA')
    plt.tight_layout()

    plt.savefig(save_dir + file_name + '.png',bbox_inches='tight',dpi=200)
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

if __name__ == "__main__":
    
    source_dir = sys.argv[1]
    save_dir = sys.argv[2]
    if (not source_dir.endswith('/')) or (not save_dir.endswith('/')):
        raise NotADirectoryError()
    else:
        plot_cluster_dir(source_dir=source_dir,save_dir=save_dir)

