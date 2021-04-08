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

"""
def read_cluster(fpath=None):
    dat = Table.read(fpath, format='fits')
    cluster_df = dat.to_pandas()
    return cluster_df
"""

def read_cluster(fpath=None):
    with fits.open(fpath) as data:
        df = pd.DataFrame(data[0].data)
        print()
        print(df)
        print()
        print(df.columns)
    return df

def plot_cluster(fpath=None,save_dir=None):
    cluster_df = read_cluster(fpath=fpath)

    phi = cluster_df.RA * (np.pi/180)
    rho = np.abs(cluster_df.DEC-90) * (np.pi/180)
    x,y = rho * np.cos(phi), rho * np.sin(phi)
    
    plt.figure(figsize=(5,5))

    hist, xedges, yedges = np.histogram2d(x=x, y=y, bins=(74,74))
    cmap = mpl.cm.viridis
    im = plt.imshow(np.log10(hist), cmap=cmap)

    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    plt.colorbar(im, cax=cax)

    ax.set_xlabel("x"), ax.set_ylabel("y")
    ax.set_facecolor('#DADADA')
    plt.tight_layout()
    
    file_name = Path(fpath).stem
    
    plt.title(file_name)
    plt.savefig(save_dir + file_name + '.png',bbox_inches='tight',dpi=200)

    plt.close()
    return None

"""

"""
def plot_cluster_dir(source_dir=None,save_dir=None):
    fpath_arr = glob(source_dir + "*.fits")
    
    for fpath in fpath_arr:
        plot_cluster(fpath=fpath,save_dir=save_dir)
    return None

if __name__ == "__main__":
    
    source_dir = sys.argv[1]
    save_dir = sys.argv[2]
    
    if (not source_dir.endswith('/')) or (not save_dir.endswith('/')):
        raise NotADirectoryError()
    else:
        plot_cluster_dir(source_dir=source_dir,save_dir=save_dir)

