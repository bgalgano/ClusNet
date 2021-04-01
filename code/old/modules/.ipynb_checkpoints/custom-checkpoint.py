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

clusterList = np.load('../../data/eROSITA/clusterList.npy')
class Cluster:
    def __init__(self, fpath):
        self.fpath = fpath
        self.id = Path(fpath).stem
        
        with fits.open(fpath) as data:
            cluster_df = pd.DataFrame(data[0].data)

        self.image = cluster_df.to_numpy()
        
        meta = clusterList[clusterList['id']==int(self.id)]
        self.meta = meta
        
        self.M500 = np.log10(meta['M500_msolh'])[0]
        self.R500 = np.log10(meta['r500_kpch'])[0]
        self.Rpixel = np.log10(meta['R500_pixel'])[0]
        self.Tkev = np.log10(meta['T_kev'])[0]
        self.mid_pix = 192
        
        if type(self.meta) == np.ndarray:
            self.meta_col = meta.dtype.names
        else:
            self.meta_col = meta.columns()
        self.lam = 0.1133929878

    def __repr__(self):
        """
        print cluster metadata
        """
        return str(self.meta)
    
    def to_pandas(self):
        """
        convert metadata (as recarray) to pandas DataFrame
        """
        self.meta = pd.DataFrame(self.meta)
        return
    
    def add_noise(self):
        """
        add Poisson noise to cluster image matrix
        """
        self.image += np.random.poisson(lam=lam, size=cluster.shape)
        return
        
    def shift(self):
        """
        shift cluster randomly within bounds of image
        """
        r = self.Rpixel
        mid = self.mid_pixel #center pixel index of 384x384 image

        x = np.random.randint(low=-1*(mid+1*r)/2,high=(mid+r)/2,size=1)[0]
        y = np.random.randint(low=-1*r,high=r,size=1)[0]

        xi = mid + x
        yi = mid + y

        cluster_y_shift = np.roll(self.image,shift=y_shift,axis=0)
        self.image = np.roll(cluster_y_shift,shift=x_shift,axis=1)
        
        return 
    
    def plot(self,circle=True,savefig=False,spath='../figs/eROSITA/'):
        """
        plot cluster image with radius and mass information
        """
        mid_pixel = 192
        Rpixel = self.Rpixel
        cluster = self.image
        plt.figure(figsize=(4,4))
        cluster_id = self.id
        
        cluster_row = self.meta
        log_m = np.log10(cluster_row['M500_msolh'])[0]
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
        ax.set_xticks(np.linspace(0,384,7))
        ax.set_yticks(np.linspace(0,384,7))

        ax.set_xlabel("x"), ax.set_ylabel("y")
        ax.set_facecolor('#DADADA')
        
        if circle:
            circle = plt.Circle((mid_pixel, mid_pixel), Rpixel, color="red",fill=False,zorder=2000)
            ax.add_patch(circle)
        plt.tight_layout()
        if savefig:
            plt.savefig('fpath' + file_name + '.png',dpi=250,bbox_inches='tight')

        ax.invert_yaxis()
        plt.show()

        plt.close()
        return None