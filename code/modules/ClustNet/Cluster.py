from astropy.table import Table
from glob import glob
import os
from pathlib import Path
import matplotlib.pylab as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

import numpy as np
import pandas as pd
import sys
from astropy.io import fits
import random
import time
from os.path import expanduser
home = expanduser("~")
cluster_path = '/repos/ClusNet/data/eROSITA_no_background/clusterList.npy'
clusterList=np.load(home+cluster_path)
#clusterDir = np.load('../../data/eROSITA_no_background/')

class Cluster:
    def __init__(self, fpath=None):
        
        self.fpath = fpath
        self.lam = 0.1133929878

        if fpath is None:
            self.id = 'noise'
            self.image = np.zeros(shape=(384,384))
            self.meta = None
            self.meta_col = None
            
            self.M500 = None
            self.R500 = None
            self.Rpixel = None
            self.Tkev = None
            
            self.noise = np.random.poisson(lam=self.lam, size=self.image.shape)
            self.image += self.noise
            
        else:
            self.id = Path(fpath).stem
        
            with fits.open(fpath) as data:
                cluster_df = pd.DataFrame(data[0].data)
            self.image = cluster_df.to_numpy()
        
            meta = clusterList[clusterList['id']==int(self.id)]
            self.meta = meta

            self.M500 = np.log10(meta['M500_msolh'])[0]
            self.R500 = np.log10(meta['r500_kpch'])[0]
            self.Rpixel = meta['R500_pixel'][0]
            self.Tkev = np.log10(meta['T_kev'])[0]
            
            if type(self.meta) == np.ndarray:
                self.meta_col = meta.dtype.names
            else:
                self.meta_col = meta.columns()
            
        self.w_pix  = len(self.image[:,0])
        self.xmid = int(self.w_pix/2)
        self.ymid = int(self.w_pix/2)
        self.mid_pix = (self.xmid,self.ymid)
        
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
        self.noise = np.random.poisson(lam=self.lam, size=self.image.shape)
        self.image += self.noise

        return
        
    def shift(self,delta=64):
        """
        shift cluster randomly within bounds of image
        """
        
        low = -1*delta
        high = delta
        
        r = self.Rpixel
        xmid = self.xmid #center pixel index of 384x384 image
        ymid = self.ymid
        
        x_shift = np.random.randint(low=low,high=high,size=1)[0]
        y_shift = np.random.randint(low=low,high=high,size=1)[0]

        xi = xmid + x_shift
        yi = ymid + y_shift

        cluster_y_shift = np.roll(self.image,shift=y_shift,axis=0)
        self.image = np.roll(cluster_y_shift,shift=x_shift,axis=1)
        
        self.xmid = self.xmid + x_shift
        self.ymid = self.ymid + y_shift
        self.mid_pix = (self.xmid,self.ymid)
        return
    
    def add_border(self,pad_width=64):
        """
        add border or pad to image with constant values
        default is pad width of x pixels, constant is 0
        """
        image_border = np.pad(array=self.image,
                              constant_values=0,
                              mode='constant',
                              pad_width=pad_width)
        self.image = image_border
        self.w_pix  = len(image_border[:,0])
        self.xmid += int(pad_width)
        self.ymid += int(pad_width)
        self.mix_pix = (self.xmid,self.ymid)
        return
    
    def plot(self,circle=True,square=True,savefig=False,spath='../figs/eROSITA/'):
        """
        plot cluster image with radius and mass information
        """
        w_pix = self.w_pix
        mid_pix = self.mid_pix
        
        Rpixel = self.Rpixel
        cluster = self.image
        cluster_id = self.id
        
        plt.figure(figsize=(4,4))        
        cmap = mpl.cm.rainbow
        im = plt.imshow(np.log10(cluster),cmap=cmap,interpolation='none')

        ax = plt.gca()
        
        if self.meta is None:
            plt.title('Noise')
        else:
            
            cluster_row = self.meta
            log_m = np.log10(cluster_row['M500_msolh'])[0]
            plt.title(r'$\log\left(\frac{M_\mathrm{500c}}{h^{-1}\,M_\odot}\right) = $'+'{:.2f}'.format(log_m))
            
            if circle:
                circle = plt.Circle((self.xmid, self.ymid), self.Rpixel, color="red",fill=False,zorder=2000)
                ax.add_patch(circle)
        
            if square:
                circle = plt.Rectangle((self.xmid-384/2, self.ymid-384/2), 384, 384, color="black",fill=False,zorder=2000)
                ax.add_patch(circle)


        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.12)

        plt.colorbar(im, cax=cax)
        ax.set_xticks(np.arange(0,w_pix,100))
        ax.set_yticks(np.arange(0,w_pix,100))

        ax.set_xlabel("x"), ax.set_ylabel("y")
        ax.set_facecolor('#DADADA')
        print(self.mid_pix)

        
        plt.tight_layout()

        if savefig:
            plt.savefig('fpath' + file_name + '.png',dpi=250,bbox_inches='tight')

        ax.invert_yaxis()
        plt.show()

        plt.close()
        return None
    
def load_dataset(k='all', globdir='../../data/eROSITA_no_background/*.fits',norm=True,addneg=True):
    clusglob = glob(globdir)
    if type(k) == float and k < 1:
        k = int(k*len(clusglob))    
    if k == 'all':
        k = len(clusglob)
    clusfpaths = random.choices(clusglob,k=k)

    x_train = []
    y_train = []
    print("\nLoading {:} clusters...".format(k))
    for i, clusfpath in enumerate(clusfpaths):   
        
        x = Cluster(fpath=clusfpath)
        x.add_noise()
        if norm:
            image = x.image / x.w_pix
        else:
            image = x.image
        x_train.append(image)
        y_train.append(1)
        
    if addneg:
        print('Adding {:} negatives...'.format(k))
        for i in range(0,k):
            x_noise = Cluster()
            x_noise.add_noise()
            if norm:
                image = x_noise.image / x_noise.w_pix
            else:
                image = x_noise.image
            x_train.append(image)
            y_train.append(0)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print("Done.")
    print('\nX labels shape:', x_train.shape)
    print('Y labels shape:', y_train.shape)
    x_train = x_train.reshape(-1, 384, 384, 1)
    return x_train, y_train


def plot(spath="./",globdir='../../data/eROSITA_no_background/*.fits'):
    
    fig, axes = plt.subplots(nrows=5,ncols=5,figsize=(6,6))
    clusglob = glob(globdir)
    clusfpaths = random.choices(clusglob,k=25)
    dataset = []
    for clusfpath in clusfpaths:
        dataset.append(Cluster(fpath=clusfpath))
        
    for i, ax in enumerate(axes.flat):
        prof = dataset[i]
        prof.add_noise()
        image = prof.image/384
        cmap = plt.cm.viridis
        ax.imshow(np.log10(image),interpolation='none',cmap=cmap,norm=mpl.colors.LogNorm())
        ax.set_yticks([])
        ax.set_xticks([])
        
    space = 0.05
    plt.tight_layout()
    plt.subplots_adjust(wspace=space,hspace=space)
    fpath = spath+'dataset_10x10_view.png'
    plt.savefig(fpath,dpi=300)
    plt.show()
    plt.close()
    print("\nDataset preview saved to:", fpath)
