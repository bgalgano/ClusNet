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
class Profile:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.image = np.random.normal(loc=loc,scale=scale,size=(128,128)) 
        self.noise = False
        self.lam = 0.1133929878
        self.mid_pixel = 64 # 128/2
        self.size = 128

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
    
    def add_noise(self,lam=self.lam):
        """
        add Poisson noise to cluster image matrix
        """
        self.noise = np.random.poisson(lam=lam, size=self.cluster.shape)
        self.image += self.noise
        return
        
    def shift(self):
        """
        shift cluster randomly within bounds of image
        """
        """
        shift cluster randomly within bounds of image
        """
        r = self.scale
        mid = self.mid_pixel #center pixel index of 384x384 image
        delta = self.size - self.mid_pixel - r
        
        x = np.random.randint(low=-1*delta,high=delta,size=1)[0]
        y = np.random.randint(low=-1*delta,high=delta,size=1)[0]

        self.x += mid
        self.y += mid
        image_shift = np.roll(self.image,shift=x,axis=0)
        self.image = np.roll(image_shift,shift=y,axis=1)
        
        return 
    
    def plot(self,spath='../figs/profile/'):
        """
        plot image
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(self.image,interpolation='none',cmap='binary')
        
        ticks = np.arange(0,129,32)
        plt.xticks(ticks),plt.yticks(ticks)
        plt.title(r'$loc={}, scale={}$'.format(self.loc,self.scale))
        plt.show()
        plt.close()
        return None