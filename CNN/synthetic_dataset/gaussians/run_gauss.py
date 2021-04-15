#!/usr/bin/python
# brianna galgano

# plotting
import matplotlib.pylab as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

# statistics
import numpy as np
import random
from scipy import signal
import time

# tensorflow
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# plot settings

label_size = 14

mpl.rcParams['legend.fontsize'] = label_size
mpl.rcParams['axes.labelsize'] = label_size 

mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size

mpl.rcParams['axes.labelpad'] = 10

# print settings
np.set_printoptions(precision=3, suppress=True)

class Profile:
    def __init__(self,std,size):
        self.mid_pixel = size/2 # 128/2
        self.x, self.y = self.mid_pixel, self.mid_pixel
        self.size = size
        self.std = std
        self.noise = False
        self.lam = 0.1133929878
        gkern1d = signal.gaussian(self.size, std=std).reshape(self.size, 1)
        self.image = np.outer(gkern1d, gkern1d)


    def __repr__(self):
        """
        print cluster metadata
        """
        return str(self.image)
    
    
    def add_noise(self):
        """
        add Poisson noise to cluster image matrix
        """
        self.noise = np.random.poisson(lam=self.lam, size=self.image.shape)
        self.image += self.noise
        return
        
    def shift(self):
        """
        shift cluster randomly within bounds of image
        """
        r = self.std
        mid = self.mid_pixel #center pixel index of 384x384 image
        delta = self.size - self.mid_pixel - r
        
        x = np.random.randint(low=-1*delta,high=delta,size=1)[0]
        y = np.random.randint(low=-1*delta,high=delta,size=1)[0]

        self.x += x
        self.y += y
        image_shift = np.roll(self.image,shift=x,axis=0)
        self.image = np.roll(image_shift,shift=y,axis=1)
        
        return 
    
    def plot(self,spath='../figs/profile/'):
        """
        plot image
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(self.image,interpolation='none',cmap='viridis')
        
        ticks = np.arange(0,self.size,50)
        plt.xticks(ticks),plt.yticks(ticks)
        #plt.title(r'$loc={}, scale={}$'.format(self.loc,self.scale))

        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.12)

        plt.colorbar(im, cax=cax)
        plt.show()
        plt.close()
        return None
    
