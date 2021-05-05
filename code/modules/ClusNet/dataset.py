#!/anaconda3/bin/python
#/opt/anaconda3/bin/python

# matplotlib
import matplotlib.pylab as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# statistics
import numpy as np
from scipy import signal

# object support
import copy

# time keeping
import time
from datetime import timedelta

# data munging
import random, string
import pandas as pd

# operating system
import os
import glob
# tensorflow
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# keras support
from keras.models import model_from_yaml

import glob as glob

class Profile:
    def __init__(self,std,im_size):
        self.mid_pixel = int(im_size/2) # 128/2
        self.x, self.y = self.mid_pixel, self.mid_pixel
        self.im_size = im_size
        self.std = std
        self.noise = False
        self.lam = 0.1133929878
        
        gkern1d = signal.gaussian(self.im_size, std=std).reshape(self.im_size, 1)
        self.im = np.outer(gkern1d, gkern1d)
        
        self.im_lrud  = None
        self.im_lr = None
        self.im_ud = None

    def __repr__(self):
        """
        print cluster metadata
        """
        return str(self.im)
    
    def to_pandas(self):
        """
        convert metadata (as recarray) to pandas DataFrame
        """
        self.meta = pd.DataFrame(self.meta)
        return
    
    def add_noise(self):
        """
        add Poisson noise to cluster im matrix
        """
        self.noise = np.random.poisson(lam=self.lam, size=self.im.shape)
        self.im += self.noise
        return
        
    def shift(self):
        """
        shift cluster randomly within bounds of im
        """
        """
        shift cluster randomly within bounds of im
        """
        r = self.std
        mid = self.mid_pixel #center pixel index of 384x384 image
        delta = self.im_size - self.mid_pixel - r - 10
        
        x = np.random.randint(low=-1*delta,high=delta,size=1)[0]
        y = np.random.randint(low=-1*delta,high=delta,size=1)[0]

        self.x += x
        self.y += y
        im_shift = np.roll(self.im,shift=y,axis=0)
        self.im = np.roll(im_shift,shift=x,axis=1)
        
        return 
    
    def plot(self,spath='../figs/profile/'):
        """
        plot image
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(self.im,interpolation='none',cmap='viridis')
        
        ticks = np.arange(0,self.size,50)
        plt.xticks(ticks),plt.yticks(ticks)

        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.12)

        plt.colorbar(im, cax=cax)
        # plt.show()
        plt.close()
        
        return None

    def flip_lr(self):
        im_c = np.zeros((self.im_size,self.im_size))
        im_c[self.x,self.y] = 1
        
        im_lr = np.fliplr(self.im)
        im_c_lr = np.flipud(im_c)
        
        self.im_lr = im_lr
        self.x_lr, self.y_lr = [val[0] for val in np.nonzero(im_c_lr)]
        
        self.im = im_lr
        self.x, self.y = self.x_lr, self.y_lr
        return None

    def flip_ud(self):
        im_c = np.zeros((self.im_size,self.im_size))
        im_c[self.x,self.y] = 1

        im_ud = np.flipud(self.im)
        im_c_ud = np.fliplr(im_c)
        
        self.im_ud = im_ud
        self.x_ud, self.y_ud = [val[0] for val in np.nonzero(im_c_ud)]
        
        self.im = im_ud
        self.x, self.y = self.x_ud, self.y_ud
        return None
    
    def flip_lrud(self):
        im_c = np.zeros((self.im_size,self.im_size))
        im_c[self.x,self.y] = 1
        
        im_lrud = np.fliplr(np.flipud(self.im))
        im_c_lrud = np.flipud(np.fliplr(im_c))
        
        self.im_lrud = im_lrud
        self.x_lrud, self.y_lrud = [val[0] for val in np.nonzero(im_c_lrud)]
        
        self.im = im_lrud
        self.x, self.y = self.x_lrud, self.y_lrud
        return None
    
def create_set(set_size=1000,im_size=128,noise=True,shift=True,flip=True):
    dataset = []
    for i in range(set_size):
        std = np.random.randint(low=2,high=15,size=1)
        x = Profile(std=std,im_size=im_size)
        if noise:
            x.add_noise()
        if shift:
            x.shift()

        if flip:
            # make copy of original profile
            x_copy = copy.copy(x)

            # flip all left/right
            x.flip_lr()
            x_lr = copy.copy(x)

            # flip all up/down
            x.flip_ud()
            x_ud = copy.copy(x)

            # flip all left/right and up/down
            #x.flip_lrud()


            dataset.extend([x_copy,x_lr,x_ud])

        else:
            dataset.append(x)
    return np.array(dataset)

def load_dataset(dataset,norm=True):
    # fit the keras model on the dataset
    size = len(dataset)
    data = np.array([prof.im for prof in dataset])
    labels = np.array([(prof.x,prof.y,prof.std[0]) for prof in dataset])
    
    im_size = dataset[0].im.shape[0]
    if norm:
        norm_factor = im_size
    else:
        norm_factor = 1
    
    data = data/norm_factor
    
    idx = np.arange(0,size,1)
    idx = np.random.shuffle(idx) 
    
    x_train, y_train = data[idx], labels[idx]
    x_train = x_train.reshape(-1, im_size, im_size, 1)
    print("\nDataset loaded.")
    print("Image input shape:", x_train.shape)
    print("Label input shape:", y_train.shape)

    return x_train, y_train
    
def plot_metrics(history,spath,model_id):
    
    fig, ax = plt.subplots(ncols=1,nrows=2,figsize=(7,5),sharex=True)
    for idx, stat in zip([0,1],['accuracy','loss']):
        ax[idx].plot(history.history[stat])
        ax[idx].plot(history.history['val_' + stat])
        ax[idx].set_ylabel(stat)
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'],ncol=2,frameon=False)
    ax[0].set_ylim(0.40,1)
    ax[1].set_xlim(0,0.1)
    plt.tight_layout()

    plt.savefig(spath + '/accuracy_loss_{}.png'.format(model_id),dpi=200, 
bbox_inches='tight')
    #plt.show()
    print('\n---> metrics plot saved to',spath)
