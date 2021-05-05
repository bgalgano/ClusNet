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
import random, string

# tensorflow
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.callbacks import CSVLogger

from datetime import timedelta

from time import gmtime, strftime
import time 

from os.path import expanduser
home = expanduser("~")

sys.path.append(home+'/repos/ClusNet/code/modules/')

from ClusNet import Cluster

def generate_model(kernel_size, pool_size, activation, strides, input_shape,im_size=384):
    model = tf.keras.Sequential()

    model.add(tf.keras.Input(shape=(im_size,im_size,1)))

    padding = 'same'
    
    # 1. 3×3 convolution with 16 filters
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=kernel_size,
activation=activation,padding=padding))

    # 2. 2×2, stride-2 max pooling
    model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides))

    # 3. 3×3 convolution with 32 filters
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size,
activation=activation,padding=padding))

    # 4. 2×2, stride-2 max pooling
    model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides))

    # 5. 3×3 convolution with 64 filters
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size,
activation=activation,padding=padding))

    # 6. 2×2, stride-2 max pooling
    model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides))

    # 7. global average pooling
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # 8. 10% dropout
    model.add(tf.keras.layers.Dropout(0.1))

    # 9. 200 neurons, fully connected
    model.add(tf.keras.layers.Dense(units=200,activation=activation))

    # 10. 10% dropout
    model.add(tf.keras.layers.Dropout(0.1))

    # 11. 100 neurons, fully connected
    model.add(tf.keras.layers.Dense(units=100,activation=activation))

    # 12. 20 neurons, fully connected
    model.add(tf.keras.layers.Dense(units=20,activation=activation))

    # 13. output neuron
    model.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

    model.summary()
    return model

def plot_resid(y_train,y_train_model,validation_y,validation_y_model,spath,model_id):
    fig, ax = plt.subplots(ncols=1,nrows=2,figsize=(5,5),sharex=False)
    
    y_model_ones = y_train_model[y_train==1]
    y_model_zeros = y_train_model[y_train==0]
               
    y_valid_ones = validation_y_model[validation_y==1]    
    y_valid_zeros = validation_y_model[validation_y==0]    
               
    ax[0].hist(y_valid_ones,label='valid')
    ax[0].hist(y_model_ones,label='train')
    ax[0].axvline(1,color='black',lw=1)
    
    ax[1].hist(y_valid_zeros,label='valid')
    ax[1].hist(y_model_zeros,label='train')
    ax[1].axvline(0,color='black',lw=1)
           
    plt.xlabel('Label Residual')
    plt.subplots_adjust(hspace=0)
    figpath = spath + '/resid_{}.png'.format(model_id)
    plt.savefig(figpath,dpi=200,bbox_inches='tight')
    #plt.show()
    print('\n---> residual plot saved to:',figpath)
               

def plot_1to1(y_train,y_train_model,validation_y,validation_y_model,spath,model_id):
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(5,5),sharey=False,sharex=False)

    for y, y_model, label in zip([y_train,validation_y],[y_train_model,validation_y_model],['Training data','Validation data']):
 
        ax.scatter(y,y_model,s=5,marker=".",label=label)
        
    ax.set_xlabel('Truth')
    ax.set_ylabel('Predicted')

    plt.legend(frameon=False)
    plt.xticks([0,1])
    plt.tight_layout()
    figpath = spath + '/1to1_center_xy_{}.png'.format(model_id)
    plt.savefig(figpath,
                dpi=200,
                bbox_inches='tight')

    print("\n---> 1to1 plot saved to:", figpath)
    print()

    #plt.show()

    plt.close()

def plot_metrics(history,spath,model_id):

    fig, ax = plt.subplots(ncols=1,nrows=2,figsize=(6,5),sharex=True)
    for idx, stat in zip([0,1],['accuracy','loss']):
        ax[idx].plot(history.history[stat],color='#FF5733')
        ax[idx].plot(history.history['val_' + stat],color='#21E332')
        ax[idx].set_ylabel(stat)
    plt.xlabel('Epoch')
    plt.subplots_adjust(hspace=0)
    plt.legend(['train set', 'valid set'],ncol=2,frameon=False)
    figpath = spath + '/accuracy_loss_{}.png'.format(model_id)
    plt.savefig(figpath,dpi=200,bbox_inches='tight')
    plt.show()
    print('\n---> metrics plot saved to:',figpath)
