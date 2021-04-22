#!/home-net/home-1/bgalgan1@jhu.edu/code/tf-new

# -*- coding: utf-8 -*-
# Brianna Galgano
# code to apply category CNN to eROSITA clusters

# Imports
#/anaconda3/envs/tf-cpu/bin/python

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
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import model_from_yaml
from keras.callbacks import CSVLogger

from datetime import timedelta

from time import gmtime, strftime
import time 

from os.path import expanduser
home = expanduser("~")

sys.path.append(home+'/repos/ClusNet/code/modules/')

from ClustNet import Cluster

def generate_model(kernel_size, pool_size, activation, strides, input_shape,im_size=384):
    model = keras.Sequential()

    model.add(keras.Input(shape=(im_size,im_size,1)))

    padding = 'same'
    
    # 1. 3×3 convolution with 16 filters
    model.add(layers.Conv2D(filters=16, kernel_size=kernel_size,
activation=activation,padding=padding))

    # 2. 2×2, stride-2 max pooling
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=strides))

    # 3. 3×3 convolution with 32 filters
    model.add(layers.Conv2D(filters=32, kernel_size=kernel_size,
activation=activation,padding=padding))

    # 4. 2×2, stride-2 max pooling
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=strides))

    # 5. 3×3 convolution with 64 filters
    model.add(layers.Conv2D(filters=64, kernel_size=kernel_size,
activation=activation,padding=padding))

    # 6. 2×2, stride-2 max pooling
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=strides))

    # 7. global average pooling
    model.add(layers.GlobalAveragePooling2D())

    # 8. 10% dropout
    model.add(layers.Dropout(0.1))

    # 9. 200 neurons, fully connected
    model.add(layers.Dense(units=200,activation=activation))

    # 10. 10% dropout
    model.add(layers.Dropout(0.1))

    # 11. 100 neurons, fully connected
    model.add(layers.Dense(units=100,activation=activation))

    # 12. 20 neurons, fully connected
    model.add(layers.Dense(units=20,activation=activation))

    # 13. output neuron
    model.add(layers.Dense(units=1,activation='sigmoid'))

    model.summary()
    return model

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
    #plt.show()
    print('\n---> metrics plot saved to:',figpath)


def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    k = 0.01
    epochs = 10

    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # plot settings
    label_size = 14
    mpl.rcParams['legend.fontsize'] = label_size
    mpl.rcParams['axes.labelsize'] = label_size
    mpl.rcParams['axes.labelpad'] = 10
    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size

    # print settings
    np.set_printoptions(precision=3, suppress=True)
        
    im_size = 384
    input_shape = (im_size,im_size,1) # width, height, channel number
    pool_size = (2,2)
    kernel_size = (3,3)
    activation = 'relu'
    strides = 2

    print("\nMODEL:")
    model = generate_model(kernel_size=kernel_size,
                             pool_size=pool_size,
                             activation=activation,
                             strides=2,
                             input_shape=input_shape)

    # compiler
    metrics = ["accuracy"]
    opt='Adam'
    
    loss=tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=metrics)

    # model fitting
    training_data, validation_data, modeldir = Cluster.load_dataset(k=k,validation_split=0.20)
    x_train, y_train = training_data
    validation_x, validation_y = validation_data
    batch_size = 2
    print("\n********LEARNING START********")
    start = time.time()
    
    model_id = os.path.basename(os.path.normpath(modeldir))
    csv_logger = CSVLogger(modeldir+'/history_{}.log'.format(model_id), separator=',', append=False)

    history = model.fit(x=x_train,
                        y=y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_x, validation_y),
                        verbose=2,
                       callbacks=[csv_logger])
    


    print("********LEARNING END********")
    elapsed = time.time() - start
    print("\nTIME:",str(timedelta(seconds=elapsed)))
    
    # save model assets
    print("\nModel assets saved to:", modeldir)
    model.save(modeldir)
    
    # plot loss and accuracy
    plot_metrics(history=history,
                 spath=modeldir,
                 model_id=model_id)
    
    print("\nPredicting labels:")
    y_train_model = model.predict(x_train,verbose=1,batch_size=batch_size)
    validation_y_model = model.predict(validation_x,verbose=1,batch_size=batch_size)
    
    # plot 1-to-1
    plot_1to1(y_train=y_train,
              y_train_model=y_train_model,
              validation_y=validation_y,
              validation_y_model=validation_y_model,
              spath=modeldir,
              model_id=model_id)
    
    
if __name__ == "__main__":
    main()
    

