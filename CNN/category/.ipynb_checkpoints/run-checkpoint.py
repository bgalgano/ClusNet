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
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.callbacks import CSVLogger

from datetime import timedelta

from time import gmtime, strftime
import time 

from os.path import expanduser
home = expanduser("~")

sys.path.append(home+'/repos/ClusNet/code/modules/')

from ClusNet import Cluster
from ClusNet import model as m
# plot

label_size = 14
mpl.rcParams['legend.fontsize'] = label_size
mpl.rcParams['axes.labelsize'] = label_size
mpl.rcParams['axes.labelpad'] = 10
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size

# print
np.set_printoptions(precision=3, suppress=True)

def main():
    print("\n Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
    print()
    #os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # data
    im_size = 384
    k = 0.80 # percentage of 

    # compiler
    metrics = ["accuracy"]
    opt=tf.keras.optimizers.Adam()

    # training 
    epochs = 100
    input_shape = (im_size,im_size,1) # width, height, channel number
    pool_size = (2,2)
    kernel_size = (3,3)
    activation = 'relu'
    strides = 2

    # GPU
    batch_size = 2
    
    # generate model
    model = m.generate_model(kernel_size=kernel_size,
                             pool_size=pool_size,
                             activation=activation,
                             strides=2,
                             input_shape=input_shape)
    # compile model
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=metrics)
    
    # load data
    training_data, validation_data, modeldir = Cluster.load_dataset(k=k,validation_split=0.20,noise=False)
    x_train, y_train = training_data
    validation_x, validation_y = validation_data
    
    print("\n********LEARNING START********")
    start = time.time()

    model_id = os.path.basename(os.path.normpath(modeldir))

    csv_logger = CSVLogger(modeldir + '/history_{}.log'.format(model_id),
                           separator=',',
                           append=False)

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
    m.plot_metrics(history=history,
                 spath=modeldir,
                 model_id=model_id)

    y_train_model = model.predict(x_train,verbose=1,batch_size=batch_size)
    validation_y_model = model.predict(validation_x,verbose=1,batch_size=batch_size)

    # plot 1-to-1
    m.plot_1to1(y_train=y_train,
              y_train_model=y_train_model,
              validation_y=validation_y,
              validation_y_model=validation_y_model,
              spath=modeldir,
              model_id=model_id)

    # plot residuals
    m.plot_resid(y_train=y_train,
              y_train_model=y_train_model,
              validation_y=validation_y,
              validation_y_model=validation_y_model,
              spath=modeldir,
              model_id=model_id)


if __name__ == "__main__":
    main()
    

