#!/usr/bin/env python

# -*- coding: utf-8 -*-
# Brianna Galgano
# code to apply segmentation mapping CNN to eROSITA clusters

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

import gc
from keras import backend as K 

# plot

label_size = 14
mpl.rcParams['legend.fontsize'] = label_size
mpl.rcParams['axes.labelsize'] = label_size
mpl.rcParams['axes.labelpad'] = 10
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size

# print
np.set_printoptions(precision=3, suppress=True)


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 384.0
    input_mask = input_mask
    return input_image, input_mask

@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

@tf.function
def load_image_test(datapoint):
    
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[128, 128, OUTPUT_CHANNELS])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(output_channels, 
                                           OUTPUT_CHANNELS, 
                                           strides=2,
                                           padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def print_device():
    print()
    print("GPU(s) available:")

    for GPU in tf.config.list_physical_devices('GPU'):
        print(GPU)

    print("CPU(s) available:")  
    for CPU in tf.config.list_physical_devices('CPU'):
        print(CPU)

def main():
    print_device()
    
    k = 2000 # number of clusters

    OUTPUT_CHANNELS = 1
    
    TRAIN_LENGTH = k
    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
    
    VAL_SUBSPLITS = 5
    VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

    EPOCHS = 20

    #LOSS = tf.keras.losses.BinaryCrossentropy()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(lr=1e-4)
    metrics = ["accuracy"]

    # training 
    input_shape = (128,128,OUTPUT_CHANNELS) # width, height, channel number

   
    # compile model
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=metrics)
    
    # load data
    dataset = Cluster.load_keras.dataset(mode='seg', k=k, validation_split=0.20, noise=noise)
    
    train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    test = dataset['test'].map(load_image_test)
    
    train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test.batch(BATCH_SIZE)

    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False

    up_stack = [
    pix2pix.upsample(512, OUTPUT_CHANNELS),  # 4x4 -> 8x8
    pix2pix.upsample(256, OUTPUT_CHANNELS),  # 8x8 -> 16x16
    pix2pix.upsample(128, OUTPUT_CHANNELS),  # 16x16 -> 32x32
    pix2pix.upsample(64, OUTPUT_CHANNELS),   # 32x32 -> 64x64
    ]
    
    model = unet_model(output_channels=OUTPUT_CHANNELS)
    
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    print("\n********LEARNING START********")
    start = time.time()

    model_id = os.path.basename(os.path.normpath(modeldir))

    csv_logger = CSVLogger(modeldir + '/history_{}.log'.format(model_id),
                           separator=',',
                           append=False)
    
    model_history = model.fit(train_dataset,
                              epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=test_dataset,
                              callbacks=[csv_logger, DisplayCallback()])
    
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

    del model
    gc.collect()
    K.clear_session()
    tf.compat.v1.reset_default_graph() 

if __name__ == "__main__":
    main()
    

