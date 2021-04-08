#!/code/envs/tf/bin/python
# -*- coding: utf-8 -*-
# Brianna Galgano
# code to create CNN of gaussian profile

# Imports
#/anaconda3/envs/tf-cpu/bin/python
import dataset
import Gauss

import matplotlib.pylab as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# statistics
import numpy as np
import random
import copy
from scipy import signal
import time
import random, string
import os

# tensorflow
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import model_from_yaml

from datetime import timedelta

from time import gmtime, strftime

def main():

    # create dataset

    # create model

    # load dataset

    # load model

    # fit model

    # evaluate model

    # save model

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

    cwd = os.getcwd()
    spath = cwd + '/models'

    im_size = 128
    set_size = 100

    # create gaussian dataset
    dataset = create_set(im_size=im_size,set_size=set_size)

    # create training set preview figure
    # plot(dataset=dataset, spath=spath)

    x_train, y_train = load_dataset(dataset)

    input_shape = (im_size,im_size,1) # width, height, channel number
    pool_size = (2,2)
    kernel_size = (3,3)
    activation = 'relu'
    strides = 2

    print("\nMODEL:")
    model = generate_model(kernel_size=kernel_size,
                             pool_size=pool_size,
                             activation='relu',
                             strides=2,
                             input_shape=input_shape)

    # compiler
    opt = Adam()
    loss = tf.keras.losses.MeanAbsoluteError()
    metrics = ["accuracy"]
    epochs = 2

    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=metrics)

    # model fitting
    validation_split = 0.2
    x_train, y_train = load_dataset(dataset)
    batch_size = len(dataset)

    split_at = int(x_train.shape[0] * (1-validation_split))
    validation_x = x_train[split_at:]
    validation_y = y_train[split_at:]
    validation_data=(validation_x, validation_y)
    print("\n***LEARNING START***")
    start = time.time()
    history = model.fit(x=x_train[:split_at],
                        y=y_train[:split_at],
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_x, validation_y),
                        verbose=2)
    print("***LEARNING END***")
    elapsed = time.time() - start
    print("\nTIME:",str(timedelta(seconds=elapsed)))

    # create directory to save model information
    model_id = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    model_dir = spath + '/' + model_id
    os.mkdir(model_dir)

    print("--->Model saved to:", model_dir)
    model.save(model_dir)

    """
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(model_dir+"/model_{}.yaml".format(model_id), "w") as yaml_file:
        yaml_file.write(model_yaml)

    # serialize weights to HDF5
    model.save_weights(model_dir + "/wgts_{}.h5".format(model_id))
    print("Model assets saved to:", model_dir)
    """

    # plot loss and accuracy
    plot_metrics(history=history,spath=model_dir,model_id=model_id)

    # predict labels from model
    print("\nPredicting training labels:")
    y_train_model = model.predict(x_train,verbose=1)




    # plot 1-to-1

    validation_y_model = model.predict(validation_x,verbose=1)

    plot_1tot1(y_train=y_train,
               y_train_model=y_train_model,
               validation_y=validation_y,
               validation_y_model=validation_y_model,
               spath=model_dir,
               model_id=model_id)
    print("")

if __name__ == "__main__":
    main()
