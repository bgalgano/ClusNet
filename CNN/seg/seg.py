#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import os
import pandas as pd
import time as time
from datetime import timedelta

import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
import matplotlib as mpl
import glob
import random
import sys
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from skimage.transform import resize

from os.path import expanduser
home = expanduser("~")
sys.path.append(home+'/repos/ClusNet/code/modules/')

from scipy.ndimage import gaussian_filter
from ClusNet import Cluster
from ClusNet import model as m
from ClusNet import dataset as ds

import gc
from keras import backend as K

# tensorflow
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.callbacks import CSVLogger
clusfpath = home + '/repos/ClusNet/data/eROSITA_no_background/*.fits'
clusglob = glob.glob(clusfpath)
np.seterr(divide='ignore')
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})



def r(array):
    return (np.nanmin(array),np.nanmax(array))

N_CLASSES = 3
MASK_SIZE = 2



def make_dataset(DATASET_SIZE=100,
                 IMG_SIZE=128,
                 resize=128,
                 validation_split=0.80,
                 shift=False,
                 agn=True,
                 p=True,
                 sigma=0.25,
                 intersect=False):
    home = expanduser("~")
    repodir = home + '/repos/ClusNet/'
    clusterList = np.load(repodir + 'data/eROSITA_no_background/clusterList.npy')
    clusterDir = repodir + 'data/eROSITA_no_background/'
    GLOBDIR = clusterDir
    clusglob = glob.glob(GLOBDIR + '/*.fits')
    paths = random.choices(clusglob, k=DATASET_SIZE)

    print("\nLoading {} cluster(s)...".format(len(paths)))
    image_array = []
    mask_array = []
    for i, path in enumerate(paths):
        Cluster.printProgressBar(total=len(paths),iteration=i)
        x = Cluster.Cluster(fpath=path)

        cluster = tf.image.resize(x.cluster[...,tf.newaxis],
                              preserve_aspect_ratio=True,
                              size=(IMG_SIZE,IMG_SIZE),
                              antialias=True)
        cluster = cluster.numpy()
        cluster = np.squeeze(cluster, axis=(2,))

        if p:
            # add Poisson noise
            p_noise = np.random.poisson(lam=0.1133929878, size=(IMG_SIZE,IMG_SIZE))
        else:
            p_noise = np.zeros(size=(IMG_SIZE,IMG_SIZE))

        if agn:
            # add AGNs
            num = np.random.randint(low=1,high=4,size=None)
            agn_noise = []
            for i in range(num):
                std = np.random.uniform(0.5,1.,size=None)
                agn = ds.Profile(std=std,im_size=(IMG_SIZE,IMG_SIZE))
                agn.shift()
                agn_noise.append(agn.image)
            agn_noise = np.array(agn_noise).T
            agn_all = np.sum(agn_noise,axis=-1)

            agns_filter = gaussian_filter(agn_all, sigma=sigma, mode='constant', cval=0.0)
            agns_mask = (agns_filter > 0.) * 1


        else:
            agn_all = np.zeros(size=(IMG_SIZE,IMG_SIZE))
            agns_mask = np.zeros(size=(IMG_SIZE,IMG_SIZE))

        cluster_filter = gaussian_filter(cluster, sigma=sigma, mode='constant',cval=0.0)
        cluster_mask = (cluster_filter > 0.) * 1

        catnum_map = np.sum([cluster_mask,agns_mask],axis=0)
        intersect_mask = (catnum_map==2) * 1

        image = np.array([cluster+agn_all+p_noise]).T
        image = tf.math.sigmoid(np.log10(image))

        if intersect:
            mask = np.array([cluster_mask,agns_mask,intersect_mask]).T
        else:
            mask = np.array([cluster_mask,agns_mask]).T

        image_tensor = tf.constant(image)
        mask_tensor = tf.constant(mask)

        image_array.append(image_tensor)
        mask_array.append(mask_tensor)

    image_array = np.array(image_array)
    mask_array = np.array(mask_array)

    # trainset
    split_idx = int(validation_split*DATASET_SIZE)
    img_train = image_array[:split_idx]
    msk_train = mask_array[:split_idx]

    # test set
    img_val = image_array[split_idx:]
    msk_val = mask_array[split_idx:]

    paths_train = paths[:split_idx]
    paths_val = paths[split_idx:]
    """
    x_train_save = x_train.reshape(x_train.shape[0], -1)
    x_test_save = x_test.reshape(x_test.shape[0], -1)

    train_file = h5py.File(modeldir+"/train_data.h5", 'w')
    train_file.create_dataset('x_train', data=x_train_save)
    train_file.create_dataset('y_train', data=y_train)
    train_file.close()

    test_file = h5py.File(modeldir+"/val_data.h5", 'w')
    test_file.create_dataset('x_test', data=x_test_save)
    test_file.create_dataset('y_test', data=y_test)
    test_file.close()

    np.savetxt(fname=modeldir+"/train_paths.txt",X=paths_train,delimiter="\n",fmt="%s")
    np.savetxt(fname=modeldir+"/test_paths.txt",X=paths_test,delimiter="\n",fmt="%s")
    print("\nSaved dataset paths to -->",modeldir)
    """
    print("\n\nimg_train shape:",img_train.shape)
    print("msk_train shape:",msk_train.shape)

    print("\nimg_val shape:  ",img_val.shape)
    print("mask_val shape: ",msk_val.shape)

    return img_train, msk_train, paths_train, img_val, msk_val, paths_val

def seg_img(image,mask):

    fig, ax = plt.subplots(figsize=(5,5),ncols=2,nrows=2,sharex=True,sharey=True)

    cmap = mpl.cm.binary_r
    vmin, vmax = 0, 1
    norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)

    ax[0][0].imshow(image,
                 cmap=cmap,
                 interpolation='none',
                 norm=norm, aspect = 1)

    ax[0][1].imshow(mask[:,:,0],
                 cmap=mpl.cm.Blues,
                 interpolation='none', aspect = 1)
    ax[1][0].imshow(mask[:,:,1],
                 cmap=mpl.cm.Reds,
                 interpolation='none', aspect = 1)
    ax[1][1].imshow(mask[:,:,2],
                 cmap=mpl.cm.Greens,
                 interpolation='none', aspect = 1,label='intersect')


    labels = ['image','cluster','AGN','intersect']
    colors = ['white','black','black','black']
    for k, ax_ in enumerate(ax.reshape(-1)):
        #ax[i].axis('off')
        ax_.set_xticks([])
        ax_.set_yticks([])
        ax_.annotate(labels[k], xy=(1, 0), xycoords='axes fraction', fontsize=12,
                xytext=(-5, 5), textcoords='offset points',
                ha='right', va='bottom',color=colors[k])
    plt.tight_layout()

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
    plt.show()
    plt.close()

def train_model(model, train_x, train_y, my_epochs, my_batch_size=None, m_split=0.1):
    """Feed a dataset into the model in order to train it."""
    history = model.fit(x=train_x, y=train_y, batch_size=my_batch_size, epochs=my_epochs, verbose = 1, validation_split=m_split)
    # Gather the model's trained weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    # The list of epochs is stored separately from the
    # rest of history.
    epochs = history.epoch

    # Isolate the root mean squared error for each epoch.
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]

    return epochs, rmse, history.history

def plot_card(x_val,y_val,prediction,k=5,seg_cmap=mpl.cm.viridis,spath=True):

    if spath is not None:
        figpath = spath + '/res_im/'
        os.mkdir(figpath)

    idx = np.arange(len(prediction))

    for j in random.choices(idx,k=k):
        predict_image = prediction[j].numpy()
        x_val_image = x_val[j]
        y_val_image = y_val[j]

        fig, ax = plt.subplots(figsize=(7,7),nrows=3,ncols=MASK_SIZE,sharex=True,sharey=True)
        idx = np.random.randint(0,prediction.shape[0])
        cmap = mpl.cm.binary
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        labels = ['cluster','agn','intersect']

        cmaps = [mpl.cm.Blues,mpl.cm.Reds,mpl.cm.Greens]
        for i in range(mask_num):

            input_ = y_val[idx,:,:,i]
            output_ = prediction[idx,:,:,i]
            contrast = output_-input_
            im = ax[0][i].imshow(input_,interpolation='none',cmap=cmaps[i],norm=norm)
            im = ax[1][i].imshow(output_,interpolation='none',cmap=cmaps[i],norm=norm)
            im = ax[2][i].imshow(contrast,interpolation='none',cmap=cmaps[i],norm=norm)

            ax[0][i].set_title(labels[i])
            for j in range(2):
                ax[j][i].set_aspect('equal')
                ax[j][i].set_xticks([])
                ax[j][i].set_yticks([])

        ax[0][0].set_ylabel('Input',fontsize=12)
        ax[1][0].set_ylabel('Ouput',fontsize=12)
        ax[2][0].set_ylabel('Contrast',fontsize=12)

        #cax = fig.add_axes([1, 0.25, 0.01, 0.5])
        #cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm = norm)

        plt.subplots_adjust(wspace=-.5,hspace=0)

        if spath:
            plt.savefig(figpath + 'deck_{}.png'.format(i),dpi=200,bbox_inches='tight')
        plt.show()
        plt.close()

def plot_results_v2(x_val,y_val,prediction,seg_cmap=mpl.cm.viridis,spath=None):

    if spath is not None:
        figpath = spath + '/res_im/'
        os.mkdir(figpath)

    idx = np.arange(len(prediction))

    for i in random.choices(idx,k=10):
        predict_image = prediction[i].numpy()
        x_val_image = x_val[i]
        y_val_image = y_val[i]

        fig, ax = plt.subplots(figsize =(12,5),nrows=1,ncols=4,sharex=True,sharey=True)

        cmap = mpl.cm.binary
        norm = mpl.colors.Normalize(vmin=-2.,vmax=1.)

        bounds = [-0.5,0.5,1.5,2.5,3.5]
        norm = mpl.colors.BoundaryNorm(bounds, seg_cmap.N, extend='both')

        """
        cmap = mpl.colors.ListedColormap(['white', 'red','green','blue'])
        bounds = [-0.5,0.5,1.5,2.5,3.5]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        """

        ax[0].set_title('Input')
        ax[0].imshow(np.log10(x_val_image[:,:,0]),cmap=cmap,interpolation='none',norm=norm)

        ax[1].set_title('Truth')
        ax[1].imshow(y_test_image[:,:,0], norm = norm, cmap = seg_cmap, aspect=1)

        ax[2].set_title('Prediction')
        ax[2].imshow(predict_image[:,:,0], norm = norm, cmap = seg_cmap ,aspect=1)

        ax[3].set_title('Contrast')
        ax[3].imshow(predict_image[:,:,0]-y_test_image[:,:,0], norm = norm, cmap = seg_cmap)

        cax = fig.add_axes([1, 0.25, 0.01, 0.5])
        cbar = mpl.colorbar.ColorbarBase(cax, cmap = seg_cmap, norm = norm, ticks=[0,1,2,3])
        for k in range(4):
            #ax[i].axis('off')
            ax[k].set_xticks([])
            ax[k].set_yticks([])

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05)
        if spath is not None:
            plt.savefig(figpath + 'deck_{}.png'.format(i),dpi=200,bbox_inches='tight')
        plt.show()
        plt.close()

def custom_loss(y_true, y_pred):
    labels = tf.cast(y_true,np.float32)
    logits = tf.cast(y_pred,np.float32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                   logits=logits)
    return loss

def p_GPU_stat():
    print("\n"+"*"*20)
    print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
    print("*"*20 + "\n")

def plot_metrics(history,spath,model_id,save):
    metrics = history.history.keys()
    N = len(metrics)/2
    cmap = mpl.cm.get_cmap('rainbow', N)    # PiYG

    colors = []
    for i in range(cmap.N):
        rgba = cmap(i)
        colors.append(mpl.colors.rgb2hex(rgba))

    fig, ax = plt.subplots(ncols=1,nrows=2,figsize=(6,5),sharex=True,sharey=True)
    for metric, color in zip(metrics,colors):

        ax[0].plot(history.history[metric],color=color,label=metric)
        ax[1].plot(history.history['val_' + metric],color=color,label=metric)

    xticks = np.arange(0,len(history.history[metric]))
    yticks = np.arange(0,1.1,0.2)
    ax[0].set_xticks(xticks), ax[1].set_xticks(xticks)
    ax[0].set_yticks(yticks), ax[1].set_yticks(yticks)

    ax[0].annotate('training set', xy=(1, 1), xycoords='axes fraction', fontsize=10,
            horizontalalignment='right', verticalalignment='bottom')
    ax[1].annotate('validation set', xy=(1, 1), xycoords='axes fraction', fontsize=10,
            horizontalalignment='right', verticalalignment='bottom')
    plt.xlabel('Epoch')
    plt.subplots_adjust(hspace=0)
    ax[0].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)


    if save:
        figpath = spath + '/accuracy_loss_{}.png'.format(model_id)
        plt.savefig(figpath,dpi=200,bbox_inches='tight')
        print('\n---> metrics plot saved to:',figpath)
    plt.show()
    plt.close()

def plot_metrics_v2(history,spath,model_id,metrics):
    N = len(metrics)
    cmap = mpl.cm.get_cmap('rainbow', N)    # PiYG

    colors = []
    for i in range(cmap.N):
        rgba = cmap(i)
        colors.append(mpl.colors.rgb2hex(rgba))
    fig, ax = plt.subplots(ncols=1,nrows=2,figsize=(6,5),sharex=True)
    for metric, color in zip(metrics,colors):
        ax[0].plot(history.history[metric],color=color,label=metric)
        ax[1].plot(history.history['val_' + metric],color=color)


    ax[0].annotate('training set', xy=(1, 1), xycoords='axes fraction', fontsize=10,
            horizontalalignment='right', verticalalignment='bottom')
    ax[1].annotate('validation set', xy=(1, 1), xycoords='axes fraction', fontsize=10,
            horizontalalignment='right', verticalalignment='bottom')
    plt.xlabel('Epoch')
    plt.subplots_adjust(hspace=0)
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=N)
    figpath = spath + '/accuracy_loss_{}.png'.format(model_id)


    if save:
        plt.savefig(figpath,dpi=200,bbox_inches='tight')
        print('\n---> metrics plot saved to:',figpath)
    if show:
        plt.show()
    plt.close()

def get_unet(IMG_SIZE,activation,kernel_initializer):
    concat = tf.keras.layers.Concatenate(axis=MASK_SIZE)
    # (batch_size,img_size,img_size,channel)
    # (0,1,2,3)
    input_shape = (IMG_SIZE,IMG_SIZE,1)

    x0 = keras.Input(shape=input_shape)

    elayer1 = keras.Sequential()
    elayer1.add(layers.Conv2D(64, kernel_size=(3, 3), activation=activation, padding='same',kernel_initializer=kernel_initializer))
    elayer1.add(layers.MaxPooling2D((2, 2)))

    x1 = elayer1(x0)

    elayer2 = keras.Sequential()
    elayer2.add(layers.Conv2D(32, kernel_size=(3, 3), activation=activation, padding='same',kernel_initializer=kernel_initializer))
    elayer2.add(layers.MaxPooling2D((2, 2)))

    x2 = elayer2(x1)

    elayer3 = keras.Sequential()
    elayer3.add(layers.Conv2D(16, kernel_size=(3, 3), activation=activation, padding='same',kernel_initializer=kernel_initializer))
    elayer3.add(layers.MaxPooling2D((2, 2)))

    x3 = elayer3(x2)

    elayer4 = keras.Sequential()
    elayer4.add(layers.Conv2D(8, kernel_size=(3, 3), activation=activation, padding='same',kernel_initializer=kernel_initializer))
    elayer4.add(layers.MaxPooling2D((2, 2)))

    x4 = elayer4(x3)

    dlayer1 = keras.Sequential()
    dlayer1.add(layers.Conv2DTranspose(8, kernel_size=(3, 3), activation=activation, padding='same',kernel_initializer=kernel_initializer))
    dlayer1.add(layers.UpSampling2D((2, 2)))

    x5 = dlayer1(x4)

    x6 = concat([x5,x3])

    dlayer2 = keras.Sequential()
    dlayer2.add(layers.Conv2DTranspose(16, kernel_size=(3, 3), activation=activation, padding='same',kernel_initializer=kernel_initializer))
    dlayer2.add(layers.UpSampling2D((2, 2)))

    x7 = dlayer2(x6)

    x8 = concat([x7, x2])

    dlayer3 = keras.Sequential()
    dlayer3.add(layers.Conv2DTranspose(32, kernel_size=(3, 3), activation=activation, padding='same',kernel_initializer=kernel_initializer))
    dlayer3.add(layers.UpSampling2D((2, 2)))

    x9 = dlayer3(x8)

    x10 = concat([x9, x1])

    dlayer4 = keras.Sequential()
    dlayer4.add(layers.Conv2DTranspose(64, kernel_size=(3, 3), activation=activation, padding='same',kernel_initializer=kernel_initializer))
    dlayer4.add(layers.UpSampling2D((2, 2)))

    x11 = dlayer4(x10)

    x12 = concat([x11, x0])

    x13 = layers.Conv2DTranspose(N_CLASSES-1, (1, 1), activation=activation)(x12)

    inputs = x0
    outputs = x13
    the_U = tf.keras.Model(inputs, outputs)
    return the_U

def main():

    # WELCOME
    p_GPU_stat()

    # DATASET SETTINGS
    # image
    IMG_SIZE = 128

    # dataset
    DATASET_SIZE = 100

    # noise
    p = False
    agn = True
    shift = False

    # segmentation
    sigma = 0.25

    # splitting/batching
    batch_size = 16
    validation_split = 0.1

    img_train, msk_train, paths_train, img_val, msk_val, paths_val = make_dataset(DATASET_SIZE=DATASET_SIZE)

    # COMPLILER SETTINGS

    # metrics
    all_metrics = [tf.keras.metrics.Accuracy(name='acc'),
                   tf.keras.metrics.MeanSquaredError(name='mse'),
                   tf.keras.metrics.MeanIoU(name='iou',num_classes=N_CLASSES)]
    metrics = all_metrics

    # loss
    #loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = tf.keras.losses.MeanSquaredError()

    # optimizer
    optimizer=tf.keras.optimizers.Adam(lr=1e-5)

    # layer
    activation = 'relu'

    # model
    epochs = 10

    crab = False

    if crab:
        modeldir = Cluster.mkdir_model(spath=home+'scratch/seg')
    else:
        modeldir = Cluster.mkdir_model(spath=home+'/repos/ClusNet/models/seg')
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

    model_id = os.path.basename(os.path.normpath(modeldir))

    csv_logger = CSVLogger(modeldir + '/history_{}.log'.format(model_id),
                           separator=',',
                           append=False)


    the_U = get_unet(IMG_SIZE=IMG_SIZE,activation=activation)
    the_U.compile(optimizer=optimizer,
                  loss=loss,metrics=all_metrics)
    the_U.summary()

    print("\n********LEARNING START********")
    start = time.time()

    # Train the model on the normalized training set.
    history = the_U.fit(x=img_train,
                        y=msk_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=2,
                        validation_split=validation_split)


    print("********LEARNING END********")

    elapsed = time.time() - start
    print("\nTIME:",str(timedelta(seconds=elapsed)))

    print("\nModel assets saved to:", modeldir)
    the_U.save(modeldir)


    prediction = the_U(img_val, training=False)
    plot_results(x_val=img_val,
                 y_val=msk_val,
                 prediction=prediction,
                 seg_cmap=mpl.cm.rainbow,
                 spath=modeldir)

        # plot loss and accuracy
    plot_metrics(history=history,
                 spath=modeldir,
                 model_id=model_id,
                metrics=metrics)

    del the_U
    gc.collect()
    K.clear_session()
    tf.compat.v1.reset_default_graph()


if __name__ == "__main__":
    main()
