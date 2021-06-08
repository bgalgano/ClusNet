#!/home-net/home-1/bgalgan1@jhu.edu/code/tf-new
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
import matplotlib as mpl
import glob
import random
import sys
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

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

home = expanduser("~")
repodir = home + '/repos/ClusNet/'
clusterList = np.load(repodir + 'data/eROSITA_no_background/clusterList.npy')
clusterDir = repodir + 'data/eROSITA_no_background/'
GLOBDIR = clusterDir

def make_dataset(paths,validation_split,modeldir):
    data_length = len(paths)
    
    image_list = []
    mask_list = []
    print("\nLoading {} cluster(s)...".format(len(paths)))
    for i, clusfpath in enumerate(paths):

        Cluster.printProgressBar(total=len(paths),iteration=i)
        image, mask = Cluster.read_cluster(clusfpath,shift=False,agn=True,poisson=False,sigma=0.5)

        image_tensor = tf.constant(image)

        image_tensor = image_tensor[..., tf.newaxis]
        image_list += [image_tensor]

        mask_tensor = tf.constant(mask)

        mask_tensor = mask_tensor[..., tf.newaxis]
        mask_list += [mask_tensor] 
        
    image_array = np.array(image_list)
    mask_array = np.array(mask_list)

    # trainset
    split_idx = int(validation_split*data_length)
    x_train = image_array[:split_idx]
    y_train = mask_array[:split_idx]

    # test set
    x_test = image_array[split_idx:]
    y_test = mask_array[split_idx:]
    
    paths_train = paths[:split_idx]
    paths_test = paths[split_idx:]
    
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
    
    print("\nx_train shape:",x_train.shape)
    print("y_train shape:",y_train.shape)
    
    print("\nx_test shape:",x_test.shape)
    print("y_test shape:",x_test.shape)

    return x_train,y_train,paths_train,x_test,y_test,paths_test
    
def get_x(output_shape):
    concat = tf.keras.layers.Concatenate()

    x0 = keras.Input(shape=output_shape)

    elayer1 = keras.Sequential()
    elayer1.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'))
    elayer1.add(layers.MaxPooling2D((2, 2)))

    x1 = elayer1(x0)


    elayer2 = keras.Sequential()
    elayer2.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same'))
    elayer2.add(layers.MaxPooling2D((2, 2)))

    x2 = elayer2(x1)

    elayer3 = keras.Sequential()
    elayer3.add(layers.Conv2D(16, kernel_size=(3, 3), activation="relu", padding='same'))
    elayer3.add(layers.MaxPooling2D((2, 2)))

    x3 = elayer3(x2)

    elayer4 = keras.Sequential()
    elayer4.add(layers.Conv2D(8, kernel_size=(3, 3), activation="relu", padding='same'))
    elayer4.add(layers.MaxPooling2D((2, 2)))

    x4 = elayer4(x3)

    dlayer1 = keras.Sequential()
    dlayer1.add(layers.Conv2DTranspose(8, kernel_size=(3, 3), activation="relu", padding='same'))
    dlayer1.add(layers.UpSampling2D((2, 2)))

    x5 = dlayer1(x4)

    x6 = concat([x5,x3])

    dlayer2 = keras.Sequential()
    dlayer2.add(layers.Conv2DTranspose(16, kernel_size=(3, 3), activation="relu", padding='same'))
    dlayer2.add(layers.UpSampling2D((2, 2)))

    x7 = dlayer2(x6)

    x8 = concat([x7, x2])

    dlayer3 = keras.Sequential()
    dlayer3.add(layers.Conv2DTranspose(32, kernel_size=(3, 3), activation="relu", padding='same'))
    dlayer3.add(layers.UpSampling2D((2, 2)))

    x9 = dlayer3(x8)

    x10 = concat([x9, x1])

    dlayer4 = keras.Sequential()
    dlayer4.add(layers.Conv2DTranspose(64, kernel_size=(3, 3), activation="relu", padding='same'))
    dlayer4.add(layers.UpSampling2D((2, 2)))

    x11 = dlayer4(x10)

    x12 = concat([x11, x0])

    x13 = layers.Conv2DTranspose(1, kernel_size=(3, 3), activation="relu", padding='same')(x12)

    return x0, x13

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

def plot_results(x_test,y_test,prediction,seg_cmap=mpl.cm.viridis,spath=None):
    
    if spath is not None:
        figpath = spath + '/res_im/'
        os.mkdir(figpath)

    idx = np.arange(len(prediction))

    for i in random.choices(idx,k=10):
        predict_image = prediction[i].numpy()
        x_test_image = x_test[i]
        y_test_image = y_test[i]

        fig, ax = plt.subplots(figsize =(12,5),nrows=1,ncols=4,sharex=True,sharey=True)
        ax[0].set_title('Input')

        cmap = mpl.cm.binary
        norm = mpl.colors.Normalize(vmin=-2.,vmax=1.)

        ax[0].imshow(np.log10(x_test_image[:,:,0]),cmap=cmap,interpolation='none',norm=norm)

        bounds = [-0.5,0.5,1.5,2.5,3.5]
        norm = mpl.colors.BoundaryNorm(bounds, seg_cmap.N, extend='both')

        """
        cmap = mpl.colors.ListedColormap(['white', 'red','green','blue'])
        bounds = [-0.5,0.5,1.5,2.5,3.5]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        """
        ax[2].set_title('Prediction')
        ax[2].imshow(predict_image[:,:,0], norm = norm, cmap = seg_cmap,aspect=1)


        ax[1].set_title('Truth')
        ax[1].imshow(y_test_image[:,:,0], norm = norm, cmap = seg_cmap,aspect=1)


        ax[-1].set_title('Contrast')
        ax[-1].imshow(predict_image[:,:,0]-y_test_image[:,:,0], norm = norm, cmap = seg_cmap)

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
        #plt.show()
        plt.close()
        
def p_GPU_stat():
    print("\n"+"*"*20)
    print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
    print("*"*20 + "\n")

        
def main():
    
    p_GPU_stat()
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    
    epochs = 2
    batch_size = 16
    validation_split = 0.1
    k=0.1
    
    modeldir = Cluster.mkdir_model(spath=home+'/repos/ClusNet/models/seg')
    model_id = os.path.basename(os.path.normpath(modeldir))
    
    paths = Cluster.get_fpaths(k=k,globdir=GLOBDIR)

    x_train,y_train,paths_train,x_test,y_test,paths_test = make_dataset(paths=paths,
                                                                        validation_split=0.90,
                                                                        modeldir=modeldir)
    
    csv_logger = CSVLogger(modeldir + '/history_{}.log'.format(model_id),
                           separator=',',
                           append=False)
    
    
    image_size = 384
    output_shape = (image_size, image_size, 1)
    input_shape = (image_size, image_size, 1)

    # compile settings
    learning_rate = 0.0005
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss = tf.keras.losses.MeanSquaredError()
    #metrics = [tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.Accuracy()]
    metrics = [tf.keras.metrics.Accuracy()]
 
    x0, x13 = get_x(output_shape=output_shape)
    the_U = keras.Model(x0, x13)
    the_U.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    #the_U.summary()
    
    print("\n********LEARNING START********")
    start = time.time()

    # Train the model on the normalized training set.
    history = the_U.fit(x=x_train, 
                        y=y_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        verbose=2, 
                        validation_split=validation_split,
                        callbacks=[csv_logger])

    print("********LEARNING END********")

    elapsed = time.time() - start
    print("\nTIME:",str(timedelta(seconds=elapsed)))
    
    print("\nModel assets saved to:", modeldir)
    the_U.save(modeldir)
    
    # plot loss and accuracy
    m.plot_metrics(history=history,
                 spath=modeldir,
                 model_id=model_id)
    
    prediction = the_U(x_test, training=False)
    plot_results(x_test=x_test,
                 y_test=y_test,
                 prediction=prediction,
                 seg_cmap=mpl.cm.rainbow,
                 spath=modeldir)
    
    del the_U
    gc.collect()
    K.clear_session()
    tf.compat.v1.reset_default_graph() 
    
    os.system('say "dank"')
    print("\n")
    
if __name__ == "__main__":
    main()
    
