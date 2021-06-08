#!/anaconda3/envs/tf-cpu/bin/python
# -*- coding: utf-8 -*-
# Brianna Galgano
# code to create CNN of gaussian profile

# Imports
import matplotlib.pylab as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# statistics
import numpy as np
import random
from scipy import signal
import time
import os
# tensorflow
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import timedelta

from time import gmtime, strftime
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
    
    def to_pandas(self):
        """
        convert metadata (as recarray) to pandas DataFrame
        """
        self.meta = pd.DataFrame(self.meta)
        return
    
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

        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.12)

        plt.colorbar(im, cax=cax)
        #plt.show()
        plt.close()
        return None
    
def create_set(set_size=1000,im_size=128,noise=True,shift=True):
    dataset = []
    for i in range(set_size):
        std = np.random.randint(low=2,high=15,size=1)
        x = Profile(std=std,size=im_size)
        if noise:
            x.add_noise()
        if shift:
            x.shift()
            
        dataset.append(x)
    return dataset

def plot(dataset, spath):
    fig, axes = plt.subplots(nrows=10,ncols=10,figsize=(9,9))
    for i, ax in enumerate(axes.flat):
        prof = dataset[i]
        ax.imshow(prof.image,interpolation='none',cmap='magma')
        ax.set_yticks([])
        ax.set_xticks([])
    space = 0.05
    plt.tight_layout()
    plt.subplots_adjust(wspace=space,hspace=space)
    fpath = spath+'figs/dataset_10x10_view.png'
    plt.savefig(fpath,dpi=300)
    #plt.show()
    plt.close()
    #print("\nDataset preview saved to:", fpath)
    
def load_dataset(dataset,norm=True):
    # fit the keras model on the dataset
    size = len(dataset)
    data = np.array([prof.image for prof in dataset])
    labels = np.array([(prof.x,prof.y) for prof in dataset])

    idx = np.arange(0,size,1)
    test_idx = random.sample(list(idx),k=100)
    train_idx = np.delete(idx, test_idx)
    
    im_size = dataset[0].image.shape[0]
    if norm:
        norm_factor = im_size
    else:
        norm_factor = 1
    x_train, y_train = data[train_idx], labels[train_idx]/im_size
    x_test, y_test = data[test_idx], labels[test_idx]/im_size

    x_train = x_train.reshape(-1, im_size, im_size, 1)
    x_test = x_test.reshape(-1, im_size, im_size, 1)
    
    return x_train, y_train

def plot_1tot1(y_train,y_train_model,spath,s):
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(9,5),sharex=True,sharey=True)

    lims = [np.min([ax[0].get_xlim(), ax[0].get_ylim()]),
            np.max([ax[0].get_xlim(), ax[0].get_ylim()])]
    for idx, label in zip([0,1], ['X','Y']):
        ax[idx].scatter(y_train[:,idx],y_train_model[:,idx],s=5,marker=".",label='Training data')
        ax[idx].plot(lims, lims, 'k-', alpha=0.5, zorder=200)
        ax[idx].set_aspect('equal')
        ax[idx].set_xlim(lims), ax[idx].set_ylim(lims)
        ax[idx].set_xlabel('Truth {}'.format(label))
        ax[idx].set_ylabel('Predicted {}'.format(label))

    plt.legend()
    plt.tight_layout()
    fname = spath + 'figs/'
    
    plt.savefig(fname + '1to1_center_xy_{}.png'.format(s), dpi=250, bbox_inches='tight')
    
    #plt.show()
    plt.close()
    
    print("---> 1to1 plot saved to:", fname)

def generate_model(kernel_size, pool_size, activation, strides, input_shape,im_size=128):
    model = keras.Sequential(
        [
            keras.Input(shape=(im_size,im_size,1)),

            # 1. 3×3 convolution with 16 filters
            layers.Conv2D(filters=16, kernel_size=(9,9), activation=activation),

            # 2. 2×2, stride-2 max pooling
            layers.MaxPooling2D(pool_size=pool_size, strides=strides),

            # 3. 3×3 convolution with 32 filters
            layers.Conv2D(filters=32, kernel_size=kernel_size, activation=activation),

            # 4. 2×2, stride-2 max pooling
            layers.MaxPooling2D(pool_size=pool_size, strides=strides),

            # 5. 3×3 convolution with 64 filters
            layers.Conv2D(filters=64, kernel_size=kernel_size, activation=activation),

            # 6. 2×2, stride-2 max pooling
            layers.MaxPooling2D(pool_size=pool_size, strides=strides),

            # 7. global average pooling
            layers.GlobalAveragePooling2D(),

            # 8. 10% dropout
            layers.Dropout(0.1),

            # 9. 200 neurons, fully connected
            layers.Dense(units=200),

            # 10. 10% dropout
            layers.Dropout(0.1),

            # 11. 100 neurons, fully connected
            layers.Dense(units=100),

            # 12. 20 neurons, fully connected
            layers.Dense(units=20),

            # 13. output neuron
            layers.Dense(units=2)

        ]
    )
    #model.summary()
    return model
    
def plot_metrics(history,spath,s):
    fig, ax = plt.subplots(ncols=1,nrows=2,figsize=(6,5),sharex=True)
    for idx, stat in zip([0,1],['accuracy','loss']):
        ax[idx].plot(history.history[stat])
        ax[idx].plot(history.history['val_' + stat])
        ax[idx].set_ylabel(stat)
    plt.xlabel('epoch')
    plt.subplots_adjust(hspace=0)
    plt.legend(['train', 'valid'],ncol=2,frameon=False)
    fname = spath + 'figs/'
    plt.savefig(fname + 'accuracy_loss_{}.png'.format(s),dpi=300)
    #plt.show()
    print('\n---> metrics plot saved to',fname)

    
def main():
    home = '/home-1/bgalgan1@jhu.edu'
    spath = home + '/models/gauss/'
    im_size = 128
    train_size = 1000
    
    # create gaussian dataset
    dataset = create_set(im_size=128)
    
    # create training set preview figure
    #plot(dataset=dataset, spath=spath)
    
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
    opt = Adam(lr=0.0005)
    loss = 'mse'
    metrics = ["accuracy"]
    epochs = 300
    model.compile(optimizer=opt,
                  loss=loss, 
                  metrics=metrics)
    
    # model fitting
    validation_split = 0.2
    x_train, y_train = load_dataset(dataset)
    batch_size = len(dataset)
    
    print("\nLEARNING START.")
    start = time.time()
    history = model.fit(x=x_train,
                        y=y_train, 
                        epochs=epochs, 
                        batch_size=batch_size,
                        validation_split=validation_split,
                        verbose=2)

    
    # print time elapsed metrics
    print("LEARNING END.")
    elapsed = time.time() - start
    print("\nTime elasped:",str(timedelta(seconds=elapsed)) )
    
    # save current model with timedate stamp
    s = strftime("%Y-%m-%d_%H-%M-%S")
    
    fpath = spath + "~/assets/model_" + s
    model.save(fpath)

    print("Model assets saved to:", fpath)
    # plot loss and accuracy
    plot_metrics(history=history,spath=spath,s=s)
    
    # predict labels from model
    print("\nPredicting training labels:")
    y_train_model = model.predict(x_train,verbose=1)
    
    # plot 1-to-1
    plot_1tot1(y_train=y_train,
               y_train_model=y_train_model,
               spath=spath,
               s=s)
    

if __name__ == "__main__":
    main()

