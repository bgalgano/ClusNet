#!/Users/mcp/opt/anaconda3/bin/python
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
            x.flip_lrud()
            x_lrud = copy.copy(x)
            
            dataset.extend([x_copy,x_lr,x_ud,x_lrud])
            
        else:
            dataset.append(x)
    return np.array(dataset)

def load_dataset(dataset,norm=True):
    # fit the keras model on the dataset
    size = len(dataset)
    data = np.array([prof.im for prof in dataset])
    labels = np.array([(prof.x,prof.y,prof.std[0]) for prof in dataset])

    idx = np.arange(0,size,1)
    #train_idx = np.delete(idx, test_idx)
    train_idx = idx
    
    im_size = dataset[0].im.shape[0]
    if norm:
        norm_factor = im_size
    else:
        norm_factor = 1
    
    x_train, y_train = data[train_idx], labels[train_idx]/im_size
    x_train = x_train.reshape(-1, im_size, im_size, 1)
    print("\nDataset loaded.")
    print("Image input shape:", x_train.shape)
    print("Label input shape:", y_train.shape)

    return x_train, y_train

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

def generate_model(kernel_size, pool_size, activation, strides, input_shape,im_size=128):
    model = keras.Sequential()
    
    model.add(keras.Input(shape=(im_size,im_size,1)))
    
    padding = 'valid'
    # 1. 3×3 convolution with 16 filters
    model.add(layers.Conv2D(filters=16, kernel_size=kernel_size, activation=activation,padding=padding))
    
    # 2. 2×2, stride-2 max pooling
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=strides))

    # 3. 3×3 convolution with 32 filters
    model.add(layers.Conv2D(filters=32, kernel_size=kernel_size, activation=activation,padding=padding))

    # 4. 2×2, stride-2 max pooling
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=strides))

    # 5. 3×3 convolution with 64 filters
    model.add(layers.Conv2D(filters=64, kernel_size=kernel_size, activation=activation,padding=padding))

    # 6. 2×2, stride-2 max pooling
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=strides))

    # 7. global average pooling
    model.add(layers.GlobalAveragePooling2D())

    # 8. 10% dropout
    model.add(layers.Dropout(0.1))

    # 9. 200 neurons, fully connected
    model.add(layers.Dense(units=200))

    # 10. 10% dropout
    model.add(layers.Dropout(0.1))

    # 11. 100 neurons, fully connected
    model.add(layers.Dense(units=100))

    # 12. 20 neurons, fully connected
    model.add(layers.Dense(units=20))

    # 13. output neuron
    model.add(layers.Dense(units=3))

    model.summary()
    return model
def plot_1tot1(y_train,y_train_model,validation_y,validation_y_model,spath,model_id):
    fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(10,3),sharey=False,sharex=False)

    for y, y_model, label in zip([y_train,validation_y],[y_train_model,validation_y_model],['Traini$
        for idx, ax_label in zip([0,1,2], ['X','Y','Sigma']):

            ax[idx].scatter(y[:,idx],y_model[:,idx],s=5,marker=".",label=label)

            lims = [np.min([ax[idx].get_xlim(), ax[idx].get_ylim()]),
                    np.max([ax[idx].get_xlim(), ax[idx].get_ylim()])]
            ax[idx].plot(lims, lims, 'k-', alpha=1, zorder=0,lw=1)
            ax[idx].set_aspect('equal')
            ax[idx].set_xlim(lims), ax[idx].set_ylim(lims)
            ax[idx].set_xlabel('Truth {}'.format(ax_label))
            ax[idx].set_ylabel('Predicted {}'.format(ax_label))

    plt.legend(frameon=False)
    plt.tight_layout()
    ax[2].set_ylim(0,0.25)
    ax[2].set_ylim(0,0.25)
    plt.subplots_adjust(wspace=0.01)
    plt.savefig(spath + '/1to1_center_xy_{}.png'.format(model_id), dpi=200,
bbox_inches='tight')

    print("\n---> 1to1 plot saved to:", spath)

    #plt.show()

    plt.close()
    
def plot_metrics(history,spath,model_id):
    
    fig, ax = plt.subplots(ncols=1,nrows=2,figsize=(6,5),sharex=True)
    for idx, stat in zip([0,1],['accuracy','loss']):
        ax[idx].plot(history.history[stat])
        ax[idx].plot(history.history['val_' + stat])
        ax[idx].set_ylabel(stat)
    plt.xlabel('epoch')
    plt.subplots_adjust(hspace=0)
    plt.legend(['train', 'valid'],ncol=2,frameon=False)
    
    plt.savefig(spath + '/accuracy_loss_{}.png'.format(model_id),dpi=200, bbox_inches='tight')
    #plt.show()
    print('\n---> metrics plot saved to',spath)

    
def main():
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

    mode = 'cpu'
    home = os.path.expanduser("~")
    if mode == 'crab':
        home = '/home-1/bgalgan1@jhu.edu'
    if mode == 'cpu':
        home = home + '/repos/neural/CNNs/center_pix'
        
    spath = home + '/models'
    
    im_size = 128
    set_size = 1000
    
    # create gaussian dataset
    dataset = create_set(im_size=im_size,set_size=set_size)
    
    # create training set preview figure
    # plot(dataset=dataset, spath=spath)
        
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
    loss = tf.keras.losses.MeanSquaredError()
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
    
    # serialize weights to HDF5
    model.save(model_dir + "/model_{}".format(model_id))
    print("Model assets saved to:", model_dir)
    
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(model_dir+"/model_{}/{}.yaml".format(model_id,model_id), "w") as yaml_file:
        yaml_file.write(model_yaml)

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

 