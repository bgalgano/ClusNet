from astropy.table import Table
from glob import glob
import os
from pathlib import Path
import matplotlib.pylab as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

import numpy as np
import pandas as pd
import sys
from astropy.io import fits
import random, string
import time
from os.path import expanduser
<<<<<<< HEAD
=======
import progressbar
import h5py
from copy import copy
import tensorflow as tf
from scipy import ndimage
from scipy.ndimage import gaussian_filter

>>>>>>> a14ae80b9a7b1e2f6a938b072e63d7cd7bdefb34
home = expanduser("~")
repodir = home + '/repos/ClusNet/'
clusterList = np.load(repodir + 'data/eROSITA_no_background/clusterList.npy')
clusterDir = repodir + 'data/eROSITA_no_background/'
GLOBDIR = clusterDir

sys.path.append(home+'/repos/ClusNet/code/modules/')

from ClusNet import dataset as ds
from ClusNet import model as m

class Cluster:
    def __init__(self, fpath=None):
        
        self.fpath = fpath
        self.lam = 0.1133929878

        if fpath is None:
            self.id = 'noise'
            self.image = np.zeros(shape=(384,384))
            self.cluster = np.zeros(shape=(384,384))
            self.meta = None
            self.meta_col = None
            
            self.M500 = None
            self.R500 = None
            self.Rpixel = None
            self.Tkev = None
            self.cat = 0
                        
        else:
            self.id = Path(fpath).stem
        
            with fits.open(fpath) as data:
                cluster_df = pd.DataFrame(data[0].data)
            
            self.cluster = cluster_df.to_numpy()           
            
            meta = clusterList[clusterList['id']==int(self.id)]
            self.meta = meta
            self.M500 = np.log10(meta['M500_msolh'])[0]
            self.R500 = np.log10(meta['r500_kpch'])[0]
            self.Rpixel = meta['R500_pixel'][0]
            self.Tkev = np.log10(meta['T_kev'])[0]
            self.cat = 1
            
            if type(self.meta) == np.ndarray:
                self.meta_col = meta.dtype.names
            else:
                self.meta_col = meta.columns()
                
        self.poisson = np.zeros(shape=(384,384))
        self.agn = np.zeros(shape=(384,384))
        self.image = self.poisson + self.agn + self.cluster
        
        self.w_pix  = len(self.image[:,0])
        self.xmid = int(self.w_pix/2)
        self.ymid = int(self.w_pix/2)
        self.mid_pix = (self.xmid,self.ymid)
        
     

        
    def __repr__(self):
        """
        print cluster metadata
        """
        return str(self.meta)
    
    def to_pandas(self):
        """
        convert metadata (as recarray) to pandas DataFrame
        """
        self.meta = pd.DataFrame(self.meta)
        return
    
    def add_poisson(self):
        """
        add Poisson noise to cluster image matrix
        """
        # generate a poisson distribution
        # 2D distribution in pixel size of image
        poisson_noise = np.random.poisson(lam=self.lam, size=self.image.shape)
        
        # image of just poisson noise
        self.poisson = poisson_noise
        self.image += self.poisson
        
    def add_agn(self,num=3):
        """
        add agn noise to cluster image matrix
        """
        agns = []
        for i in range(num):
            std = np.random.uniform(1.,2.,size=None)
            agn_ = ds.Profile(std=std,im_size=(384,384))  
            agn_.shift()
            agns.append(agn_.image)
        agns = np.array(agns).T
        self.agn = agns
        self.image += np.sum(self.agn,axis=-1)
        return agns
        
    def shift(self,delta=64):
        """
        shift cluster randomly within bounds of image
        """
        
        low = -1 * delta
        high = delta
        
        r = self.Rpixel
        xmid = self.xmid #center pixel index of 384x384 image
        ymid = self.ymid
        
        x_shift = np.random.randint(low=low,high=high,size=1)[0]
        y_shift = np.random.randint(low=low,high=high,size=1)[0]

        xi = xmid + x_shift
        yi = ymid + y_shift


        image_y_shift = np.roll(self.image,shift=y_shift,axis=0)
        self.image = np.roll(image_y_shift,shift=x_shift,axis=1)
        
        cluster_y_shift = np.roll(self.cluster,shift=y_shift,axis=0)
        self.cluster = np.roll(cluster_y_shift,shift=x_shift,axis=1)
        
        agn_y_shift = np.roll(self.agn,shift=y_shift,axis=0)
        self.agn = np.roll(agn_y_shift,shift=x_shift,axis=1)
        
        noise_y_shift = np.roll(self.poisson,shift=y_shift,axis=0)
        self.poisson = np.roll(noise_y_shift,shift=x_shift,axis=1)
        
        self.xmid = self.xmid + x_shift
        self.ymid = self.ymid + y_shift
        self.mid_pix = (self.xmid,self.ymid)
        return
    
    def add_border(self,pad_width=64):
        """
        add border or pad to image with constant values
        default is pad width of x pixels, constant is 0
        """
        image_border = np.pad(array=self.image,
                              constant_values=0,
                              mode='constant',
                              pad_width=pad_width)
        self.image = image_border
        self.w_pix  = len(image_border[:,0])
        self.xmid += int(pad_width)
        self.ymid += int(pad_width)
        self.mix_pix = (self.xmid,self.ymid)
        return
    
    def plot(self,circle=True,square=False,savefig=False,spath='../figs/eROSITA/'):
        """
        plot cluster image with radius and mass information
        """
        w_pix = self.w_pix
        mid_pix = self.mid_pix
        
        Rpixel = self.Rpixel
        cluster = self.image
        cluster_id = self.id

        plt.figure(figsize=(4,4))        
        cmap = plt.cm.rainbow
        cmap.set_under(color="white",alpha=0)

        agn_cmap = plt.cm.viridis
        agn_cmap.set_under(color="white",alpha=0)

        im = plt.imshow(np.log10(self.image),cmap=cmap,interpolation='none')
        #im = plt.imshow(np.log10(agn),cmap=agn_cmap,interpolation='none')

        ax = plt.gca()
        
        if self.meta is None:
            plt.title('Noise')
        else:
            
            cluster_row = self.meta
            log_m = np.log10(cluster_row['M500_msolh'])[0]
            plt.title(r'$\log\left(\frac{M_\mathrm{500c}}{h^{-1}\,M_\odot}\right) = $'+'{:.2f}'.format(log_m))
            
            if circle:
                circle = plt.Circle((self.xmid, self.ymid), self.Rpixel, color="red",fill=False,zorder=2000)
                ax.add_patch(circle)
        
            if square:
                circle = plt.Rectangle((self.xmid-384/2, self.ymid-384/2), 384, 384, color="black",fill=False,zorder=2000)
                ax.add_patch(circle)


        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.12)

        plt.colorbar(im, cax=cax)
        ax.set_xticks(np.arange(0,w_pix,100))
        ax.set_yticks(np.arange(0,w_pix,100))

        ax.set_xlabel("x"), ax.set_ylabel("y")
        ax.set_facecolor('white')
        
        plt.tight_layout()

        if savefig:
            plt.savefig('fpath' + file_name + '.png',dpi=250,bbox_inches='tight')

        ax.invert_yaxis()
        plt.show()

        plt.close()
        return None
    
<<<<<<< HEAD
def load_dataset(k='all', globdir=GLOBDIR,norm=True,addneg=True,savefpaths=True,validation_split=0.20,noise=True):
    clusglob = glob(globdir+'*.fits')
=======
    def get_mask(self,sigma=1.,kernel='gauss'):
                
        cluster_filter = gaussian_filter(self.cluster, sigma=sigma, mode='constant',cval=0.0)
        cluster_mask = (cluster_filter > 0.) * 1

        agn_all = np.sum(self.agn,axis=-1)
        agns_filter = gaussian_filter(agn_all, sigma=sigma, mode='constant',cval=0.0)
        agns_mask = (agns_filter > 0.) * 2

        catnum_map = cluster_mask + agns_mask
     
        return catnum_map
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 30, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
def read_cluster(clusfpath,shift=False,agn=True,poisson=True,sigma=2):
    x = Cluster(fpath=clusfpath)

    if shift:
        x.shift()
        
    if agn:
        num = np.random.randint(low=1,high=4,size=None)
        agns = x.add_agn(num=num)

    if poisson:
        x.add_poisson()
        
    image = x.image
    
    mask = x.get_mask(sigma=sigma)
    
    return image, mask

        
def load_keras_dataset(clusfpaths,shift=False,add_neg=True,noise=True):
    clusfpaths = get_fpaths(k=k,globdir=GLOBDIR)
            
    masks = []
    images = []
    labels=[]
    for i in range(len(clusfpaths)): 
        printProgressBar(total=len(clusfpaths),iteration=i)
        image, mask, label = read_cluster(clusfpath=clusfpaths[i],shift=shift,noise=noise)
        images.append(image)
        masks.append(mask)
        labels.append(label)
        
        if add_neg:
            neg_image, neg_mask, neg_label = read_cluster(clusfpath=None,shift=shift,noise=noise)
            images.append(image)
            masks.append(mask)
            labels.append(label)
            
    dataset = tf.data.Dataset.from_tensor_slices({"image": images, "segmentation mask": masks,"label":labels})
    print("dataset created:", dataset)
    
    return dataset


def get_fpaths(k='all',globdir=GLOBDIR):
        
    clusglob = glob(globdir+'*.fits')
    
    # determine which clusters are chosen
>>>>>>> a14ae80b9a7b1e2f6a938b072e63d7cd7bdefb34
    if type(k) == float and k < 1:
        k = int(k*len(clusglob))    
    if k == 'all':
        k = len(clusglob)
    clusfpaths = random.choices(clusglob,k=k)
    clusfpaths = np.array(clusfpaths)
    
    return clusfpaths
    
def load_dataset(mode='cat',k='all', globdir=GLOBDIR,norm=True,shift=True,addneg=True,save=True,validation_split=0.20,noise=True):
    clusfpaths = get_fpaths(k='all',globdir=GLOBDIR)

    x_train = []
    y_train = []
    print("Loading {:} clusters...".format(k))
    savepaths = []
    for i, clusfpath in enumerate(clusfpaths):   
        
        x = Cluster(fpath=clusfpath)
<<<<<<< HEAD
        if noise:
            x.add_noise()
        if norm:
            image = x.image / x.w_pix
        else:
            image = x.image
        x_train.append(image)
        y_train.append(1)
        savepaths.append(clusfpath)
=======
        x_copy = x
        
        if shift:
            x_copy.shift()
        if norm:
            image = x_copy.image / x_copy.w_pix
        for j in range(multiplicity):
            if noise:
                x_copy.add_noise()
            else:
                image = x_copy.image
            x_train.append(image)
            
            if mode == 'cat':
                y_train.append(1.)
            savepaths.append(clusfpath)
        printProgressBar(total=POS_NUM,iteration=i*multiplicity)
>>>>>>> a14ae80b9a7b1e2f6a938b072e63d7cd7bdefb34
        
    if addneg:
        print('-Adding {:} negatives...'.format(k))
        for i in range(0,k):
            x_noise = Cluster()
            if noise:
                x_noise.add_noise()
            if norm:
                image = x_noise.image / x_noise.w_pix
            else:
                image = x_noise.image
            x_train.append(image)
<<<<<<< HEAD
            y_train.append(0)
=======
            
            if mode == 'cat':
                y_train.append(0.)
>>>>>>> a14ae80b9a7b1e2f6a938b072e63d7cd7bdefb34
            savepaths.append('noise')

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    savepaths = np.array(savepaths)
    
    im_size = x_train[0].shape[0] # 384
    
    idx = np.array(list(range(k*2)))
    random.shuffle(idx) 
    
    x_train = x_train[idx]
    y_train = y_train[idx]
    
    savepaths = savepaths[idx]
    
    # split data set into validation and training
    # 80% training, 20% validation
    split_at = int(x_train.shape[0] * (1-validation_split))

    x_validation = x_train[split_at:].reshape(-1, 384, 384, 1)
    y_validation = y_train[split_at:].reshape((-1,1))
    
    x_train = x_train[:split_at].reshape(-1, 384, 384, 1)
    y_train = y_train[:split_at].reshape((-1,1))
    validation_data = (x_validation, y_validation)
    training_data = (x_train,y_train)
    
                         # print out label shapes
    print("Done.")
    print('\nTraining image shape:', training_data[0].shape)
    print('Training labels shape:', training_data[1].shape)
    print('y_train', y_train)
    
    print('\nValidation image shape:', validation_data[0].shape)
    print('Validation labels shape:', validation_data[1].shape)
    print('y_validation', y_train[split_at:])

    modeldir = mkdir_model(spath=home+'/repos/ClusNet/models/category')
    
    if savefpaths:
        print("\nSaving dataset paths to-->",modeldir)
        training_cluspaths = savepaths[:split_at]
        validation_clusfpaths = savepaths[split_at:]
        
        np.savetxt(fname=modeldir+"/tr_paths.txt",X=training_cluspaths,delimiter="\n",fmt="%s")
        np.savetxt(fname=modeldir+"/val_paths.txt",X=validation_clusfpaths,delimiter="\n",fmt="%s")
    return training_data, validation_data, modeldir

def mkdir_model(spath=home+'/repos/ClusNet/models/category'):
    # create directory to save model information
    model_id = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    modeldir = spath + '/' + model_id
    
    print("\nModel directory created -->",modeldir)
    os.mkdir(modeldir)
    
    return modeldir
    
def plot(spath="./",clusfpaths=None,globdir=None,size=5):
    nrows = size
    ncols = size
    k = nrows * ncols
    fig, axes = plt.subplots(nrows=nrows,
                             ncols=ncols,
                             figsize=(6,6))
    
    
    #clusglob = glob(globdir + '*.fits')
    #clusfpaths = random.choices(clusglob,k=k)
    
    dataset = []
    labels = []
    for clusfpath in clusfpaths:
        
        pos = Cluster(fpath=clusfpath)
  
        dataset.append(pos)
        labels.append(1)
        
        neg = Cluster()
        dataset.append(neg)
        labels.append(0)
        
    x = list(range(k*2))
    random.shuffle(x)
    idx = random.choices(x, k=k)
    
    dataset = np.array(dataset)[idx]
    labels = np.array(labels)[idx]
    
    for i, ax in enumerate(axes.flat):
        prof = dataset[i]

        image = prof.image/384
        cmap = plt.cm.viridis
        cmap.set_bad(color='gray')
        ax.imshow(image,interpolation='none',cmap=cmap)
        ax.text(0.90, 0.90,labels[i],
                horizontalalignment='center',
                verticalalignment='center',
                transform = ax.transAxes,
                color="white",
               weight='bold')
        ax.set_yticks([]), ax.set_xticks([])
        
    space = 0.05
    plt.tight_layout()
    plt.subplots_adjust(wspace=space,hspace=space)
    fpath = spath + 'dataset_{}x{}_view.png'.format(size,size)
    #plt.savefig(fpath,dpi=300)
    plt.show()
    plt.close()
    print("\nDataset preview saved to:", fpath)
