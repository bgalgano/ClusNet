
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
            #x.flip_lrud()


            dataset.extend([x_copy,x_lr,x_ud])

        else:
            dataset.append(x)
    return np.array(dataset)

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
