def compile_model():

    # compiler
    opt = Adam()
    loss = tf.keras.losses.MeanAbsoluteError()
    metrics = ["accuracy"]
    epochs = 2

    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=metrics)
    return model

def generate_model(kernel_size, pool_size, activation, strides, input_shape,im_size=128):
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

    for y, y_model, label in zip([y_train,validation_y],[y_train_model,validation_y_model],['Training data','Validation data']):
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

    fig, ax = plt.subplots(ncols=1,nrows=2,figsize=(7,5),sharex=True)
    for idx, stat in zip([0,1],['accuracy','loss']):
        ax[idx].plot(history.history[stat])
        ax[idx].plot(history.history['val_' + stat])
        ax[idx].set_ylabel(stat)
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'],ncol=2,frameon=False)
    ax[0].set_ylim(0.40,1)
    ax[1].set_xlim(0,0.1)
    plt.tight_layout()

    plt.savefig(spath + '/accuracy_loss_{}.png'.format(model_id),dpi=200,
bbox_inches='tight')
    #plt.show()
    print('\n---> metrics plot saved to',spath)
