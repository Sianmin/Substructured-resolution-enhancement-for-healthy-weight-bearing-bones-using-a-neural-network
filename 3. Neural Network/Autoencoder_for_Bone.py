import tensorflow as tf
import time
import os
from keras import backend as K
from keras import callbacks, regularizers
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
from Common import LiveDrawing, LoadingDatasets
from Networks import Networks as Networkclass
import numpy as np
from Common import LiveDrawing, Load_LRDV
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

    # Paramters
    [isGAN, useDis, useSED, epoch_show] = [False, False, False, True]
    [ratio, patch_n] = [10, 8]
    [epochs, batch_size, train_ratio] = [5, 64, 1]
    rp = ratio*patch_n
    now = time.localtime()

    # Model Information
    modelname="AUTOENCODER"
    filepath = f"Models/{modelname}/"

    # Initialization
    # LR_set_train, LR_set_test, HR_set_train, HR_set_test, BC_train, BC_test, SED_train, SED_test = LoadingDatasets(ratio, patch_n,
    #                                                                                           train_ratio, useDis, useSED)
    print("Complete data load")
    modelpath_g = filepath + "{epoch:02d}-G.hdf5"
    # os.makedirs(filepath, exist_ok=True)

    # Network Training
    TIME_START = datetime.now()
    Networks = Networkclass(ratio, patch_n, batch_size, isGAN)
    Autoencoder, encoder, decoder = Networks.AutoEncoder()
    plot_model(encoder, to_file=filepath + 'Encoder.png', show_shapes=True)
    plot_model(Autoencoder, to_file=filepath + 'Autoencoder.png', show_shapes=True)
    plot_model(decoder, to_file=filepath + 'Decoder.png', show_shapes=True)

    LR_DV = np.expand_dims(np.expand_dims(Load_LRDV(1, 1), axis=2), axis=0)
    Autoencoder.load_weights("Models/AUTOENCODER/01-G.hdf5")
    plt.gray()
    for i in range(0, 1000, rp):
        for j in range(0, 1000, rp):
            if np.sum(LR_DV[:, i:i+rp, j:j+rp, :]) >20:
                plt.subplot(121)
                plt.imshow(np.squeeze(np.squeeze(LR_DV[:, i:i + rp, j:j + rp, :])), vmin=0, vmax=1)
                plt.subplot(122)
                plt.imshow(np.squeeze(np.squeeze(Autoencoder.predict(LR_DV[:, i:i+rp, j:j+rp, :]))), vmin=0, vmax=1)
                plt.show(block=False)
                plt.pause(1)
    # checkpoint_g = ModelCheckpoint(modelpath_g, monitor='val_loss', verbose=1, mode='auto')
    # history_g = Autoencoder.fit(HR_set_train, HR_set_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[checkpoint_g])

    TIME_END = datetime.now()
    print(["#Training time:" ,(TIME_END-TIME_START)])