import tensorflow as tf
import time
import os
from keras import backend as K
from keras import callbacks, regularizers
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
from Common import LiveDrawing, LoadingDatasets, Load_LRDV
from Networks import Networks as Networkclass
import numpy as np
import matplotlib.pyplot as plt

height, width = 2080, 1883

if __name__ == '__main__':
    # GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

    # Paramters
    [useDis, useSED, test] = [False, False, False]
    [ratio, patch_n] = [10, 8]
    [epochs, batch_size, train_ratio] = [50, 64, 1]
    rp = ratio*patch_n
    now = time.localtime()

    # Model Information
    modelname="VAE-32"
    filepath = f"Models/{modelname}/"

    # Initialization
    print("Complete data load")
    modelpath_g = filepath + "{epoch:02d}-G.hdf5"
    os.makedirs(filepath, exist_ok=True)

    # Network Training
    TIME_START = datetime.now()
    Networks = Networkclass(ratio, patch_n)
    vae, encoder, decoder = Networks.VariationalAutoEncoder()
    plot_model(encoder, to_file=filepath + 'Encoder.png', show_shapes=True)
    plot_model(vae, to_file=filepath + 'VAE.png', show_shapes=True)
    plot_model(decoder, to_file=filepath + 'Decoder.png', show_shapes=True)

    def call_test():
        LR_DV = np.expand_dims(np.expand_dims(Load_LRDV(1, 1), axis=2), axis=0)
        vae.load_weights(f"{filepath}{epochs:02d}-G.hdf5")
        plt.gray()
        index = 0
        for i in range(0, height, rp):
            for j in range(0, width, rp):
                if np.sum(LR_DV[:, i:i + rp, j:j + rp, :]) > rp * rp * 0.2 and np.sum(
                        LR_DV[:, i:i + rp, j:j + rp, :]) < rp * rp * 0.7:
                    fig = plt.figure(1)
                    plt.subplot(121)
                    plt.imshow(np.squeeze(np.squeeze(LR_DV[:, i:i + rp, j:j + rp, :])), vmin=0, vmax=1)
                    plt.subplot(122)
                    plt.imshow(np.squeeze(np.squeeze(vae.predict(LR_DV[:, i:i + rp, j:j + rp, :]))), vmin=0, vmax=1)
                    fig.savefig(f"{filepath}index-{index:03d}.png", dpi=1200)
                    index += 1
    if test:
        call_test()
    else:
        LR_set_train, LR_set_test, HR_set_train, HR_set_test, BC_train, BC_test, SED_train, SED_test = LoadingDatasets(ratio, patch_n, train_ratio, useDis, useSED)
        checkpoint = ModelCheckpoint(modelpath_g, verbose=1, mode='auto')
        history = vae.fit(HR_set_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[checkpoint])
        call_test()

    TIME_END = datetime.now()
    print(["#Training time:" ,(TIME_END-TIME_START)])