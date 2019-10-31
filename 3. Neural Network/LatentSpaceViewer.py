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
import scipy.interpolate as spi

height, width = 2080, 1883

if __name__ == '__main__':
    # GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

    # Paramters
    [ratio, patch_n, latent_dim] = [10, 8, 64]
    rp = ratio*patch_n
    mode = 2 # 1: Encoding, 2: Latent Interpolation HR 3: Decoding
    # Model Information
    modelname=f"VAE-{latent_dim}"
    filepath = f"Models/{modelname}/"

    # Network Training
    TIME_START = datetime.now()
    Networks = Networkclass(ratio, patch_n)
    vae, encoder, decoder = Networks.VariationalAutoEncoder()
    vae.load_weights(f"{filepath}50-G.hdf5")
    LR_DV = Load_LRDV(1, 1)

    if mode == 1:
        [x, y] = [1000, 1500]
        patch = np.reshape(LR_DV[y:y+rp, x:x+rp], (1, rp, rp, 1))
        LV = encoder.predict(patch)
        print(LV)
    elif mode == 2:
        LR_patch = np.zeros((1, rp, rp, 1))
        HR_DV = np.zeros((height, width, 1))
        grid_x, grid_y = np.mgrid[0:height, 0:width]
        Latent_Map = np.zeros((height, width, latent_dim))
        point=[]
        value=[]
        [NX, NY, patch_n] =[width, height, rp]
        for winy in range(0, NY, int(patch_n/8)):
            for winx in range(0, NX, int(patch_n/8)):
                i = NY - patch_n if winy > NY - patch_n else winy
                j = NX - patch_n if winx > NX - patch_n else winx
                LR_patch[0, :, :, 0] = LR_DV[i:i+patch_n, j:j+patch_n]
                predict_patch = vae.predict(LR_patch)
                LV = encoder.predict(LR_patch)[2] # mean, std, z 순으로 나옴.
                point.append((i, j )) # y, x 순
                value.append((LV))
                predict_patch = np.reshape(predict_patch, (1, rp, rp))
                HR_DV[i: i + rp, j: j +rp, 0] = predict_patch
        plt.gray()
        plt.imsave(f"{filepath}before_interpolation.png", np.squeeze(HR_DV), origin='lower')
        point = np.array(point)
        value = np.squeeze(np.array(value), axis = 1)
        HR_DV = np.zeros((height, width, 1))
        for i in range(latent_dim):
            Latent_Map[:, :, i] = spi.griddata(point, value[:, i], (grid_x, grid_y), method='cubic')
            print(i)
        index = 0
        for winy in range(0, NY, 5):
            for winx in range(0, NX, 5):
                i = NY - patch_n if winy > NY - patch_n else winy
                j = NX - patch_n if winx > NX - patch_n else winx
                patch = decoder.predict(np.expand_dims(Latent_Map[i, j, :], axis = 0))
                HR_DV[i:i+rp, j:j+rp, :] = patch
                print((i, j))
                index += 1
                plt.imsave(f"{filepath}/processing/{index}after_interpolation.png", np.squeeze(patch), origin='lower')
        plt.imsave(f"{filepath}/after_interpolation.png", np.squeeze(HR_DV), origin='lower')
    else:

        encoding = np.reshape(LV, (1, 64))
        patch = decoder.predict(encoding)
        plt.gray()
        print(patch.shape)
        plt.imshow(np.squeeze(patch, axis=(0, 3)))
        plt.show()