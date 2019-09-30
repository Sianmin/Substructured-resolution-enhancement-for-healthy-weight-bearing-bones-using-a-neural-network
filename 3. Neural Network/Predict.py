import keras
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
import matplotlib.pyplot as plt
import numpy as np
from keras.utils.io_utils import HDF5Matrix

# GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

# Initialization
[isGAN, epoch_show] = [False, False]
[ratio, patch_n] = [10, 8]
[epochs, batch_size] = [10, 32]
rp = ratio*patch_n


filepath = "Models/Quilting1-09-06-01-23/"
gpath = filepath+"01-G.hdf5"
qpath = filepath+"01-Q.hdf5"
LR_set_train, LR_set_test, HR_set_train, HR_set_test = LoadingDatasets(ratio, patch_n, 1/100, False)
[BC_train, BC_test] = [0, 0]
Networks = Networkclass(ratio, patch_n, batch_size, isGAN)
LVDR = LiveDrawing(filepath, LR_set_train, HR_set_train, LR_set_test, HR_set_test, BC_train, BC_test, ratio, patch_n, useDis=False)
GN, QN = Networks.SangMinNet_20190830(rp, int(rp/2))
GN.load_weights(gpath)
QN.load_weights(qpath)
# crops = LVDR.predcit_Quilt(GN, QN)

plt.gray()
In_set = HDF5Matrix("%sDataset_QN.hdf5" % (filepath), 'In')[:]
Out_set = HDF5Matrix("%sDataset_QN.hdf5" % (filepath), 'Out')[:]
i = 0
while 1:
    plt.subplot(121)
    plt.imshow(np.squeeze(In_set[i, :, :, :]))
    ig = QN.predict(np.expand_dims(In_set[i, :, :, :], axis = 0))
    plt.subplot(122)
    plt.imshow(np.squeeze(ig[0, :, :, :]))
    plt.show(block=False)
    plt.pause(1)
    i+=1
# for i in range(crops.shape[0]):
#     print(crops.shape)
#     plt.imshow(np.squeeze(crops[i, :, :, :]), vmin=0, vmax=1)
#     plt.pause(1)
#     plt.show(block=False)