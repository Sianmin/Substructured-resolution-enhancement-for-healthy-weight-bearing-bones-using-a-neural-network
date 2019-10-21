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
from matplotlib.widgets import Slider, Button, RadioButtons

height, width = 2080, 1883

if __name__ == '__main__':
    # GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

    # Paramters
    [ratio, patch_n] = [10, 8]
    rp = ratio*patch_n

    # Model Information
    modelname="VAE"
    filepath = f"Models/{modelname}/"

    # Network Training
    TIME_START = datetime.now()
    Networks = Networkclass(ratio, patch_n)
    vae, encoder, decoder = Networks.VariationalAutoEncoder()
    vae.load_weights("Models/VAE/07-G.hdf5")

    # GUI 만들기