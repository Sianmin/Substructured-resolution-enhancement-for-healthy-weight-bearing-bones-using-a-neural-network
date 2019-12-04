import tensorflow as tf
from keras import backend as K
from Common import LiveDrawing, LoadingDatasets, Load_LRDV
from Networks import Networks as Networkclass
from Quilt import Quilt as Quilt
import matplotlib.pyplot as plt

# GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

# Initialization
[isGAN, epoch_show] = [False, False]
[ratio, patch_n] = [10, 8]
[epochs, batch_size] = [10, 32]
rp = ratio*patch_n

plt.gray()
for filepath in ["Models/ResNet_MAE/"]:
    gpath = f"{filepath}{epochs:02d}-G.hdf5"
    Networks = Networkclass(ratio, patch_n)
    LVDR = LiveDrawing(filepath, [], [], [], [], ratio, patch_n, useDis=False)
    GN = Networks.Generator_SRGAN()
    GN.load_weights(gpath)
    for subject in range(1, 12):
        rec_img = LVDR.predictModel(GN, subject)
        plt.imsave(f"{filepath}subject{subject}.png", rec_img[:,:,0], vmin=0, vmax=1, origin='lower')
    Quilt(filepath, GN)