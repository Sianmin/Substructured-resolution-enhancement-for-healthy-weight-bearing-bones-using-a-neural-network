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

if __name__ == '__main__':
    # GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

    # Parameters
    useDis, useSED, epoch_show = False, False, False
    [ratio, patch_n] = [10, 8]
    [epochs, batch_size, train_ratio] = [20, 16, 10/11]
    modelname = "SRGAN_BVTV_AUTOENCODER"

    # Settings
    rp = ratio*patch_n
    now = time.localtime()
    filepath = f"Models/{modelname}-{now.tm_mon:02d}-{now.tm_mday:02d}-{now.tm_hour:02d}-{now.tm_min:02d}/"
    os.makedirs(filepath, exist_ok=True)
    LR_set_train, LR_set_test, HR_set_train, HR_set_test, BC_train, BC_test, SED_train, SED_test = LoadingDatasets(ratio, patch_n, train_ratio, useDis, useSED)
    print("Complete data load")
    modelpath = filepath + "{filepath}{epoch:02d}-{val_loss:.4f}.hdf5"
    modelpath_g = filepath + "{epoch:02d}-G.hdf5"
    modelpath_d = filepath + "{epoch:02d}-D.hdf5"
    modelpath_q = filepath + "{epoch:02d}-Q.hdf5"
    LVDR = LiveDrawing(filepath, LR_set_train, HR_set_train, LR_set_test, HR_set_test, BC_train, BC_test, SED_train, SED_test, ratio, patch_n, epoch_show=epoch_show, useDis=useDis, useSED=useSED)

    TIME_START = datetime.now()
    # Loading Networks
    Networks = Networkclass(ratio, patch_n, batch_size)
    encoder, generator, discriminator, combined = Networks.SRGAN_1()
    plot_model(generator, to_file=f'{filepath}generator.png', show_shapes = True)
    plot_model(discriminator, to_file=f'{filepath}discriminator.png', show_shapes = True)
    plot_model(combined, to_file=f'{filepath}combined.png', show_shapes = True)

    #Training GAN
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        num_batches = int(LR_set_train.shape[0] / batch_size)
        for i in range(num_batches):
            if i % 1000 == 0 : print(f"Batch {i+1}/{num_batches}")
            LR_batch = LR_set_train[i * batch_size:(i + 1)*batch_size]
            HR_batch = HR_set_train[i * batch_size:(i + 1) * batch_size]

            gen_img = generator.predict(LR_batch)

            valid = np.ones((batch_size, 1))
            not_valid = np.zeros((batch_size, 1))

            discriminator.train_on_batch(HR_batch, valid)
            discriminator.train_on_batch(gen_img, not_valid)

            feature = encoder.predict(HR_batch)
            combined.train_on_batch([LR_batch],[valid, feature, LR_batch]) # discirminator가 valid하고 feature 잘 나오고 BVTV 맞게 학습.
    #     #Callbacks
            if i % 5000 == 0:
                LVDR.on_epoch_end_GAN(epoch, generator)
        generator.save_weights(modelpath_g.format(epoch=epoch), True)
        discriminator.save_weights(modelpath_d.format(epoch=epoch), True)
    TIME_END = datetime.now()
    print(["#Training time:" ,(TIME_END-TIME_START)])