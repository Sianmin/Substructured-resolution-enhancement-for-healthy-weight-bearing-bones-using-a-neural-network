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

    # Initialization
    [isGAN, useDis, useSED, epoch_show] = [True, False, False, False]
    [ratio, patch_n] = [10, 8]
    [epochs, batch_size, train_ratio] = [20, 16, 10/11]
    rp = ratio*patch_n
    now = time.localtime()
    modelname="SRGAN_BVTV"
    filepath = "Models/%s-%02d-%02d-%02d-%02d/" % (modelname, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
    os.makedirs(filepath, exist_ok=True)
    LR_set_train, LR_set_test, HR_set_train, HR_set_test, BC_train, BC_test, SED_train, SED_test = LoadingDatasets(ratio, patch_n,
                                                                                              train_ratio, useDis, useSED)
    print("Complete data load")
    modelpath = filepath + "{epoch:02d}-{val_loss:.4f}.hdf5"
    modelpath_g = filepath + "{epoch:02d}-G.hdf5"
    modelpath_d = filepath + "{epoch:02d}-D.hdf5"
    modelpath_q = filepath + "{epoch:02d}-Q.hdf5"
    LVDR = LiveDrawing(filepath, LR_set_train, HR_set_train, LR_set_test, HR_set_test, BC_train, BC_test, SED_train, SED_test, ratio, patch_n, epoch_show=epoch_show, isGAN=isGAN, useDis=useDis, useSED=useSED)

    # Network Training
    Networks = Networkclass(ratio, patch_n, batch_size, isGAN)
    TIME_START = datetime.now()
    if not isGAN: #GAN이 아닌 네트워크로 트레이닝
        GN = Networks.AdjNet_20190926()
        plot_model(GN, to_file=filepath + 'Generator.png', show_shapes=True)
        # plot_model(QN, to_file=filepath + 'Quilter.png', show_shapes=True)
        checkpoint_g = ModelCheckpoint(modelpath_g, monitor='val_loss', verbose=1, mode='auto')
        # checkpoint_q = ModelCheckpoint(modelpath_q, monitor='val_loss', verbose=1, mode='auto')
        history_g = GN.fit([LR_set_train, SED_train], HR_set_train,
                           validation_data=([LR_set_test, SED_test], HR_set_test),
                           epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[LVDR, checkpoint_g])

    else: #GAN인 네트워크로 트레이닝
        #Loading Networks
        vgg, generator, discriminator, combined = Networks.SRGAN_1()
        plot_model(generator, to_file=filepath + 'generator.png', show_shapes = True)
        plot_model(discriminator, to_file=filepath + 'discriminator.png', show_shapes = True)
        plot_model(combined, to_file=filepath + 'combined.png', show_shapes = True)

        #Training

        for epoch in range(1, epochs + 1):
            print('Epoch {}/{}'.format(epoch, epochs))
            num_batches = int(LR_set_train.shape[0] / batch_size)
            for i in range(num_batches):
                print('Batch {}/{}'.format(i+1, num_batches))
                LR_batch = LR_set_train[i * batch_size:(i + 1)*batch_size]
                HR_batch = HR_set_train[i * batch_size:(i + 1) * batch_size]

                gen_img = generator.predict(LR_batch)

                valid = np.ones((batch_size, 1))
                not_valid = np.zeros((batch_size, 1))

                discriminator.train_on_batch(HR_batch, valid)
                discriminator.train_on_batch(gen_img, not_valid)

                feature = vgg.predict(np.concatenate((HR_batch, HR_batch, HR_batch), axis=3))
                combined.train_on_batch([LR_batch, HR_batch],[valid, feature])
        #     #Callbacks
            LVDR.on_epoch_end_GAN(epoch, generator)
            generator.save_weights(modelpath_g.format(epoch=epoch), True)
            discriminator.save_weights(modelpath_d.format(epoch=epoch), True)
    TIME_END = datetime.now()
    print(["#Training time:" ,(TIME_END-TIME_START)])