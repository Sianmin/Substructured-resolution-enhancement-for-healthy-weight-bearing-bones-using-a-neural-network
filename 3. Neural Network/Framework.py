import tensorflow as tf
import time
import os
from keras import backend as K
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
from Common import LiveDrawing, LoadingDatasets
from Networks import Networks as Networkclass
from keras_radam import RAdam
import numpy as np

if __name__ == '__main__':
    # GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

    # Parameters
    useDis, useSED, epoch_show, isGAN = False, False, False, False
    [ratio, patch_n, epochs, batch_size] = [10, 8, 10, 16]
    rp = ratio * patch_n
    LR_set_train, HR_set_train, Dis_train, SED_train = LoadingDatasets(ratio, patch_n, useDis, useSED)
    print("Complete data load")
    optimizer = RAdam()
    # Settings
    now = time.localtime()
    for case in range(5, 7):
        if case == 1:
            modelname = "ResNet_MAE"

            filepath = f"Models/{modelname}/"
            os.makedirs(filepath, exist_ok=True)
            LVDR = LiveDrawing(filepath, LR_set_train, HR_set_train, Dis_train, SED_train, ratio, patch_n, epoch_show=epoch_show, useDis=useDis, useSED=useSED)
            modelpath = filepath + "{filepath}{epoch:02d}-{val_loss:.4f}.hdf5"
            modelpath_g = filepath + "{epoch:02d}-G.hdf5"

            TIME_START = datetime.now()
            # Loading Networks
            Networks = Networkclass(ratio, patch_n)
            GN = Networks.Generator_SRGAN()
            plot_model(GN, to_file=filepath + 'Generator.png', show_shapes=True)
            checkpoint_g = ModelCheckpoint(modelpath_g, monitor='val_loss', verbose=1, mode='auto')
            optimizer = RAdam()
            GN.compile(optimizer=optimizer, loss='mae')
            history_g = GN.fit(LR_set_train, HR_set_train,
                               epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[LVDR, checkpoint_g])
            TIME_END = datetime.now()
            print(["#Training time:", (TIME_END - TIME_START)])
        if case == 2:
            modelname = "ResNet_MSE"
            now = time.localtime()
            filepath = f"Models/{modelname}/"
            os.makedirs(filepath, exist_ok=True)
            LVDR = LiveDrawing(filepath, LR_set_train, HR_set_train, Dis_train, SED_train, ratio, patch_n,
                               epoch_show=epoch_show, useDis=useDis, useSED=useSED)
            modelpath = filepath + "{filepath}{epoch:02d}-{val_loss:.4f}.hdf5"
            modelpath_g = filepath + "{epoch:02d}-G.hdf5"
            modelpath_d = filepath + "{epoch:02d}-D.hdf5"
            modelpath_q = filepath + "{epoch:02d}-Q.hdf5"

            TIME_START = datetime.now()
            # Loading Networks
            Networks = Networkclass(ratio, patch_n)

            GN = Networks.Generator_SRGAN()
            plot_model(GN, to_file=filepath + 'Generator.png', show_shapes=True)
            checkpoint_g = ModelCheckpoint(modelpath_g, monitor='val_loss', verbose=1, mode='auto')
            optimizer = RAdam()
            GN.compile(optimizer=optimizer, loss='mse')
            history_g = GN.fit(LR_set_train, HR_set_train,
                               epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[LVDR, checkpoint_g])
        if case == 6:
            modelname = "ResNet_MSE+MAE"
            now = time.localtime()
            filepath = f"Models/{modelname}/"
            os.makedirs(filepath, exist_ok=True)
            LVDR = LiveDrawing(filepath, LR_set_train, HR_set_train, Dis_train, SED_train, ratio, patch_n,
                               epoch_show=epoch_show, useDis=useDis, useSED=useSED)
            modelpath = filepath + "{filepath}{epoch:02d}-{val_loss:.4f}.hdf5"
            modelpath_g = filepath + "{epoch:02d}-G.hdf5"
            modelpath_d = filepath + "{epoch:02d}-D.hdf5"
            modelpath_q = filepath + "{epoch:02d}-Q.hdf5"

            TIME_START = datetime.now()
            # Loading Networks
            Networks = Networkclass(ratio, patch_n)

            GN = Networks.Generator_SRGAN()
            plot_model(GN, to_file=filepath + 'Generator.png', show_shapes=True)
            checkpoint_g = ModelCheckpoint(modelpath_g, monitor='val_loss', verbose=1, mode='auto')


            def LOSS_MSE_MAE(y_true, y_pred):
                MSE = K.mean(K.square(y_pred-y_true))
                MAE = K.mean(K.abs(y_pred - y_true))
                return MSE + MAE
            GN.compile(optimizer=optimizer, loss=LOSS_MSE_MAE)
            history_g = GN.fit(LR_set_train, HR_set_train,
                               epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[LVDR, checkpoint_g])
        if case == 4:
            modelname = "SRGAN"
            filepath = f"Models/{modelname}/"
            os.makedirs(filepath, exist_ok=True)
            modelpath = filepath + "{filepath}{epoch:02d}-{val_loss:.4f}.hdf5"
            modelpath_g = filepath + "{epoch:02d}-G.hdf5"
            modelpath_d = filepath + "{epoch:02d}-D.hdf5"
            modelpath_q = filepath + "{epoch:02d}-Q.hdf5"
            LVDR = LiveDrawing(filepath, LR_set_train, HR_set_train, Dis_train, SED_train, ratio, patch_n,
                               epoch_show=epoch_show, useDis=useDis, useSED=useSED)
            # Loading Networks
            Networks = Networkclass(ratio, patch_n)
            vgg, generator, discriminator, combined = Networks.SRGAN()
            plot_model(generator, to_file=f'{filepath}generator.png', show_shapes=True)
            plot_model(discriminator, to_file=f'{filepath}discriminator.png', show_shapes=True)
            plot_model(combined, to_file=f'{filepath}combined.png', show_shapes=True)

            # Training GAN
            for epoch in range(1, epochs + 1):
                print(f"Epoch {epoch}/{epochs}")
                num_batches = int(LR_set_train.shape[0] / batch_size)
                for i in range(num_batches):
                    LR_batch = LR_set_train[i * batch_size:(i + 1) * batch_size]
                    HR_batch = HR_set_train[i * batch_size:(i + 1) * batch_size]
                    gen_img = generator.predict(LR_batch)

                    valid = np.ones((batch_size, 1))
                    not_valid = np.zeros((batch_size, 1))

                    d_loss_real = discriminator.train_on_batch(HR_batch, valid)
                    d_loss_fake = discriminator.train_on_batch(gen_img, not_valid)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                    feature = vgg.predict(np.concatenate((HR_batch, HR_batch, HR_batch), axis=3))
                    g_loss = combined.train_on_batch([LR_batch], [valid, feature])  # discirminator가 valid하고 feature 잘 나오고 BVTV 맞게 학습.
                    if i % 500 == 0: print(f"Batch {i + 1}/{num_batches}\td_loss: {d_loss}\tg_loss: {g_loss}")
                LVDR.on_epoch_end_GAN(epoch, generator)
                generator.save_weights(modelpath_g.format(epoch=epoch), True)
                discriminator.save_weights(modelpath_d.format(epoch=epoch), True)
        if case == 5:
            TIME_START = datetime.now()
            modelname = "SRGAN_BV"
            filepath = f"Models/{modelname}/"
            os.makedirs(filepath, exist_ok=True)
            modelpath = filepath + "{filepath}{epoch:02d}-{val_loss:.4f}.hdf5"
            modelpath_g = filepath + "{epoch:02d}-G.hdf5"
            modelpath_d = filepath + "{epoch:02d}-D.hdf5"
            modelpath_q = filepath + "{epoch:02d}-Q.hdf5"
            LVDR = LiveDrawing(filepath, LR_set_train, HR_set_train, Dis_train, SED_train, ratio, patch_n,
                               epoch_show=epoch_show, useDis=useDis, useSED=useSED)
            # Loading Networks
            Networks = Networkclass(ratio, patch_n)
            vgg, generator, discriminator, combined = Networks.SRGAN_BV()
            plot_model(generator, to_file=f'{filepath}generator.png', show_shapes=True)
            plot_model(discriminator, to_file=f'{filepath}discriminator.png', show_shapes=True)
            plot_model(combined, to_file=f'{filepath}combined.png', show_shapes=True)

            # Training GAN
            for epoch in range(1, epochs + 1):
                print(f"Epoch {epoch}/{epochs}")
                num_batches = int(LR_set_train.shape[0] / batch_size)
                for i in range(num_batches):
                    LR_batch = LR_set_train[i * batch_size:(i + 1) * batch_size]
                    HR_batch = HR_set_train[i * batch_size:(i + 1) * batch_size]
                    gen_img = generator.predict(LR_batch)

                    valid = np.ones((batch_size, 1))
                    not_valid = np.zeros((batch_size, 1))

                    d_loss_real = discriminator.train_on_batch(HR_batch, valid)
                    d_loss_fake = discriminator.train_on_batch(gen_img, not_valid)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                    feature = vgg.predict(np.concatenate((HR_batch, HR_batch, HR_batch), axis=3))
                    g_loss = combined.train_on_batch([LR_batch], [valid, feature, HR_batch])  # discirminator가 valid하고 feature 잘 나오고 BVTV 맞게 학습.
                    if i % 100 == 0:
                        print(f"Batch {i + 1}/{num_batches}\td_loss: {d_loss}\tg_loss: {g_loss}")
                        TIME_END = datetime.now()
                        print([f"{epoch} #Training time:", (TIME_END - TIME_START)])
                LVDR.on_epoch_end_GAN(epoch, generator)
                generator.save_weights(modelpath_g.format(epoch=epoch), True)
                discriminator.save_weights(modelpath_d.format(epoch=epoch), True)
