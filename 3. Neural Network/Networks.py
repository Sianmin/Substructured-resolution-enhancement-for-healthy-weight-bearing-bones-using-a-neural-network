from keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Dropout, MaxPooling2D, UpSampling2D, Concatenate, Activation, Lambda
from keras.models import Model
from keras.applications import VGG19
from keras.optimizers import Adam, Nadam
from keras import losses
from keras import backend as K
import math
import tensorflow as tf

class Networks:
    def __init__(self, ratio, patch_n, batch_size, isGAN = False):
        self.ratio, self.patch_n = ratio, patch_n
        self.rp = ratio*patch_n
        self.batch_size = batch_size
        [height, width] = [2080, 1883]
        [self.NY, self.NX] = [math.floor(height / ratio), math.floor(width / ratio)]

    def Loss_MSE_BVTV(self, y_true, y_pred):
        MSE = K.mean(K.square(y_pred - y_true))
        BV_TV = K.mean(K.square(K.sum(y_pred)-K.sum(y_true)))
        c = 10
        loss = MSE + c * BV_TV
        return loss

    def build_vgg19(self):
        vgg = VGG19(include_top=False, input_shape=(self.rp, self.rp, 3))
        vgg.outputs = [vgg.layers[9].output]
        img = Input(shape=((self.rp, self.rp, 3)))
        img_features = vgg(img)
        return Model(img, img_features)

    '''SRGAN-1'''
    def Generator_SRGAN_1(self):
        def residual_block(layer_input, filters):
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu')(layer_input)
            d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Concatenate()([d, layer_input])
            return d
        def deconv2d(layer_input):
            u = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(layer_input)
            return u

        img_lr = Input((self.patch_n, self.patch_n, 1))
        img_lr1 = UpSampling2D((self.ratio, self.ratio), interpolation='bilinear')(img_lr)
        c1 = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(img_lr1)

        r = residual_block(c1, 32)
        for _ in range(12 - 1):
            r = residual_block(r, 32)

        c2 = Conv2D(32, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Concatenate()([c2, c1])

        u1 = deconv2d(c2)
        u2 = deconv2d(u1)

        gen_hr = Conv2D(1, kernel_size=3, strides=1, padding='same', activation='sigmoid')(u2)

        generator = Model(img_lr, gen_hr)
        generator.summary()
        return generator

    def Discriminator_SRGAN_1(self):
        def d_block(layer_input, filters, strides=1, bn=True):
            d = Conv2D(filters, (3, 3), strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        d0 = Input((self.rp, self.rp, 1))
        d1 = d_block(d0, 16, bn=False)
        d2 = d_block(d1, 16)
        d3 = d_block(d2, 16*2)
        d4 = d_block(d3, 16*2, strides=2)
        d5 = d_block(d4, 16*4)
        d6 = d_block(d5, 16*4, strides=2)
        d7 = d_block(d6, 16*8)
        d8 = d_block(d7, 16*8, strides=2)
        d8 = Flatten()(d8)

        d9 = Dense(16*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        discriminator = Model(d0, validity)
        discriminator.summary()
        return discriminator

    def SRGAN_1(self):
        optimizer = Adam(0.0001, 0.5, 0.9)

        # VGG
        vgg = self.build_vgg19()
        vgg.trainable = False
        vgg.compile(loss='mse', optimizer = optimizer, metrics=['accuracy'])
        # discriminator
        discriminator = self.Discriminator_SRGAN_1()
        discriminator.compile(optimizer=optimizer,
                              loss=['binary_crossentropy'],
                              metrics=['accuracy'])
        discriminator.trainable = False
        discriminator.summary()
        # Generator
        generator = self.Generator_SRGAN_1()

        gen_input = Input((self.patch_n, self.patch_n, 1))
        hr_input = Input((self.rp, self.rp, 1))

        fake = generator(gen_input)
        fake_3c = Concatenate()([fake, fake, fake])
        fake_feature = vgg(fake_3c)
        # Combined Model
        validity = discriminator(fake)
        combined = Model([gen_input, hr_input], [validity, fake_feature])
        combined.compile(optimizer=optimizer,
                         loss=[self.Loss_MSE_BVTV, 'mse'],
                               loss_weights=[1e-3, 1])
        combined.summary()
        return vgg, generator, discriminator, combined