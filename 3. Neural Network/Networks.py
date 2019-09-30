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
        c = 1
        loss = MSE + c * BV_TV
        return loss

    def build_vgg19(self):
        vgg = VGG19(include_top=False, input_shape=(self.rp, self.rp, 3))
        vgg.outputs = [vgg.layers[9].output]
        img = Input(shape=((self.rp, self.rp, 3)))
        img_features = vgg(img)
        return Model(img, img_features)

    def SR_Resnet_20190806_1(self, num_filters = 64, num_res_block = 16):
        x_in = Input(shape=(self.rp, self.rp, 1))

        x = Conv2D(num_filters, kernel_size = 9, padding='same')(x_in)
        x = x_1 = PReLU(shared_axes=[1, 2])(x)

        def res_block(self, x_in, num_filters):
            x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
            x = BatchNormalization()(x)
            x = PReLU(shared_axes=[1, 2])(x)
            x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
            x = BatchNormalization()(x)
            x = Add()([x_in, x])
            return x

        for _ in range(num_res_block):
            x = self.res_block(x, num_filters)

        x = Conv2D(num_filters, kernel_size = 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x_1, x])

        x = Conv2D(64, (5, 5), padding='same', activation='relu')(x)
        x = Conv2D(64, (5, 5), padding='same', activation='relu')(x)
        x = Conv2D(1, kernel_size = 9, padding='same', activation='tanh')(x)

        return Model(x_in, x)

    def SRCNN_20190806_1(self):
        x_in = Input(shape=(self.rp, self.rp, 1))
        x = Conv2D(64, kernel_size = 10, padding='same')(x_in)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, kernel_size = 10, padding='same')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, kernel_size=5, padding='same')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, kernel_size=3, padding='same')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, kernel_size=3, padding='same')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, kernel_size=3, padding='same')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, kernel_size=3, padding='same')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, kernel_size=3, padding='same')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Conv2D(1, kernel_size = 3, padding='same')(x)
        return Model(x_in, x)

    def Jungjin_20190807_1(self, filter = 32):
        x_in = Input(shape=(self.patch_n, self.patch_n, 1))
        x_u = UpSampling2D((self.ratio, self.ratio), interpolation='bilinear')(x_in)
        x1 = Conv2D(filter, kernel_size = 3, padding='same', activation='relu')(x_u)
        x = Dropout(0.2)(x1)
        x = BatchNormalization()(x1)

        x2 = MaxPooling2D((2, 2))(x)
        x2 = Conv2D(filter*2, kernel_size = 3, padding='same', activation='relu')(x2)
        x = Dropout(0.2)(x2)
        x = BatchNormalization()(x)

        x = MaxPooling2D((2,2))(x)
        x = Conv2D(filter*4, kernel_size = 3, padding='same', activation='relu')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)

        x3 = UpSampling2D((2,2))(x)
        x3 = Conv2D(filter*2, kernel_size = 3, padding='same', activation='relu')(x3)

        x = Concatenate()([x3, x2])
        x = BatchNormalization()(x)

        x = UpSampling2D((2,2))(x)
        x = Conv2D(filter, kernel_size = 3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Concatenate()([x, x1])

        x = Conv2D(1, kernel_size = 3, padding='same', activation='sigmoid')(x)
        return Model(x_in, x)

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
                         loss=['binary_crossentropy', 'mse'],
                               loss_weights=[1e-3, 1])
        combined.summary()
        return vgg, generator, discriminator, combined

    def JungJinNet_Dis_20190827(self, filter = 32):
        x_in = Input(shape=(self.patch_n, self.patch_n, 1))
        dis_in = Input(shape=(self.patch_n+1, self.patch_n+1, 6))

        x_in1 = UpSampling2D((self.ratio, self.ratio), interpolation = 'bilinear')(x_in)
        dis_in1 = UpSampling2D((self.ratio, self.ratio), interpolation = 'nearest')(dis_in)
        dis1 = Conv2D(6, kernel_size = self.ratio +1, strides = 1)(dis_in1)

        x_con = Concatenate()([x_in1, dis1])

        x1 = Conv2D(filter, kernel_size=3, padding='same', activation='relu')(x_con)
        x = Dropout(0.2)(x1)
        x = BatchNormalization()(x1)

        x2 = MaxPooling2D((2, 2))(x)
        x2 = Conv2D(filter * 2, kernel_size=3, padding='same', activation='relu')(x2)
        x = Dropout(0.2)(x2)
        x = BatchNormalization()(x)

        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(filter * 4, kernel_size=3, padding='same', activation='relu')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)

        x3 = UpSampling2D((2, 2))(x)
        x3 = Conv2D(filter * 2, kernel_size=3, padding='same', activation='relu')(x3)

        x = Concatenate()([x3, x2])
        x = BatchNormalization()(x)

        x = UpSampling2D((2, 2))(x)
        x = Conv2D(filter, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Concatenate()([x, x1])

        x = Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')(x)
        return Model([x_in, dis_in], x)

    def SangMinNet_20190830(self, Np, Nb):
        [rp] = [self.rp]
        def GenerationNet(): #Simple CNN
            Patch_in = Input((self.patch_n, self.patch_n, 1))
            filter = 32
            U1 = UpSampling2D((self.ratio, self.ratio), interpolation='bilinear')(Patch_in)
            C1 = Conv2D(filter, kernel_size=3, padding='same', activation='relu')(U1)
            D1 = Dropout(0.2)(C1)
            B1 = BatchNormalization()(D1)

            P2 = MaxPooling2D((2, 2))(B1)
            C2 = Conv2D(filter * 2, kernel_size=3, padding='same', activation='relu')(P2)
            D2 = Dropout(0.2)(C2)
            B2 = BatchNormalization()(D2)

            P3 = MaxPooling2D((2, 2))(B2)
            C3 = Conv2D(filter * 4, kernel_size=3, padding='same', activation='relu')(P3)
            D3 = Dropout(0.2)(C3)
            B3 = BatchNormalization()(D3)

            U4 = UpSampling2D((2, 2))(B3)
            C4 = Conv2D(filter * 2, kernel_size=3, padding='same', activation='relu')(U4)

            x = Concatenate()([C2, C4])
            B5 = BatchNormalization()(x)

            U6 = UpSampling2D((2, 2))(B5)
            C6 = Conv2D(filter, kernel_size=3, padding='same', activation='relu')(U6)
            B6 = BatchNormalization()(C6)
            x = Concatenate()([C1, B6])

            Patch_out = Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')(x)

            GenerationNetwork = Model(Patch_in, Patch_out)

            return GenerationNetwork
        def QuiltNet(): # Encoder-Decoder
            Patch_in = Input((Np, Np, 1))
            C1 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(Patch_in)
            B1 = BatchNormalization()(C1)

            M2 = MaxPooling2D((2, 2))(B1)
            C2 = Conv2D(64, kernel_size=3, padding='same', activation='relu')(M2)
            B2 = BatchNormalization()(C2)

            M3 = MaxPooling2D((2, 2))(B2)
            C3 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(M3)
            B3 = BatchNormalization()(C3)

            M4 = MaxPooling2D((2, 2))(B3)
            C4 = Conv2D(256, kernel_size=3, padding='same', activation='relu')(M4)
            B4 = BatchNormalization()(C4)


            U6 = UpSampling2D((2, 2))(B4)
            C6 = Conv2D(256, kernel_size=3, padding='same', activation='relu')(U6)
            C6 = Concatenate()([C6, C3])
            B6 = BatchNormalization()(C6)

            U7 = UpSampling2D((2, 2))(B6)
            C7 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(U7)
            C7 = Concatenate()([C7, C2])
            B7 = BatchNormalization()(C7)

            U9 = UpSampling2D((2, 2))(B7)
            C9 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(U9)
            C9 = Concatenate()([C9, Patch_in])
            B9 = BatchNormalization()(C9)

            Patch_out = Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')(B9)
            QuiltNetwork = Model(Patch_in, Patch_out)
            return QuiltNetwork
        def QNloss(true, pred):
            TotalMSE = losses.mean_squared_error(true, pred)
            # 0: math.ceil((Np-Nb)/2)    math.floor((Np+Nb)/2):Np
            batch_size = K.shape(true)[0]
            print(batch_size)
            one_tensor = K.zeros((batch_size, rp, Nb, 1))

            true_ol = true[:, :, 0:math.ceil((Np - Nb) / 2),:]
            true_or = true[:, :, math.floor((Np + Nb) / 2):Np, :]
            true_o = K.concatenate([true_ol, one_tensor, true_or], axis = 2)

            pred_ol = pred[:, :, 0:math.ceil((Np - Nb) / 2), :]
            pred_or = pred[:, :, math.floor((Np + Nb) / 2):Np, :]
            pred_o = K.concatenate([pred_ol, one_tensor, pred_or], axis = 2)

            OverlapMSE = losses.mean_squared_error(true_o, pred_o)
            return TotalMSE + 10*OverlapMSE
        GN = GenerationNet()
        QN = QuiltNet()

        optimizer = Adam()
        GN.compile(optimizer = optimizer, loss='mse')
        QN.compile(optimizer = optimizer, loss=QNloss)
        GN.summary()
        QN.summary()
        return GN, QN

    def AdjNet_20190917(self, filter = 64):
        x_in = Input(shape=(3*self.patch_n, 3*self.patch_n, 1))
        dis_in = Input(shape=(3*self.patch_n+1, 3*self.patch_n+1, 6))

        x_1 = UpSampling2D((self.ratio, self.ratio), interpolation='bilinear')(x_in)
        dis1 = Conv2D(1, kernel_size=3, padding='same', activation='relu')(dis_in)
        dis2 = UpSampling2D((self.ratio, self.ratio), interpolation='bilinear')(dis1)
        dis3 = Conv2D(6, kernel_size = self.ratio+1, strides = 1, activation='relu')(dis2)
        x_con = Concatenate()([x_1, dis3])

        x = Conv2D(filter, kernel_size=3, padding='same', activation='relu')(x_con)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D((3, 3))(x)
        x1 = Conv2D(filter * 2, kernel_size=3, padding='same', activation='relu')(x)
        x = Dropout(0.2)(x1)
        x = BatchNormalization()(x)

        x = MaxPooling2D((2, 2))(x)
        x2 = Conv2D(filter * 4, kernel_size=3, padding='same', activation='relu')(x)
        x = Dropout(0.2)(x2)
        x = BatchNormalization()(x)

        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(filter * 8, kernel_size=3, padding='same', activation='relu')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)

        x3 = UpSampling2D((2, 2))(x)
        x3 = Conv2D(filter * 4, kernel_size=3, padding='same', activation='relu')(x3)

        x = Concatenate()([x3, x2])
        x = BatchNormalization()(x)

        x = UpSampling2D((2, 2))(x)
        x = Conv2D(filter*2, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Concatenate()([x, x1])

        x = Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')(x)

        optimizer = Adam()
        GN = Model([x_in, dis_in], x)
        GN.compile(optimizer = optimizer, loss='binary_crossentropy')
        GN.summary()
        return GN

    def ESRGAN(self):
        def SubpixelConv2D(name, scale=2):
            def subpixel_shape(input_shape):
                dims = [input_shape[0],
                        None if input_shape[1] is None else input_shape[1] * scale,
                        None if input_shape[2] is None else input_shape[2] * scale,
                        int(input_shape[3] / (scale ** 2))]
                output_shape = tuple(dims)
                return output_shape

            def subpixel(x):
                return tf.depth_to_space(x, scale)

            return Lambda(subpixel, output_shape=subpixel_shape, name=name)
        def build_generator():
            def dense_block(input):
                x1 = Conv2D(64, kernel_size=3, strides=1, padding='same')(input)
                x1 = LeakyReLU(0.2)(x1)
                x1 = Concatenate()([input, x1])

                x2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x1)
                x2 = LeakyReLU(0.2)(x2)
                x2 = Concatenate()([input, x1, x2])

                x3 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x2)
                x3 = LeakyReLU(0.2)(x3)
                x3 = Concatenate()([input, x1, x2, x3])

                x4 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x3)
                x4 = LeakyReLU(0.2)(x4)
                x4 = Concatenate()([input, x1, x2, x3, x4])

                x5 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x4)
                x5 = Lambda(lambda x: x * 0.2)(x5)
                x = Add()([x5, input])
                return x

            def RRDB(input):
                x = dense_block(input)
                x = dense_block(x)
                x = dense_block(x)
                x = Lambda(lambda x: x * 0.2)(x)
                out = Add()[x, input]
                return out

            def upsampling(x, number):
                x = Conv2D(256, kernel_size=3, strides=1, padding='same', name='upSampleConv2d_' + str(number))(x)
                x = self.SubpixelConv2D('upSampleSubPixel_' + str(number), 2)(x)
                x = PReLU(shared_axes=[1, 2], name='upSamplePReLU_' + str(number))(x)

            LR_input = Input(shape = (self.patch_n, self.patch_n, 1))

            # Pre-residual
            x_start = Conv2D(64, kernel_size = 3, strides=1, padding='same')(LR_input)
            x_start = LeakyReLU(0.2)(x_start)

            x = RRDB(x_start)

            x = Conv2D(64, kernel_size = 3, strides=1, padding='same')(x)
            x = Lambda(lambda x: x*0.2)(x)
            x = Add()([x, x_start])

            x = upsampling(x, self.ratio)

            x = Conv2D(64, kernel_size = 3, strides=1, padding='same')(x)
            x = LeakyReLU(0.2)(x)
            hr_output = Conv2D(1, kernel_size=3, strides=1, padding='same', activation='tanh')(x)

            model = Model(inputs=LR_input, outputs=hr_output)
            return model
        def build_discriminator(filters=64):
            def conv2d_block(input, filters, strides=1, bn=True):
                d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(input)
                d = LeakyReLU(alpha=0.2)(d)
                if bn:
                    d = BatchNormalization(momentum=0.8)(d)
                return d
            # Input high resolution image
            img = Input(shape=(self.rp, self.rp, 1))
            x = conv2d_block(img, filters, bn=False)
            x = conv2d_block(x, filters, strides=2)
            x = conv2d_block(x, filters * 2)
            x = conv2d_block(x, filters * 2, strides=2)
            x = conv2d_block(x, filters * 4)
            x = conv2d_block(x, filters * 4, strides=2)
            x = conv2d_block(x, filters * 8)
            x = conv2d_block(x, filters * 8, strides=2)
            x = Dense(filters * 16)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.4)(x)
            x = Dense(1)(x)
            model = Model(inputs=img, outputs=x)
            return model
        def build_esrgan():
            def comput_loss(x):
                img_hr, generated_hr = x
                # Compute the Perceptual loss
                gen_feature = self.vgg(self.preprocess_vgg(generated_hr))
                ori_feature = self.vgg(self.preprocess_vgg(img_hr))
                percept_loss = tf.losses.mean_squared_error(gen_feature, ori_feature)
                # Compute the RaGAN loss
                fake_logit, real_logit = self.RaGAN([img_hr, generated_hr])
                gen_loss = K.mean(
                    K.binary_crossentropy(K.zeros_like(real_logit), real_logit) +
                    K.binary_crossentropy(K.ones_like(fake_logit), fake_logit))
                # Compute the pixel_loss with L1 loss
                # pixel_loss = tf.losses.absolute_difference(generated_hr, img_hr)
                return [percept_loss, gen_loss]

        def AdjNet_20190917(self, filter=64):
            x_in = Input(shape=(3 * self.patch_n, 3 * self.patch_n, 1))
            dis_in = Input(shape=(3 * self.patch_n + 1, 3 * self.patch_n + 1, 6))

            x_1 = UpSampling2D((self.ratio, self.ratio), interpolation='bilinear')(x_in)
            dis1 = Conv2D(1, kernel_size=3, padding='same', activation='relu')(dis_in)
            dis2 = UpSampling2D((self.ratio, self.ratio), interpolation='bilinear')(dis1)
            dis3 = Conv2D(6, kernel_size=self.ratio + 1, strides=1, activation='relu')(dis2)
            x_con = Concatenate()([x_1, dis3])

            x = Conv2D(filter, kernel_size=3, padding='same', activation='relu')(x_con)
            x = Dropout(0.2)(x)
            x = BatchNormalization()(x)

            x = MaxPooling2D((3, 3))(x)
            x1 = Conv2D(filter * 2, kernel_size=3, padding='same', activation='relu')(x)
            x = Dropout(0.2)(x1)
            x = BatchNormalization()(x)

            x = MaxPooling2D((2, 2))(x)
            x2 = Conv2D(filter * 4, kernel_size=3, padding='same', activation='relu')(x)
            x = Dropout(0.2)(x2)
            x = BatchNormalization()(x)

            x = MaxPooling2D((2, 2))(x)
            x = Conv2D(filter * 8, kernel_size=3, padding='same', activation='relu')(x)
            x = Dropout(0.2)(x)
            x = BatchNormalization()(x)

            x3 = UpSampling2D((2, 2))(x)
            x3 = Conv2D(filter * 4, kernel_size=3, padding='same', activation='relu')(x3)

            x = Concatenate()([x3, x2])
            x = BatchNormalization()(x)

            x = UpSampling2D((2, 2))(x)
            x = Conv2D(filter * 2, kernel_size=3, padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = Concatenate()([x, x1])

            x = Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')(x)

            optimizer = Adam()
            GN = Model([x_in, dis_in], x)
            GN.compile(optimizer=optimizer, loss='binary_crossentropy')
            GN.summary()
            return GN

    def AdjNet_20190926(self, filter = 64):
        x_in = Input(shape=(3*self.patch_n, 3*self.patch_n, 1))
        SED_in = Input(shape=(3*self.patch_n, 3*self.patch_n, 3))
        x_con = Concatenate()([x_in, SED_in])

        x = Conv2D(filter*2, kernel_size=3, padding='same', activation='relu')(x_con)
        x = Conv2D(filter*2, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(filter*2, kernel_size=3, padding='same', activation='relu')(x)

        x = MaxPooling2D((3, 3))(x)
        x0 = Conv2D(filter, kernel_size=3, padding='same', activation='relu')(x)
        x = Dropout(0.2)(x0)
        x = BatchNormalization()(x)

        x = MaxPooling2D((2, 2))(x)
        x1 = Conv2D(filter * 2, kernel_size=3, padding='same', activation='relu')(x)
        x = Dropout(0.2)(x1)
        x = BatchNormalization()(x)

        x = MaxPooling2D((2, 2))(x)
        x2 = Conv2D(filter * 4, kernel_size=3, padding='same', activation='relu')(x)
        x = Dropout(0.2)(x2)
        x = BatchNormalization()(x)

        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(filter * 16, kernel_size=3, padding='same', activation='relu')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)

        x3 = UpSampling2D((2, 2))(x)
        x3 = Conv2D(filter * 4, kernel_size=3, padding='same', activation='relu')(x3)

        x = Concatenate()([x3, x2])
        x = BatchNormalization()(x)

        x = UpSampling2D((2, 2))(x)
        x = Conv2D(filter*2, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Concatenate()([x, x1])

        x = UpSampling2D((2, 2))(x)
        x = Conv2D(filter, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Concatenate()([x, x0])

        x = UpSampling2D((self.ratio, self.ratio))(x)
        x = Conv2D(filter*4, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(filter*4, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(filter * 2, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(filter, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')(x)

        optimizer = Adam()
        GN = Model([x_in, SED_in], x)
        GN.compile(optimizer = optimizer, loss='binary_crossentropy')
        GN.summary()
        return GN
