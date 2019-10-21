from keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Dropout, MaxPooling2D, UpSampling2D, Concatenate, Activation, Lambda, Conv2DTranspose, Reshape
from keras.models import Model
from keras.applications import VGG19
from keras.optimizers import Adam, Nadam
from keras_radam import RAdam
from keras import losses
from keras import backend as K
import math
import tensorflow as tf
from keras.losses import mse, binary_crossentropy

class Networks:
    def __init__(self, ratio, patch_n):
        self.ratio, self.patch_n = ratio, patch_n
        self.rp = ratio*patch_n
        [height, width] = [2080, 1883]
        [self.NY, self.NX] = [math.floor(height / ratio), math.floor(width / ratio)]

    def LOSS_BVTV(self, y_true, y_pred):
        BV_TV = K.square(K.mean(y_pred - y_true))
        return BV_TV

    def build_vgg19(self):
        vgg = VGG19(include_top=False, input_shape=(self.rp, self.rp, 3))
        vgg.outputs = [vgg.layers[9].output]
        img = Input(shape=((self.rp, self.rp, 3)))
        img_features = vgg(img)
        return Model(img, img_features)

    def VariationalAutoEncoder(self):
        filter = 64
        latent_dim = 16
        intermediate_dim = 10
        # 6400 -> 16
        def sampling(args):
            z_mean, z_log_var = args
            batch, dim = K.shape(z_mean)[0], K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1)
            return z_mean + K.exp(0.5 * z_log_var) * epsilon
        input_e = Input((self.rp, self.rp, 1))
        x = Conv2D(filter, kernel_size = 3 , strides=1, padding='same', activation='relu')(input_e)
        x = Conv2D(filter, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(filter*2, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = Conv2D(filter * 2, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(filter*4, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = Conv2D(filter * 4, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(filter*8, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = Flatten()(x)
        z_mean = Dense(latent_dim)(x)
        z_log_var = Dense(latent_dim)(x)
        z = Lambda(sampling)([z_mean, z_log_var])
        encoder =  Model(input_e, [z_mean, z_log_var, z])

        input_d = Input((latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim**2, activation='relu')(input_d)
        x = Reshape((intermediate_dim, intermediate_dim, 1))(x)
        x = Conv2D(filter*8, kernel_size = 3 , strides=1, padding='same', activation='relu')(x)
        x = Conv2D(filter*8, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(filter *4,  kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = Conv2D(filter *4, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(filter * 2, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = Conv2D(filter *2, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(filter, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        output_d = Conv2D(1, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)
        decoder= Model(input_d, output_d)

        output = decoder(encoder(input_e)[2])
        vae = Model(input_e, output)
        # optimizer = RAdam()
        reconstruction_loss = binary_crossentropy(K.flatten(input_e), K.flatten(output))
        reconstruction_loss *= (self.rp**2)
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss =  K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer = 'adam')
        vae.summary()

        return vae, encoder, decoder
    def AutoEncoder(self):
        filter = 8
        def encoderNet():
            input = Input((self.rp, self.rp, 1))
            x = Conv2D(filter, kernel_size = 3 , strides=1, padding='same', activation='relu')(input)
            x = MaxPooling2D((2, 2))(input)
            x = Conv2D(filter*2, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = MaxPooling2D((2, 2))(x)
            x = Conv2D(filter*4, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = MaxPooling2D((2, 2))(x)
            output = Conv2D(filter*8, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            return Model(input, output)
        def decoderNet():
            input = Input((int(self.rp/8), int(self.rp/8), filter*8))
            x = UpSampling2D((2, 2))(input)
            x = Conv2D(filter * 4, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(filter * 2, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(filter, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            output = Conv2D(1, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)
            return Model(input, output)

        encoder = encoderNet()
        decoder = decoderNet()

        input = Input((self.rp, self.rp, 1))
        encodered = encoder(input)
        decodered = decoder(encodered)
        autoencoder = Model(input, decodered)

        optimizer = RAdam()

        autoencoder.compile(loss='mse', optimizer = optimizer, metrics=['accuracy'])
        autoencoder.summary()

        return autoencoder, encoder, decoder

    '''SRGAN-1'''

    def Generator(self):
        filter = 128
        dis_lr = Input((self.patch_n + 1, self.patch_n + 1, 6))
        d1 = Conv2D(filter, kernel_size=2, strides=1, activation='relu')(dis_lr)
        d2 = MaxPooling2D((2,2))(d1)
        d2 = Conv2D(filter*2, kernel_size=3, strides=1, activation='relu', padding='same')(d2)
        d2 = MaxPooling2D((2, 2))(d1)
        d2 = Conv2D(filter*4, kernel_size=3, strides=1, activation='relu', padding='same')(d2)

        img_lr = Input((self.patch_n, self.patch_n, 1))
        l1 = Conv2D(filter, kernel_size=3, strides=1, padding='same', activation='relu')(img_lr)
        l1 = Conv2D(filter, kernel_size=3, strides=1, padding='same', activation='relu')(l1)
        l2 = MaxPooling2D((2,2))(l1)
        l2 = Conv2D(filter*2, kernel_size=3, strides=1, padding='same', activation='relu')(l2)
        l2 = Conv2D(filter * 2, kernel_size=3, strides=1, padding='same', activation='relu')(l2)
        l3 = MaxPooling2D((2, 2))(l2)
        l3 = Conv2D(filter * 4, kernel_size=3, strides=1, padding='same', activation='relu')(l3)
        l3 = Conv2D(filter * 4, kernel_size=3, strides=1, padding='same', activation='relu')(l3)

        x = Concatenate()([d2, l3])
        x = Conv2DTranspose(filter * 4, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = Concatenate()([x, l2])
        x = Conv2D(filter * 2, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = Conv2DTranspose(filter * 4, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = Concatenate()([x, l1])
        x = Conv2D(filter * 2, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        gen_hr = Conv2D(1, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)

        generator = Model([img_lr, dis_lr], gen_hr)
        generator.summary()
        return generator

    def Discriminator(self):
        def d_block(layer_input, filters, strides=1, bn=True):
            d = Conv2D(filters, (3, 3), strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        d0 = Input((self.rp, self.rp, 1))
        d1 = d_block(d0, 8, bn=False)
        d2 = d_block(d1, 8)
        d3 = d_block(d2, 8 * 2)
        d4 = d_block(d3, 8 * 2, strides=2)
        d5 = d_block(d4, 8 * 4)
        d6 = d_block(d5, 8 * 4, strides=2)
        d7 = d_block(d6, 8 * 8)
        d8 = d_block(d7, 8 * 8, strides=2)
        d8 = Flatten()(d8)

        d9 = Dense(4*4)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        discriminator = Model(d0, validity)
        discriminator.summary()
        return discriminator

    def SRGAN(self):
        optimizer = RAdam()

        # AutoEncoder
        # autoencoder, encoder, decoder = self.AutoEncoder()
        # autoencoder.load_weights("Models/AUTOENCODER/01-G.hdf5")
        # autoencoder.outputs = [autoencoder.layers[1].output]
        # encoder.trainable = False

        # discriminator
        discriminator = self.Discriminator()
        discriminator.compile(optimizer=optimizer,
                              loss=['binary_crossentropy'],
                              metrics=['accuracy'])
        discriminator.trainable = False
        discriminator.summary()
        # Generator
        generator = self.Generator()

        gen_input = Input((self.patch_n, self.patch_n, 1))
        dis_input = Input((self.patch_n+1, self.patch_n+1, 6))

        fake = generator(gen_input)
        # Combined Model
        validity = discriminator(fake)
        combined = Model([gen_input, dis_input], [validity, fake, fake])
        combined.compile(optimizer=optimizer,
                         loss=['binary_crossentropy', 'mse', self.LOSS_BVTV],
                         loss_weights=[1e-3, 1, 1])
        combined.summary()
        return generator, discriminator, combined