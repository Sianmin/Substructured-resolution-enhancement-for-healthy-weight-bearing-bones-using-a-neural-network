from keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Dropout, MaxPooling2D, UpSampling2D, Concatenate, Activation, Lambda
from keras.models import Model
from keras.applications import VGG19
from keras.optimizers import Adam, Nadam
from keras_radam import RAdam
from keras import losses
from keras import backend as K
import math
import tensorflow as tf

class Networks:
    def __init__(self, ratio, patch_n, batch_size):
        self.ratio, self.patch_n = ratio, patch_n
        self.rp = ratio*patch_n
        self.batch_size = batch_size
        [height, width] = [2080, 1883]
        [self.NY, self.NX] = [math.floor(height / ratio), math.floor(width / ratio)]

    def Loss_MSE_BVTV(self, y_true, y_pred):
        MSE = K.mean(K.square(y_pred - y_true))
        BV_TV = K.mean(K.square(K.sum(y_pred)-K.sum(y_true)))
        c = 2
        loss = MSE + c * BV_TV
        return loss

    def LOSS_BVTV(self, y_true, y_pred):
        BV_TV = K.square(K.mean(y_pred) - K.mean(y_true))
        return BV_TV

    def SubpixelConv2D(self, name, scale = self.ratio):
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

    def AutoEncoder(self):
        def encoderNet():
            input = Input((self.rp, self.rp, 1))
            x = Conv2D(32, kernel_size = 3 , strides=1, padding='same', activation='relu')(input)
            x = MaxPooling2D((2, 2))(input)
            x = Conv2D(32*2, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = MaxPooling2D((2, 2))(x)
            x = Conv2D(32*4, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = MaxPooling2D((2, 2))(x)
            x = Conv2D(32*8, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = MaxPooling2D((2, 2))(x)
            output = Conv2D(32*16, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            return Model(input, output)
        def decoderNet():
            input = Input((int(self.rp/16), int(self.rp/16), 32*16))
            x = UpSampling2D((2, 2))(input)
            x = Conv2D(32 * 8, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(32 * 4, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(32 * 2, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
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

    '''ESRGAN'''

    def Generator_ESRGAN(self):
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
            x4 = Concatenate()([input, x1, x2, x3, x4])  # 这里跟论文原图有冲突，论文没x3???

            x5 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x4)
            x5 = Lambda(lambda x: x * 0.2)(x5)
            x = Add()([x5, input])
            return x

        def RRDB(input):
            x = dense_block(input)
            x = dense_block(x)
            x = dense_block(x)
            x = Lambda(lambda x: x * 0.2)(x)
            out = Add()([x, input])
            return out

        def upsample(x, number):
            x = Conv2D(256, kernel_size=3, strides=1, padding='same', name='upSampleConv2d_' + str(number))(x)
            x = self.SubpixelConv2D('upSampleSubPixel_' + str(number), self.ratio)(x)
            x = PReLU(shared_axes=[1, 2], name='upSamplePReLU_' + str(number))(x)
            return x

        img_lr = Input((self.patch_n, self.patch_n, 1))

        # Pre-residual
        x_start = Conv2D(64, kernel_size=3, strides=1, padding='same')(img_lr)
        x_start = LeakyReLU(0.2)(x_start)

        # Residual-in-Residual Dense Block
        x = RRDB(x_start)

        # Post-residual block
        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        x = Lambda(lambda x: x * 0.2)(x)
        x = Add()([x, x_start])
        # Upsampling depending on factor
        x = upsample(x, 1)
        x = upsample(x, self.ratio)

        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        x = LeakyReLU(0.2)(x)
        hr_output = Conv2D(1, kernel_size=3, strides=1, padding='same', activation='tanh')(x)

        generator = Model(img_lr, hr_output)
        generator.summary()
        return generator

    def Discriminator_ESRGAN(self):
        def conv2d_block(input, filters, strides=1, bn=True):
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        d0 = Input((self.rp, self.rp, 1))
        d1 = conv2d_block(d0, 64, bn=False)
        d2 = conv2d_block(d1, 64, strides=2)
        d3 = conv2d_block(d2, 64*2)
        d4 = conv2d_block(d3, 64*2, strides=2)
        d5 = conv2d_block(d4, 64*4)
        d6 = conv2d_block(d5, 64*4, strides=2)
        d7 = conv2d_block(d6, 64*8)
        d8 = conv2d_block(d7, 64*8, strides=2)
        x = Dense(64 * 16)(d8)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.4)(x)
        x = Dense(1)(x)

        discriminator = Model(d0, x)
        discriminator.summary()
        return discriminator

    def RaGAN_ESRGAN(self):
        def interpolating(x):
            u = K.random_uniform((K.shape(x[0])[0],) + (1,) * (K.ndim(x[0]) - 1))
            return x[0] * u + x[1] * (1 - u)

        def comput_loss(x):
            real, fake = x
            fake_logit = (fake - K.mean(real))
            real_logit = (real - K.mean(fake))
            return [fake_logit, real_logit]

        # Input LR images
        imgs_hr = Input((self.rp, self.rp, 1))
        generated_hr = Input((self.rp, self.rp, 1))

        # Create a high resolution image from the low resolution one
        real_discriminator_logits = self.discriminator(imgs_hr)
        fake_discriminator_logits = self.discriminator(generated_hr)

        # x_inter = Lambda(interpolating)([imgs_hr, generated_hr])
        # x_inter_score = self.discriminator(x_inter)

        total_loss = Lambda(comput_loss, name='comput_loss')([real_discriminator_logits, fake_discriminator_logits])
        # print(len(total_loss),total_loss)
        # Output tensors to a Model must be the output of a Keras `Layer`
        fake_logit = Lambda(lambda x: x, name='fake_logit')(total_loss[0])
        real_logit = Lambda(lambda x: x, name='real_logit')(total_loss[1])

        # grads = K.gradients(x_inter_score, [x_inter])[0]
        # print(x_inter)
        # print(x_inter_score)
        # print(grads)
        # grad_norms = K.sqrt(K.sum(grads ** 2, list(range(1, K.ndim(grads)))) + 1e-9)
        dis_loss = K.mean(K.binary_crossentropy(K.zeros_like(fake_logit), fake_logit) +
                          K.binary_crossentropy(K.ones_like(real_logit), real_logit))
        # dis_loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logit), logits=fake_logit) +
        #     tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_likes(real_logit), logits=real_logit))
        # dis_loss = K.mean(- (real_logit - fake_logit)) + 10 * K.mean((grad_norms - 1) ** 2)

        model = Model(inputs=[imgs_hr, generated_hr], outputs=[fake_logit, real_logit])

        model.add_loss(dis_loss)
        model.compile(optimizer=Adam(self.dis_lr))

        model.metrics_names.append('dis_loss')
        model.metrics_tensors.append(dis_loss)
        return model


    def ESRGAN(self):
        autoencoder, encoder, decoder = self.AutoEncoder()
        autoencoder.load_weights("Models/AUTOENCODER/10-G.hdf5")

        def comput_loss(x):
            img_hr, generated_hr = x

            # Compute the Perceptual loss
            gen_feature = encoder(generated_hr)
            ori_feature = encoder(img_hr)
            percept_loss = tf.losses.mean_squared_error(gen_feature, ori_feature)

            # Compute the RaGAN loss
            fake_logit, real_logit = self.RaGAN([img_hr, generated_hr])
            gen_loss = K.mean(
                K.binary_crossentropy(K.zeros_like(real_logit), real_logit) +
                K.binary_crossentropy(K.ones_like(fake_logit), fake_logit))

            # Compute the pixel_loss with L1 loss
            # pixel_loss = tf.losses.absolute_difference(generated_hr, img_hr)
            return [percept_loss, gen_loss]

        optimizer = RAdam()
        # AutoEncoder

        # autoencoder.outputs = [autoencoder.layers[1].output]
        encoder.trainable = False

        # discriminator
        discriminator = self.Discriminator_SRGAN_1()
        discriminator.compile(optimizer=optimizer, loss=['binary_crossentropy'],)
        discriminator.trainable = False
        discriminator.summary()
        # Generator
        generator = self.Generator_SRGAN_1()

        gen_input = Input((self.patch_n, self.patch_n, 1))
        fake = generator(gen_input)

        fake_feature = encoder(fake)
        # Combined Model
        validity = discriminator(fake)
        combined = Model(gen_input, [validity, fake_feature, fake])
        combined.compile(optimizer=optimizer,
                         loss=['binary_crossentropy', 'mse', self.LOSS_BVTV],
                               loss_weights=[1e-3, 1, 1])
        combined.summary()
        return encoder, generator, discriminator, combined