import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D,Add,MaxPooling1D, Concatenate, UpSampling1D, Add, Lambda, LeakyReLU


def unet_block(inputs,kshape= 3):
    conv1 = Conv1D(48, kshape, activation='relu', padding='same')(inputs)
    conv1 = Conv1D(48, kshape, activation='relu', padding='same')(conv1)
    conv1 = Conv1D(48, kshape, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(64, kshape, activation='relu', padding='same')(pool1)
    conv2 = Conv1D(64, kshape, activation='relu', padding='same')(conv2)
    conv2 = Conv1D(64, kshape, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(128, kshape, activation='relu', padding='same')(pool2)
    conv3 = Conv1D(128, kshape, activation='relu', padding='same')(conv3)
    conv3 = Conv1D(128, kshape, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = Conv1D(256, kshape, activation='relu', padding='same')(pool3)
    conv4 = Conv1D(256, kshape, activation='relu', padding='same')(conv4)
    conv4 = Conv1D(256, kshape, activation='relu', padding='same')(conv4)

    up1 = Concatenate([UpSampling1D(size=2)(conv4), conv3], axis=-1)
    conv5 = Conv1D(128, kshape, activation='relu', padding='same')(up1)
    conv5 = Conv1D(128, kshape, activation='relu', padding='same')(conv5)
    conv5 = Conv1D(128, kshape, activation='relu', padding='same')(conv5)

    up2 = Concatenate([UpSampling1D(size=2)(conv5), conv2], axis=-1)
    conv6 = Conv1D(64, kshape, activation='relu', padding='same')(up2)
    conv6 = Conv1D(64, kshape, activation='relu', padding='same')(conv6)
    conv6 = Conv1D(64, kshape, activation='relu', padding='same')(conv6)

    up3 = Concatenate([UpSampling1D(size=2)(conv6), conv1], axis=-1)
    conv7 = Conv1D(48, kshape, activation='relu', padding='same')(up3)
    conv7 = Conv1D(48, kshape, activation='relu', padding='same')(conv7)
    conv7 = Conv1D(48, kshape, activation='relu', padding='same')(conv7)

    out = Conv1D(2, 1, activation='linear')(conv7)
    return out


def cnn_block(inputs,kshape= 3):
    conv1 = Conv1D(64, kshape, padding='same')(inputs)
    conv1 = LeakyReLU(alpha=0.0)(conv1)
    conv1 = Conv1D(64, kshape,  padding='same')(conv1)
    conv1 = LeakyReLU(alpha=0.0)(conv1)
    conv1 = Conv1D(64, kshape, padding='same')(conv1)
    conv1 = LeakyReLU(alpha=0.0)(conv1)
    conv1 = Conv1D(64, kshape,  padding='same')(conv1)
    conv1 = LeakyReLU(alpha=0.0)(conv1)
    conv1 = Conv1D(64, kshape, padding='same')(conv1)
    conv1 = LeakyReLU(alpha=0.0)(conv1)
    conv1 = Conv1D(64, kshape,  padding='same')(conv1)
    conv1 = LeakyReLU(alpha=0.0)(conv1)
    conv1 = Conv1D(64, kshape, padding='same')(conv1)
    conv1 = LeakyReLU(alpha=0.0)(conv1)
    out = Conv1D(1, 1, activation='linear')(conv1)
    return out


def cnn_denoiser(N = 256,kshape= 3,channels = 20):
    fids = Input(shape=(N,channels))
    noisy_spectrum = Input(shape=(N,1))
    cnn_out = cnn_block(fids,kshape= kshape)
    res = Add()([noisy_spectrum,cnn_out])
    model = Model(inputs=[fids,noisy_spectrum], outputs=[res])
    return model