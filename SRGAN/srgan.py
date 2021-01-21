import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.applications.vgg19 import preprocess_input
import time
from utils import normalize_01, normalize_11

LR_SIZE = 24
HR_SIZE = 96


def upsample(x_in, num_filters):
  x = L.Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
  x = L.Lambda(pixle_shuffle(scale=2))(x)
  return L.PReLU(shared_axes=[1,2])(x)


def res_block(x_in, num_filters, momentum=0.8):
    x = L.Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = L.BatchNormalization(momentum=momentum)(x)
    x = L.PReLU(shared_axes=[1, 2])(x)
    x = L.Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = L.BatchNormalization(momentum=momentum)(x)
    x = L.Add()([x_in, x])
    return x


def sr_resnet(num_filters=64, num_res_blocks=16):
  x_in = L.Input(shape=(24, 24, 3))
  x = L.Lambda(normalize_01)(x_in)

  x = L.Conv2D(num_filters, kernel_size = 3, padding= 'same')(x)
  x = x_1 = L.PReLU(shared_axes=[1,2])(x)

  for _ in range(num_res_blocks):
    x = res_block(x, num_filters)

  x = L.Conv2D(num_filters, kernel_size=3, padding='same')(x)
  x = L.BatchNormalization()(x)
  x = L.Add()([x_1,x])

  x = upsample(x, num_filters*4)
  x = upsample(x, num_filters*4)

  x = L.Conv2D(3, kernel_size=9, padding='same',activation='tanh')(x)
  x = L.Lambda(denormalize_11)(x)

  return Model(x_in,x)


def discriminator_block(x_in, num_filters, strides = 1, 
                        batchnorm = True, momentum = 0.8):
  x = L.Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')(x_in)
  if batchnorm:
    x = L.BatchNormalization(momentum=momentum)(x)
  return L.LeakyReLU(alpha=0.2)(x)


def discriminator(num_filters=64):
  x_in = L.Input(shape=(HR_SIZE,HR_SIZE,3))
  x = L.Lambda(normalize_11)(x_in)

  x = discriminator_block(x,num_filters, batchnorm=False)
  x = discriminator_block(x, num_filters, strides=2)

  x = discriminator_block(x,num_filters*4)
  x = discriminator_block(x, num_filters*4, strides=2) 

  x = discriminator_block(x,num_filters*4)
  x = discriminator_block(x, num_filters*4, strides=2)

  x = discriminator_block(x,num_filters*8)
  x = discriminator_block(x, num_filters*8, strides=2)

  x = L.Flatten()(x)
  x = L.Dense(1024)(x)
  x = L.LeakyReLU(alpha=0.2)(x)
  x = L.Dense(1, activation='sigmoid')(x)

  return Model(x_in,  x)


