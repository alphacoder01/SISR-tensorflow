from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow as tf

DIV2K_RGB_MEAN =  np.array([0.4488,0.4371,0.4040]) * 255


def normalize(x, rgb_mean=DIV2K_RGB_MEAN):
  return (x-rgb_mean) / 127.5

def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
  return x*127.5 + rgb_mean

def normalize_01(x):
  return x/255.0

def normalize_11(x):
  return x/127.5 -1

def denormalize_11(x):
  return (x+1)*127.5

def psnr(x1,x2):
  return tf.image.psnr(x1,x2,max_val=255)

def pixle_shuffle(scale):
  return lambda x: tf.nn.depth_to_space(x,scale)


def resolve(model, lr_batch):
  lr_batch = tf.cast(lr_batch, tf.float32)
  sr_batch = model(lr_batch)
  sr_batch = tf.clip_by_value(sr_batch,0,255)
  sr_batch = tf.round(sr_batch)
  sr_batch = tf.cast(sr_batch, tf.uint8)
  return sr_batch


def eval(model, dataset):
  psnr_values = []
  for lr, hr in dataset:
    sr = resolve(model, lr)
    psnr_value = psnr(hr, sr)[0]
    psnr_values.append(psnr_value)
  
  return tf.reduce_mean(psnr_values)


def vgg_22():
  return _vgg(5)

def vgg_54():
  return _vgg(20)

def _vgg(output_layer):
  vgg = VGG19(input_shape=(None,None,3), include_top=False)
  return Model(vgg.input, vgg.layers[output_layer].output)