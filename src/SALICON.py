import sys
from os import listdir, makedirs
from os.path import isfile, join
import os
import numpy as np
import math
import json
import cv2
import scipy.misc
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers, optimizers, regularizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.python.ops import math_ops
import matplotlib.pyplot as plt
from data_utils import import_train_data
import keras.backend as K
from PIL import Image
from scipy.misc import imsave


def get_data():
    osie_dir = 'osie_dataset/data/'
    osie_stimuli = osie_dir + 'stimuli'
    osie_labels = osie_dir + 'fixation_maps'

    fine_image_data, coarse_image_data, label_data = import_train_data(osie_dir, 'stimuli', 'fixation_maps', use_cache=False)

    num_val = int(math.floor(len(fine_image_data)*0.1))
    val_idx = np.random.choice(len(fine_image_data), num_val, replace=False)
    train_idx = np.setdiff1d(range(len(fine_image_data)), val_idx)

    X_train = [fine_image_data[train_idx], coarse_image_data[train_idx]]
    Y_train = label_data[train_idx]

    X_val = [fine_image_data[val_idx], coarse_image_data[val_idx]]
    Y_val = label_data[val_idx]
    return X_train, Y_train, X_val, Y_val


class SALICON():
    def __init__(self, weights=''):
        self.build_salicon_model()
        #load weights if provided
        if weights:
          self.model.load_weights(weights)

    def build_vgg16(self, input_shape, stream_type):
        img_input = layers.Input(shape=input_shape, name='Input_' + stream_type)
        # Block 1
        x = layers.Conv2D(64, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block1_conv1_'+stream_type)(img_input)
        x = layers.Conv2D(64, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block1_conv2_'+stream_type)(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool_'+stream_type)(x)

        # Block 2
        x = layers.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block2_conv1_'+stream_type)(x)
        x = layers.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block2_conv2_'+stream_type)(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool_'+stream_type)(x)

        # Block 3
        x = layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block3_conv1_'+stream_type)(x)
        x = layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block3_conv2_'+stream_type)(x)
        x = layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block3_conv3_'+stream_type)(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool_'+stream_type)(x)

        # Block 4
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block4_conv1_'+stream_type)(x)
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block4_conv2_'+stream_type)(x)
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block4_conv3_'+stream_type)(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool_'+stream_type)(x)

        # Block 5
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block5_conv1_'+stream_type)(x)
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block5_conv2_'+stream_type)(x)
        output = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block5_conv3_'+stream_type)(x)

        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool_'+stream_type)(output)
        x = layers.GlobalMaxPooling2D()(x)

        model = Model(inputs=img_input, outputs=x)

        #initialize each vgg16 stream with ImageNet weights
        try:
          model.load_weights('models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=False)
          model = Model(inputs=img_input, outputs=output)
          #plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
        except OSError:
          print("ERROR: VGG weights are not found.\nRun download_vgg_weights.sh in models/ directory")
          sys.exit(-1)
        return model.input, model.output


    def build_salicon_model(self):
        #create two streams separately
        fine_stream_input, fine_stream_output = self.build_vgg16(input_shape=(600, 800, 3), stream_type='fine')
        coarse_stream_input, coarse_stream_output = self.build_vgg16(input_shape=(300, 400, 3), stream_type='coarse')

        #add interpolation layer to the coarse stream
        H,W = fine_stream_output.shape[1], fine_stream_output.shape[2]
        interp_layer = layers.Lambda(lambda input_tensor: tf.image.resize_nearest_neighbor(input_tensor, (H, W), align_corners=True))(coarse_stream_output)

        #add concatenation layer followed by 1x1 convolution to combine streams
        concat_layer = layers.concatenate([fine_stream_output, interp_layer], axis=-1)

        sal_map_layer = layers.Conv2D(1, (1, 1),
                        name='saliency_map',
                        trainable=True,
                        activation='sigmoid',
                        kernel_initializer=keras.initializers.Zeros(),
                        bias_initializer=keras.initializers.Zeros())(concat_layer)

        self.model = Model(inputs=[fine_stream_input, coarse_stream_input], outputs=sal_map_layer)
        self.model.summary()
        
    def compute_saliency(self, img_path):
        vgg_mean = np.array([123, 116, 103])

        img_fine = img_to_array(load_img(img_path,
            grayscale=False,
            target_size=(600, 800),
            interpolation='nearest'))

        img_coarse = img_to_array(load_img(img_path,
            grayscale=False,
            target_size=(300, 400),
            interpolation='nearest'))

        img_fine -= vgg_mean
        img_coarse -= vgg_mean

        img_fine = img_fine[None, :]/255
        img_coarse = img_coarse[None, :]/255

        smap = np.squeeze(self.model.predict([img_fine, img_coarse], batch_size=1, verbose=0))
        #print(smap.shape)

        img = scipy.misc.imread(img_path)
        smap = (smap - np.min(smap))/((np.max(smap)-np.min(smap)))
        if smap.ndim == 1:
          smap = np.resize(smap, (37,50))
        smap = cv2.resize(smap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)  
        smap = cv2.GaussianBlur(smap, (75, 75), 25, cv2.BORDER_DEFAULT) 

        return smap



if __name__ == "__main__":
    main(sys.argv)
