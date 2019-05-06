from os import listdir, mkdir
from os.path import isfile, isdir, join
import numpy as np
import scipy.misc
from scipy.ndimage import gaussian_filter
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from random import shuffle
from sklearn.model_selection import train_test_split
import math

vgg_mean = np.array([123, 116, 103])

def get_mit1003_data():
    mit1003_dir = 'mit1003_dataset/'
    mit1003_stimuli = mit1003_dir+'/ALLSTIMULI'
    mit1003_labels = mit1003_dir+'/ALLFIXATIONMAPS'

    fine_image_data, coarse_image_data, label_data = import_train_data(mit300_dir, 'ALLSTIMULI', 'ALLFIXATIONMAPS', use_cache=False)
    return [fine_image_data, coarse_image_data], label_data

def get_osie_data():
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

#    return [fine_image_data, coarse_image_data], label_data
