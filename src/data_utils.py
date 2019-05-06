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

#assuming that the dataset structure is as follows
#dataset_root
#--> stimuli_dir
#--> fixation_maps_dir
#
# stimuli_dir and fixation_maps_dir should be provided as relative paths to dataset_root
# by default all loaded data will be saved into cache
def import_train_data(dataset_root, stimuli_dir, fixation_maps_dir, use_cache=True):

    if use_cache:
        cache_dir = join(dataset_root, '__cache')
        if not isdir(cache_dir):
            mkdir(cache_dir)

        if isfile(join(cache_dir, 'data.npy')):
            fine_image_data, coarse_image_data, label_data = np.load(join(cache_dir, 'data.npy'))
            return fine_image_data, coarse_image_data, label_data

    image_names = [f for f in listdir(join(dataset_root, stimuli_dir)) if isfile(join(dataset_root, stimuli_dir , f))]
    #shuffle(image_names)

    fine_image_data = np.zeros((len(image_names), 600, 800, 3), dtype=np.float32)
    coarse_image_data = np.zeros((len(image_names), 300, 400, 3), dtype=np.float32)
    label_data = np.zeros((len(image_names), 37, 50, 1), dtype=np.float32)
    #label_data = np.zeros((len(image_names), 37*50), dtype=np.float32)

    for i in range(len(image_names)):
        img_fine = img_to_array(load_img(join(dataset_root, stimuli_dir, image_names[i]),
            grayscale=False,
            target_size=(600, 800),
            interpolation='nearest'))
        img_coarse = img_to_array(load_img(join(dataset_root, stimuli_dir, image_names[i]),
            grayscale=False,
            target_size=(300, 400),
            interpolation='nearest'))
        label = img_to_array(load_img(join(dataset_root, fixation_maps_dir, image_names[i]),
            grayscale=True,
            target_size=(37, 50),
            interpolation='nearest'))

        img_fine -= vgg_mean
        img_coarse -= vgg_mean

        fine_image_data[i] = img_fine[None, :]/255
        coarse_image_data[i] = img_coarse[None, :]/255
        label_data[i] = label[None, :]/255
        # label = np.ndarray.flatten(label/255)
        # label_data[i] = np.exp(label)/np.sum(np.exp(label))

    if use_cache:
        np.save(join(cache_dir, 'data.npy'), (fine_image_data, coarse_image_data, label_data))

    return fine_image_data, coarse_image_data, label_data
