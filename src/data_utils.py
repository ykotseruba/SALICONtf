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

def get_mit300_data():
    mit300_dir = 'mit300_dataset/'
    mit300_stimuli = mit300_dir+'/ALLSTIMULI'
    mit300_labels = mit300_dir+'/ALLFIXATIONMAPS'

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


def import_data(dataset='osie_dataset', data_type='train'):
    fine_image_data, coarse_image_data, label_data = get_train_data(dataset, 
                                                            dataset+'_'+data_type+'.txt', 
                                                            dataset+'_label_'+data_type+'.txt')

    return [fine_image_data, coarse_image_data], label_data


def import_test_data(stimuli_dir):
    image_names = [f for f in listdir(stimuli_dir) if isfile(join(stimuli_dir, f))]
    fine_image_data = np.zeros((len(image_names), 600, 800, 3), dtype=np.float32)
    coarse_image_data = np.zeros((len(image_names), 300, 400, 3), dtype=np.float32)
    for i in range(len(image_names)):
        img_fine = img_to_array(load_img(join(stimuli_dir, image_names[i]),
            grayscale=False,
            target_size=(600, 800),
            interpolation='nearest'))
        img_coarse = img_to_array(load_img(join(stimuli_dir, image_names[i]),
            grayscale=False,
            target_size=(300, 400),
            interpolation='nearest'))

        img_fine -= vgg_mean
        img_coarse -= vgg_mean

        fine_image_data[i] = img_fine[None, :]/255
        coarse_image_data[i] = img_coarse[None, :]/255

    return fine_image_data, coarse_image_data

#load data with labels
def get_train_data(image_dir, image_list, label_list):
    fine_image_data = load_and_preprocess_images(image_dir, image_list, (600, 800, 3))
    coarse_image_data = load_and_preprocess_images(image_dir, image_list, (300, 400, 3))
    label_data = load_and_preprocess_images(image_dir, label_list, (37, 50, 1), subtract_mean=False)
    return fine_image_data, coarse_image_data, label_data

#load data without labels
def get_test_data(image_dir, image_list):
    fine_image_data = load_and_preprocess_images(image_dir, image_list, (600, 800, 3))
    coarse_image_data = load_and_preprocess_images(image_dir, image_list, (300, 400, 3))
    return fine_image_data, coarse_image_data

def load_and_preprocess_images(image_dir, image_list, target_size, subtract_mean=True):

    with open(join(image_dir, image_list), 'r') as fp:
        image_list = [line.strip() for line in fp.readlines()]    
    
    image_data = np.zeros((len(image_list), target_size[0], target_size[1], target_size[2]), dtype=np.float32)

    for i in range(len(image_list)):
        img = img_to_array(load_img(image_dir + '/' + image_list[i],
            grayscale=(not subtract_mean), #only subtract mean from color images
            target_size=target_size[:2],
            interpolation='nearest'))

        if subtract_mean:
            img -= vgg_mean

        #else:
            #img = gaussian_filter(img, sigma=1)
        image_data[i] = img/255

    return image_data

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


def evaluate_model(model, img_dir, gt_dir, save_dir = '', save_smap = False):
    if save_dir and not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    images = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
    gt_images = [f for f in listdir(gt_dir) if isfile(join(gt_dir, f))]

    auc_judd = []
    kldiv = []
    nss = []
    for img_name, gt_image_name in zip(images, gt_images):
        smap = s.compute_saliency(join(img_dir, img_name))
        if save_smap:
            imsave(join(save_dir, img_name), smap)
        #img = imread(join(img_dir, img_name))
        gt_img = imread(join(gt_dir, gt_img_name))

        auc_judd.append(AUC_Judd(smap, gt_img))
        kldiv.append(KLdiv(smap, gt_img))
        nss.append(NSS(smap, gt_img))
    return sum(auc_judd)/len(auc_judd), sum(kldiv)/len(kldiv), sum(nss)/len(kldiv)


def compute_saliency(model, img_path):
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

    smap = np.squeeze(self.model.predict([img_fine, img_coarse], batch_size=1, verbose=1))

    img = scipy.misc.imread(img_path)

    smap = (smap - np.min(smap))/((np.max(smap)-np.min(smap)))
    smap = cv2.resize(smap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    return smap

