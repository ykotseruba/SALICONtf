import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import ParameterSampler
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import load_model
from data_utils import *
from SALICON import *
import math
import pickle
import os

def exp_decay(epoch):
   initial_lrate = 0.1
   k = 0.1
   lrate = initial_lrate * exp(-k*t)
   return lrate

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.00001
    drop = 0.1
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def create_model(learn_rate, momentum, decay, loss='crossentropy'):
    if loss == 'kldiv':
        model = SALICON(activation='softmax').model
        loss_fn = keras.losses.kullback_leibler_divergence
        #loss_fn = kldiv
    elif loss == 'crossentropy':
        model = SALICON(activation='sigmoid').model
        loss_fn = keras.losses.binary_crossentropy

    sgd = keras.optimizers.SGD(lr=learn_rate, momentum=momentum, decay=decay, nesterov=True)
    model.compile(loss=loss_fn, optimizer=sgd, metrics=['accuracy', 'mae'])
    return model

def finetune(dataset='osie', learn_rate=0.0001, loss='crossentropy', num_epochs=100):
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []

#    X_train, Y_train = import_data(dataset='osie_dataset', data_type='train')
#    X_val, Y_val = import_data(dataset = 'osie_dataset', data_type = 'val')

    
    #X_train, Y_train, X_val, Y_val = get_osie_data()


    #for psychosal and oddoneout data we add amount that is proportional to the amount we add to MLNet training
    #so that we can compare how different architectures deal with this data
    #MLNet is trained on SALICON with 10000 training samples
    #for SALICON training we add:
    #psychosal: 100 (16% of 630) extra training images
    #            10 (10% of 70) extra validation images
    #oddoneout: 75 (12% of 630) extra training images
    #           7 (8% of 630) extra training images

    if 'osie' in dataset:
        x_train, y_train = import_data(dataset='osie_dataset', data_type='train')
        x_val, y_val = import_data(dataset='osie_dataset', data_type='val')      

        if len(X_train) == 0:
            X_train = x_train
            Y_train = y_train
            X_val = x_val
            Y_val = y_val 
        print('Loaded OSIE')

    if 'psychosal' in dataset:
        x_train, y_train = import_data(dataset='stimuli_BMVC', data_type='train')
        x_val, y_val = import_data(dataset='stimuli_BMVC', data_type='val')

        if dataset == 'psychosal':
            num_train = len(y_train)
            num_val = len(y_val)
        else:
            num_train = 100
            num_val = 10
        x_train = [x_train[0][0:num_train], x_train[1][0:num_train]]
        y_train = y_train[0:num_train]

        x_val = [x_val[0][0:num_val], x_val[1][0:num_val]]
        y_val = y_val[0:num_val]

        if len(X_train) > 0:
            X_train = [np.concatenate([X_train[0], x_train[0]], axis=0), np.concatenate([X_train[1], x_train[1]], axis=0)]
            Y_train = np.concatenate([Y_train, y_train], axis=0)

            X_val = [np.concatenate([X_val[0], x_val[0]], axis=0), np.concatenate([X_val[1], x_val[1]], axis=0)]
            Y_val = np.concatenate([Y_val, y_val], axis=0)

            del x_train, y_train, x_val, y_val
        else:
            X_train = x_train
            Y_train = y_train
            X_val = x_val
            Y_val = y_val 

        print('Loaded psychosal data')
        
    if 'oddoneout' in dataset:
        x_train, y_train = import_data(dataset='stimuli_oddoneout', data_type='train')
        x_val, y_val = import_data(dataset='stimuli_oddoneout', data_type='val')

        if dataset == 'oddoneout':
            num_train = len(y_train)
            num_val = len(y_val)
        else:
            num_train = 75
            num_val = 7

        x_train = [x_train[0][0:num_train], x_train[1][0:num_train]] #match the proportion of oddoneout data in MLNet training set
        y_train = y_train[0:num_train]

        x_val = [x_val[0][0:num_val], x_val[1][0:num_val]]
        y_val = y_val[0:num_val]

        if len(X_train) > 0:
            X_train = [np.concatenate([X_train[0], x_train[0]], axis=0), np.concatenate([X_train[1], x_train[1]], axis=0)]
            Y_train = np.concatenate([Y_train, y_train], axis=0)

            X_val = [np.concatenate([X_val[0], x_val[0]], axis=0), np.concatenate([X_val[1], x_val[1]], axis=0)]
            Y_val = np.concatenate([Y_val, y_val], axis=0)

            del x_train, y_train, x_val, y_val
        else:
            X_train = x_train
            Y_train = y_train
            X_val = x_val
            Y_val = y_val 

        print('Loaded oddoneout data')

    model = create_model(learn_rate=learn_rate, momentum=0.9, decay=0.0005, loss=loss)
    model_name = 'SALICONtf_{}_lr{}'.format(dataset, learn_rate)
    checkpoint = keras.callbacks.ModelCheckpoint('models/'+model_name+'_{epoch:02d}.h5', monitor='val_loss', save_best_only=True, period=100, save_weights_only=False)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=0, mode='auto', restore_best_weights=True)
    lr_scheduler = keras.callbacks.LearningRateScheduler(step_decay, verbose=1)
    

    history = model.fit(X_train, Y_train,
        batch_size=1,
        epochs=num_epochs,
        validation_data=(X_val, Y_val),
        shuffle=True,
        callbacks=[checkpoint],
        verbose=1)
    
    model.save('models/'+model_name+'.h5')

    return model_name, model


def main(dataset):
    model_name, model = finetune(dataset)

if __name__ == "__main__":
    main(sys.argv[1])
