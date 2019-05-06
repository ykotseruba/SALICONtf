import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import load_model
from data_utils import *
from SALICON import *
import math
import pickle
import os

def create_model(learn_rate, momentum, decay, loss='crossentropy'):
    model = SALICON(activation='sigmoid').model
    loss_fn = keras.losses.binary_crossentropy

    sgd = keras.optimizers.SGD(lr=learn_rate, momentum=momentum, decay=decay, nesterov=True)
    model.compile(loss=loss_fn, optimizer=sgd, metrics=['accuracy', 'mae'])
    return model

def finetune(dataset='osie', learn_rate=0.0001, loss='crossentropy', num_epochs=500):

    X_train, Y_train, X_val, Y_val = get_osie_data()

    model = create_model(learn_rate=learn_rate, momentum=0.9, decay=0.0005, loss=loss)
    checkpoint = keras.callbacks.ModelCheckpoint('models/SALICONtf'+'_{epoch:02d}.h5', monitor='val_loss', save_best_only=True, period=100, save_weights_only=False)
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
