import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import load_model
from data_utils import *
from SALICONtf import *
import math
import pickle
import os

def create_model(learn_rate, momentum, decay, loss='crossentropy'):
    model = SALICONtf(activation='sigmoid').model
    loss_fn = keras.losses.binary_crossentropy

    sgd = keras.optimizers.SGD(lr=learn_rate, momentum=momentum, decay=decay, nesterov=True)
    model.compile(loss=loss_fn, optimizer=sgd, metrics=['accuracy', 'mae'])
    return model

def finetune(learn_rate=0.01, loss='crossentropy', num_epochs=500):

    X_train, Y_train, X_val, Y_val = get_osie_data()

    model = create_model(learn_rate=learn_rate, momentum=0.9, decay=0.0005, loss=loss)
    checkpoint = keras.callbacks.ModelCheckpoint('models/SALICONtf'+'_{epoch:02d}.h5', monitor='val_loss', save_best_only=True, period=100, save_weights_only=False)

    history = model.fit(X_train, Y_train,
        batch_size=1,
        epochs=num_epochs,
        validation_data=(X_val, Y_val),
        shuffle=True,
        callbacks=[checkpoint],
        verbose=1)
    
    model.save('models/'+model_name+'.h5')

    return model_name, model



if __name__ == "__main__":
    model_name, model = finetune()
