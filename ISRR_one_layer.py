#!/usr/bin/env python
from __future__ import print_function
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.models import Sequential
from keras.losses import mean_absolute_error, mean_squared_error
from keras import layers
from sklearn import preprocessing
from keras.activations import exponential, linear
import keras
import pandas as pd
from keras.optimizers import Adam, RMSprop
from keras import regularizers
import numpy as np
from tensorflow.losses import huber_loss
import random
from keras import backend as K
import sys
import tensorflow as tf
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from scipy import signal

WINDOW_SIZE= 100
NUM_ADC= 4
NUM_CLASSES= 2
sess = tf.Session()
K.set_session(sess)
np.set_printoptions(threshold=np.nan)

def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
    plt.legend()
    plt.ylim([0, 5])
    plt.show()

def custom_loss(y_true, y_pred):
    r_hat = y_pred[:, 0]
    r_true = y_true[:, 0]
    cos_th_hat= y_pred[:, 1]
    cos_th_true= y_true[:, 1]
    sin_th_hat= y_pred[:, 2]
    sin_th_true= y_true[:, 2]
    coseno= cos_th_hat*cos_th_true + sin_th_hat*sin_th_true

    return K.abs(r_true*r_true + r_hat*r_hat - 2*r_true*r_hat*coseno)

def custom_activation(x):
    return K.tanh(x)

def model_function(data, labels, test, lab_test):

    model= Sequential()
    model.add(Conv1D(filters= 3, kernel_size= 5))
    model.add(Flatten())
    model.add(Dense(2, activation=None, kernel_regularizer= regularizers.l2(0.02)))

    rms= RMSprop(lr=1e-5)

    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    history= model.fit(data, labels, batch_size=100, nb_epoch=5,  verbose=1, validation_data=(test, lab_test))
    predictions=model.predict(test, batch_size=1)
    print("Confusion Matrix:")
    #y_test.values.argmax(axis=1), predictions.argmax(axis=1))
    print(confusion_matrix(np.argmax(predictions, axis=1), np.argmax(lab_test, axis=1)))
    model.save("ISRR_tire3.hdf5")
    print("predictions-ground_truth:")
    print("predictions shape:", predictions.shape)
    print("labels test shape: ", lab_test.shape)
    print(predictions)
    plt.plot(predictions-lab_test)
    plt.title('Errors in Predictions')
    plt.show()
    ii=0

    for l in model.layers:
        print(str(l.input_shape) + ' ' + str(l.output_shape))
        ii+=1

if __name__== '__main__':
    train=0.85

    data= np.genfromtxt('tire_ISRR/tire_data/tire_data_April30.csv', delimiter=',')
    print(data.shape)
    x= data[:, :-1]
    y= np.zeros((data.shape[0], NUM_CLASSES))

    # One hot encoding
    y[:, 0]= data[:, -1]
    y[:, 1]= 1-data[:, -1]

    # Generating Random Training Samples:
    indexes= random.sample(range(x.shape[0]),int(train*int(x.shape[0])))
    missing= list(set(range(x.shape[0])) - set(indexes))

    data_train=x[indexes, :]
    label_train=y[indexes, :]
    data_test= x[missing,:]
    label_test= y[missing, :]
    print(data_train.shape)
    data_train= np.reshape(data_train, (data_train.shape[0], data_train.shape[1], 1))
    data_test= np.reshape(data_test, (data_test.shape[0], data_test.shape[1], 1))

    model= model_function(data_train, label_train, data_test, label_test)
    sess.close()
