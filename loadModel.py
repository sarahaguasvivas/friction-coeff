#!/usr/bin/env python
import sys
import h5py
import numpy as np
from keras import backend as K
from keras.models import load_model
import keras.losses
#np.set_printoptions(threshold=np.nan)

def custom_loss(y_true, y_pred):
    r_hat = y_pred[:, 1]
    r_true = y_true[:, 1]
    th_hat= y_pred[:, 0]
    th_true= y_true[:, 0]
    coseno= K.cos(th_hat-th_true)
    return K.abs(r_true**2 + r_hat**2 - 2*r_true*r_hat*coseno)

keras.losses.custom_loss=custom_loss

textFile= open("wr.txt", "w")
layer= int(sys.argv[1])

model= load_model('ISRR_tire1.hdf5')
print(model.summary())
textFile= open("wr.txt", "w")

input_size= model.layers[layer].input_shape
print(input_size)
first_input= model.layers[0].input_shape

weights= model.layers[layer].get_weights()[0] # weights
print(weights.shape)
biases= model.layers[layer].get_weights()[1] # biases

print("layer config:")
print(model.layers[layer].get_config())

strWeights= str(list(weights))
strBiases= str(list(biases))

strWeights= strWeights.replace('[', '{')
strWeights= strWeights.replace(']', '}')
strWeights= strWeights.replace('dtype=float32),', '')
strWeights= strWeights.replace('array(', '')
strWeights= strWeights.replace(', dtype=float32)', '')

textFile.write(strWeights)
strBiases= strBiases.replace('[', '{')
strBiases= strBiases.replace(']', '}')

print(strBiases)


inp= model.input
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp] , [out]) for out in outputs]   # evaluation function

test = np.ones((1, 40, 1))
layer_outs = [func([test]) for func in functors]
#print(layer_outs)

output= model.predict(test)
print(output)
