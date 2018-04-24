#!/usr/bin/env python2.7
import numpy as np
#import keras
from scipy import signal 
import matplotlib.pyplot as plt

data= np.array([])
filename= "FakeGrass.txt"

data= np.loadtxt(filename, delimiter= ',', usecols=range(4))

BUFFER_SIZE= 1000
WINDOW_LEFT_SIZE=200
WINDOW_RIGHT_SIZE= 300 
 
for i in xrange(BUFFER_SIZE, data.shape[0], BUFFER_SIZE):
    Window= data[i-BUFFER_SIZE:i, :]
    envelope= signal.hilbert(Window[:, 0])
    envelope= envelope.real
    plt.plot(envelope)
    b,a= signal.butter(8, 100, 'low', analog=True)
    envelope= signal.filtfilt(b,a, envelope, padlen=50)
    plt.plot(envelope)
    plt.plot(Window[:, 0])
    plt.show()

