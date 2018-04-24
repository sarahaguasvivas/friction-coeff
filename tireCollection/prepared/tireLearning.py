#!/usr/bin/env python2.7
import numpy as np
#import keras
from scipy import signal
import matplotlib.pyplot as plt

data= np.array([])
filename= "FakeGrass.txt"

data= np.loadtxt(filename, delimiter= ',', usecols=range(4))

BUFFER_SIZE= 800
WINDOW_LEFT_SIZE=200
WINDOW_RIGHT_SIZE= 300
count=0
peaks= np.array([])
xpeaks=np.array([])

for i in xrange(BUFFER_SIZE, data.shape[0], BUFFER_SIZE):
    Window= data[i-BUFFER_SIZE:i, :]
    b,a= signal.butter(4, 0.01)
    filtered= signal.filtfilt(b,a, Window[:, 0], padlen=100)
    envelope= signal.hilbert(filtered)
    envelope= np.abs(envelope)
    peak= signal.argrelextrema(envelope, np.greater)
    plt.plot(envelope)
    plt.plot(Window[:, 0])
    plt.show()
    window= Window[peak, 0]
    maxim= np.max(window)
    if maxim > 0.4:
        peaks1= np.where(window==maxim)
        xpeaks= np.append(xpeaks, peaks1)
        peaks=  np.append(peaks, maxim)
        count+=1
print count

