#!/usr/bin/env python2.7
import numpy as np
#import keras
from scipy import signal
import matplotlib.pyplot as plt

data= np.array([])
filename= "FakeGrass.txt"
dataFile= "data1FakeGrass.txt"
data= np.loadtxt(filename, delimiter= ',', usecols=range(4))

BUFFER_SIZE= 500 #500 worked
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
    window= Window[peak, 0]
    maxim= np.max(window)
    print window
    if maxim > 0.3:
        peaks1= np.where(envelope==maxim) 
        print peaks1
        xpeaks= np.append(xpeaks, peaks1)
        peaks=  np.append(peaks, maxim)
        j= i-BUFFER_SIZE+peaks1
        np.savetxt(dataFile, data[j-WINDOW_LEFT_SIZE: j+WINDOW_LEFT_SIZE, :], delimiter= ',')
        #print data[j-WINDOW_LEFT_SIZE: j+WINDOW_LEFT_SIZE, :]
        plt.plot(data[j-WINDOW_LEFT_SIZE: j+WINDOW_LEFT_SIZE, :])
        plt.show()
