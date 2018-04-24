#!/usr/bin/env python2.7
import numpy as np
#import keras
from scipy import signal 
import matplotlib.pyplot as plt

data= np.array([])
filename= "FakeGrass.txt"

data= np.loadtxt(filename, delimiter= ',', usecols=range(4))

BUFFER_SIZE= 300 
WINDOW_LEFT_SIZE=200
WINDOW_RIGHT_SIZE= 300 
count=0
peaks= np.array([])
xpeaks=np.array([])
for i in xrange(BUFFER_SIZE, data.shape[0], BUFFER_SIZE):
    Window= data[i-BUFFER_SIZE:i, :]
    envelope= signal.hilbert(Window[:, 0])
    peak= signal.argrelextrema(envelope, np.greater)
    window= Window[peak, 0]
    
    if np.max(window) > 0.5:
        maxim= np.max(window)
        print maxim
        peaks1= np.where(window==maxim) 
        print peaks1
        xpeaks= np.append(xpeaks, peaks1)
        peaks= np.append(peaks, maxim)
        count+=1
print count  
plt.plot(data[:, 0])
plt.plot(xpeaks,peaks, 'ro')
plt.show()

