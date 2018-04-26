#!/usr/bin/env python2.7
import numpy as np
#import keras
from scipy import signal
import matplotlib.pyplot as plt

data= np.array([])
filename= "FloorRace1.txt"
dataFile= open("data1FloorRace1.txt", "a")

data= np.loadtxt(filename, delimiter= ',', usecols=range(4))
BUFFER_SIZE= 700 #1000 worked
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
    if window.size:
        maxim= np.max(window)
        if maxim > 0.3:
            peaks1= np.argmax(envelope)
            xpeaks= np.append(xpeaks, peaks1)
            peaks=  np.append(peaks, maxim)
            j= i-BUFFER_SIZE+peaks1
            #print data[j-WINDOW_LEFT_SIZE: j+WINDOW_RIGHT_SIZE, :]
            np.savetxt(dataFile, data[j-WINDOW_LEFT_SIZE: j+WINDOW_RIGHT_SIZE, :], delimiter= ',')
            plt.plot(data[j-WINDOW_LEFT_SIZE: j+WINDOW_LEFT_SIZE, :])
            plt.show()
            count+=1

print count
print filename
