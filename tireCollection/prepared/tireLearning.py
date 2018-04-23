#!/usr/bin/env python2.7
import numpy as np
#import keras

data= np.array([])
filename= "Carpet.txt"
with open(filename) as f:
    line= f.readlines()
    u=np.fromstring(line, dtype=float, sep= ',')
    data= np.append(data,u)

BUFFER_SIZE= 500
print data
for i in xrange(0, f.shape[0], BUFFER_SIZE):
    Window= f[:i, :]
    print Window

