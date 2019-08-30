#!/usr/bin/env python3

import struct
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np

NUM_ADC= 4
datafile= open(sys.argv[1]+'.csv', 'r')
datainfo= str(datafile.readlines()[0])
datai= datainfo.split(",")
data= [float(i) for i in datai]
print(max(data))

Data= np.array(data)
print(Data.shape)

if len(Data) % NUM_ADC == 0:
    Data= np.reshape(Data, (-1, NUM_ADC))
else:
    Data= np.reshape(Data[:-1], (-1, NUM_ADC))

for i in range(NUM_ADC):
    plt.plot(Data[:, i])
plt.show()
