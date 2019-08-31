#!/usr/bin/env python3

import pandas as pd
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

listFiles= os.listdir('data')
print(listFiles)
NUM_ADC= 4
WINDOW_SIZE= 100

Data= pd.DataFrame()

for i in listFiles:
    filefile= open(os.path.join('data', i))
    data= str(filefile.readlines()[0])
    datai= data.split(",")
    data= [float(i) for i in datai]

    niapa= len(data)%(WINDOW_SIZE*NUM_ADC)
    if niapa != 0:
        data= data[:-niapa]
    data= np.reshape(data, (-1, WINDOW_SIZE*NUM_ADC))

    i= i.replace('.csv', '')
    labelTitle= i.split("_")


    if labelTitle[0] == "cement":
        data= np.append(data, 0*np.ones((data.shape[0], 1)), axis= 1)
        print("cement: ", data.shape)
    if labelTitle[0]=="carpet":
        data= np.append(data, 1*np.ones((data.shape[0], 1)), axis= 1)
        print("carpet: ", data.shape)

    Data= Data.append(pd.DataFrame(data), ignore_index=True)
    print("file:  "+ str(i) + " processed")
print("Datafile saved!")
Data.to_csv("tire_data_August_28.csv", index=False)

