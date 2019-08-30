#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from numpy import genfromtxt


listFiles= os.listdir('tire_data/results')

print(listFiles)

DataCement= pd.DataFrame()
DataCarpet= pd.DataFrame()

for i in listFiles:
    filefile= open(os.path.join('tire_data/results', i))
    data= genfromtxt(filefile, delimiter=',')
    data= np.reshape(data, (-1, 2))

    i= i.replace('.csv', '')
    labelTitle= i.split("_")
    if labelTitle[1]== "cement":
        DataCement= DataCement.append(pd.DataFrame(data), ignore_index=True)
    if labelTitle[1]=="carpet":
        DataCarpet= DataCarpet.append(pd.DataFrame(data), ignore_index=True)

DataCement.to_csv("cement_results.csv", index=False)
DataCarpet.to_csv("carpet_results.csv", index=False)

