#!/usr/bin/env python2.7
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

WINDOW_SIZE= 500
NUM_SENSORS= 4
## Defining ground truth:
GRASS_LABEL= 0.35
WOOD_LABEL= 0.17
FLOOR_LABEL= 0.05

FG1= np.loadtxt('data1FakeGrass1.txt', delimiter= ',')
FG = np.loadtxt('data1FakeGrass.txt', delimiter= ',')
RW= np. loadtxt('data1RuggedWood.txt', delimiter= ',')
RW1= np. loadtxt('data1RuggedWood1.txt', delimiter= ',')
RF= np.loadtxt('data1FloorRace1.txt', delimiter= ',')

# Defining base model
def baseline_model():
    model= Sequential()
    model.add(Dense(13, input_dim=WINDOW_SIZE*NUM_SENSORS, kernel_initializer= 'normal', activation= 'relu'))
    model.add(Dense(1, kernel_initializer= 'normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Preparing the data: 
grass= np.append(FG1, FG, axis=0)
wood= np.append(RW, RW1, axis=0)
floor= RF

labels= np.array([])
data= np.empty((WINDOW_SIZE*NUM_SENSORS, 1))
for i in xrange(0, grass.shape[0], WINDOW_SIZE):
    newD= grass[i:i+WINDOW_SIZE, :].flatten()
    data= np.column_stack([data, newD])
    labels= np.append(labels, GRASS_LABEL)

for i in xrange(0, wood.shape[0], WINDOW_SIZE):
    newD= wood[i:i+WINDOW_SIZE, :].flatten()
    data= np.column_stack([data, newD])
    labels= np.append(labels, WOOD_LABEL)

for i in xrange(0, floor.shape[0], WINDOW_SIZE):
    newD= floor[i:i+WINDOW_SIZE, :].flatten()
    data= np.column_stack([data, newD])
    labels= np.append(labels, FLOOR_LABEL)

print "number of training samples= " + str(len(labels))
print data[:, 1:].shape
seed= 7
np.random.seed(seed)
estimator= KerasRegressor(build_fn=baseline_model, epochs= 100, batch_size=5, verbose=1)
kfold= KFold(n_splits=10, random_state=seed)
results= cross_val_score(estimator, data[:, 1:].T, labels, cv= kfold)
print "Results: " + str(results.mean) + " " + str(results.std()) + "MSE"

