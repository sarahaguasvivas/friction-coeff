#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu

def conv1d(x, W, b, strides=1): # Conv1D wrapper, with bias and relu activation
	x = tf.nn.conv1d(x, W, strides=[1, strides, strides], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x) 

def conv2d(x, W, b, strides=1):
	# Conv2D wrapper, with bias and relu activation
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x) 

def conv_net(x, weights, biases):
	conv1= conv2d(x, weights['W_0'], biases['bc1'])
	conv1= maxpool2d(conv1, k=2)
	conv2= conv2d(conv1, weights['W_1'], biases['bc1'])
	conv2= maxpool2d(conv2, k=2)

	conv3= conv2d(conv2, weights['W_3'], biases['bc3'])
	conv3= maxpool2d(conv3, k=2)
	
	# Fully connected layer:
	fc1= tf.reshape(conv3, [-1, weights['W_fc'].get_shape().as_list()[0]])
	fc1= tf.add(tf.matmul(fc1, weights['W_fc']), biases['b_fc'])
	fc1= tf.nn.relu(fc1)

	out= tf.add(tf.matmul(fc1, weights['W_sm']), biases['b_sm'])
	return out

	

def maxpool2d(x, k=2):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

training_iters= 200
learning_rate= 1e-3
NUM_ADC= 4
n_classes= 3
WINDOW_SIZE=200/4
batch_size=WINDOW_SIZE

weights = {
    'W_0': tf.get_variable('W0', shape=(50,4,148), initializer=tf.contrib.layers.xavier_initializer()), 
    'W_1': tf.get_variable('W1', shape=(8,10,12,12), initializer=tf.contrib.layers.xavier_initializer()), 
    'W_2': tf.get_variable('W2', shape=(5, 15,12, 12), initializer=tf.contrib.layers.xavier_initializer()), 
    'W_3': tf.get_variable('W3', shape=(3, 20,12, 12), initializer=tf.contrib.layers.xavier_initializer()), 
    'W_fc': tf.get_variable('W4', shape=(200,12), initializer=tf.contrib.layers.xavier_initializer()), 
    'W_sm': tf.get_variable('W5', shape=(12,n_classes), initializer=tf.contrib.layers.xavier_initializer()), 
}

biases = {
    'bc1': tf.get_variable('B0', shape=(12), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(12), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(12), initializer=tf.contrib.layers.xavier_initializer()),
    'bc4': tf.get_variable('B3', shape=(12), initializer=tf.contrib.layers.xavier_initializer()),
    'b_fc': tf.get_variable('B4', shape=(12), initializer=tf.contrib.layers.xavier_initializer()),
    'b_sm': tf.get_variable('B5', shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}
x= tf.placeholder(tf.float32, [WINDOW_SIZE, NUM_ADC])
y= tf.placeholder(tf.int32, n_classes)

data1 = pd.read_csv('data1.csv')
data2 = pd.read_csv('data2.csv')

data1= tf.reshape(data1[:-1-data1.shape[0] % WINDOW_SIZE+ 1], [WINDOW_SIZE, 4, -1])
y= tf.zeros([data1.shape[2], 1], tf.int32)

data2= tf.reshape(data2[:-1-data2.shape[0] % WINDOW_SIZE + 1],[WINDOW_SIZE, 4, -1])
y= tf.concat([y, tf.ones([data2.shape[2], 1], tf.int32)], 0)
x= tf.concat([data1, data2], 2)
x= tf.cast(x, tf.float32)
pred= conv_net(x, weights, biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer= tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction= tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy= tf.reduce_mean(tf.case(correct_prediction, tf.float32))

init= tf.global_variables_initializer()

