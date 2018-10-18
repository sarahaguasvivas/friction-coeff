#!/usr/bin/env python3

#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS, #  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pandas as pd
import random
tf.logging.set_verbosity(tf.logging.INFO) 

def cnn_model_fn(features, labels, mode):
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 50, 4, 1])
  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32, 
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu)
  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=36,
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu)
  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 1])
  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=2)
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
  # Configure the Training Op (for TRAIN mode)
  
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  print("add evaluation metrics")
  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
  # Load training and eval data
  train_data = np.load('terrains_train_features.npy') 
  train_labels = np.load('terrains_train_labels.npy') 
  train_labels= train_labels.astype(int)
  train_labels= np.array(train_labels).reshape(-1)  
  train_data= train_data.transpose((2, 0, 1))
  eval_data = np.load('terrains_test_features.npy') 
  eval_data = eval_data.transpose((2, 0, 1))
  eval_labels =np.load('terrains_test_labels.npy') 
  eval_labels= eval_labels.astype(int)
  eval_labels= np.array(eval_labels).reshape(-1)
# Create the Estimator
  terrains_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/terrains_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=2,
      num_epochs=None,
      shuffle=True)
  terrains_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = terrains_classifier.evaluate(input_fn=eval_input_fn)
#  print(eval_results)


if __name__ == "__main__":
  WINDOW_SIZE= int(200/4)

  train= 0.8

  data1 = np.genfromtxt('data1.csv', delimiter=',')
  data2 = np.genfromtxt('data2.csv', delimiter=',')
  data1= np.array(data1)
  data2= np.array(data2)
  data1 = np.reshape(data1[:-1-data1.shape[0] % WINDOW_SIZE + 1], (WINDOW_SIZE, 4, -1))
  y= np.zeros((data1.shape[2], 1)) 
  data2 = np.reshape(data2[:-1-data2.shape[0] % WINDOW_SIZE + 1], (WINDOW_SIZE, 4, -1))
  y= np.vstack((y, np.ones((data2.shape[2], 1))))
  x= np.concatenate((data1, data2), axis=2)

  # Generating Random Training Samples:
  indexes= random.sample(range(x.shape[2]),int(train*int(x.shape[2])))
  missing= list(set(range(x.shape[2])) - set(indexes))

  data_train=x[:, :, indexes]
  label_train=y[indexes]
  data_test= x[:, :, missing]

  label_test= y[missing]  

  np.save('terrains_train_features', data_train)
  np.save('terrains_train_labels', label_train)
  np.save('terrains_test_features', data_test)
  np.save('terrains_test_labels', label_test)
  
  tf.app.run()

