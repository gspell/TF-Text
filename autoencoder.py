""" Auto Encoder Example.
Using an auto-encoder on MNIST handwritten digits dataset.
"""

from __future__ import division, print_function, absolute_import

import sys
sys.path.append('../')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Autoencoder:
  
  def __init__(self, x, network_architecture, learning_rate=0.01):
    self.network_architecture=network_architecture
    self.learning_rate = learning_rate
    self.inputs = x
    #self._initialize_placeholders(**self.network_architecture) # Unpacks the dictionary. Not all dictionary keys necessary
    self._initialize_network_params(**self.network_architecture)
    self.encoded_vecs = self.encoder(x)
    self.decoded_vecs = self.decoder(self.encoded_vecs)
    self._optimizer()
    
    # TODO: Put dropout in
    
  def _initialize_network_params(self, n_input, n_hidden_1, n_hidden_2):
  
    self.weights = {
          'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
          'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
          'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
          'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
    }

    self.biases =  {
          'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])), 
          'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])), 
          'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
          'decoder_b2': tf.Variable(tf.random_normal([n_input])),
    }
  """
  def _initialize_placeholders(self, n_input, n_hidden_1, n_hidden_2):
    self.inputs = tf.placeholder(tf.float32, [None, n_input])
  """

  def encoder(self, x):
    # Encoder hidden layer 1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']), self.biases['encoder_b1']))
    # Encoder hidden layer 2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']), self.biases['encoder_b2']))
    
    return layer_2
  

  def decoder(self, x):
    # Decoder hidden layer 1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']), self.biases['decoder_b1']))
    # Decoder hidden layer 2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']), self.biases['decoder_b2']))
    
    return layer_2
  
  def _optimizer(self):
    self.cost = tf.reduce_mean(tf.pow(self.inputs - self.decoded_vecs, 2))
    self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)

