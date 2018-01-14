from __future__ import print_function
import numpy as np
import tensorflow as tf

import functools

def lazy_property(function):
  attribute = '_cache_' + function.__name__
  
  @property
  @functools.wraps(function)
  def decorator(self):
    if not hasattr(self, attribute):
      setattr(self, attribute, function(self))
    return getattr(self, attribute)
  return decorator

class MLP:
  
  def __init__(self, x, y, network_architecture, drop_keep=.75, learning_rate=1e-4):
    self.network_architecture=network_architecture
    self.learning_rate = learning_rate
    self.inputs = x
    self.labels = y
    self.drop_keep = drop_keep
    
    self._initialize_network_params(**self.network_architecture)
    self.logits
    self.prediction
    self.cost
    self.accuracy
    self.optimize
    self.calc_acc()
    self.calc_auc()
    # TODO: Put dropout in
    
  def _initialize_network_params(self, n_input, n_hidden_1, n_hidden_2, n_classes):
    self.weights = {'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
               'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
               'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes])) }
    self.biases = {'b1': tf.Variable(tf.random_normal([n_hidden_1])), 
              'b2': tf.Variable(tf.random_normal([n_hidden_2])),
              'out': tf.Variable(tf.random_normal([n_classes])) }
  
  @lazy_property
  def logits(self):
    # Hidden layer with ReLU activation
    layer_1 = tf.nn.xw_plus_b(self.inputs, self.weights['h1'], self.biases['b1'])
    #layer_1 = tf.nn.relu(layer_1, name="layer_1")
    #layer_1 = tf.nn.dropout(layer_1, self.drop_keep)
    
    # Hidden layer with ReLU activation
    layer_2 = tf.nn.xw_plus_b(layer_1, self.weights['h2'], self.biases['b2'])
    #layer_2 = tf.nn.relu(layer_2, name="layer_2")
    #layer_2 = tf.nn.dropout(layer_2, self.drop_keep)
    
    # Perhaps here is the place to put dropout
    #layer_2_drop = tf.nn.dropout(layer_2, self.drop_keep)
    # Output layer with linear activation
    out_layer = tf.nn.xw_plus_b(layer_2, self.weights['out'], self.biases['out'], name="output_layer")
    return out_layer
  
  @lazy_property
  def prediction(self):
    prediction = tf.argmax(self.logits, 1, name="prediction")
    return prediction
  
  @lazy_property
  def cost(self):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
    return tf.reduce_mean(cross_entropy)
  
  @lazy_property
  def accuracy(self):
    temp_labels = tf.argmax(self.labels, 1)
    correct_predictions = tf.equal(self.prediction, temp_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
    return accuracy
  
  @lazy_property
  def optimize(self):
    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    return optimizer.minimize(self.cost)
  
  def calc_acc(self):
    self.acc, self.acc_update_op = tf.contrib.metrics.streaming_accuracy(tf.to_float(self.prediction), 
                                                                    tf.to_float(tf.argmax(self.labels,1)))

  def calc_auc(self):
    self.auc, self.auc_update_op = tf.contrib.metrics.streaming_auc(tf.to_float(self.prediction), 
                                                                    tf.to_float(tf.argmax(self.labels,1)))
                                                                        
  @staticmethod
  def _weight_and_bias(in_size, out_size):
    weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
    bias = tf.constant(0.1, shape=[out_size])
    return tf.Variable(weight), tf.Variable(bias)

