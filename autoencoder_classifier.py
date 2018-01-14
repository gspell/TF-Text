from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals

import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import sys
sys.path.append('../')
import time

import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
from six.moves import xrange  # pylint: disable=redefined-builtin
#from builtins import str
import tensorflow as tf

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, vstack

import input_data


from MLP_classifier import MLP
from autoencoder import Autoencoder


class AutoencoderClassifier:

  def __init__(self, mlp_network_params, AE_network_params, batch_size, learning_rate=0.001):
    self.mlp_network_params = mlp_network_params
    self.AE_network_params = AE_network_params
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.loss_ratio = 1
    self._initialize_placeholders(**AE_network_params)
    self._network()
    self._optimizer()
    self.accuracy()

  def _initialize_placeholders(self, n_input, n_hidden_1, n_hidden_2):
    self.inputs = tf.placeholder(tf.float32, [None, n_input])
    self.labels = tf.placeholder(tf.float32, [None, 2])
    self.num_labeled = tf.placeholder(tf.int32)
    self.drop_keep = tf.placeholder(tf.float32)
    
  def _network(self):
    self.autoencoder = Autoencoder(self.inputs, self.AE_network_params, self.learning_rate)
    features = self.autoencoder.encoded_vecs
    labeled_features = tf.slice(features, [0, 0], [self.num_labeled, -1])
    #labeled_features = tf.nn.dropout(tf.slice(self.inputs, [0,0], [self.num_labeled, -1]), 0.75)
    self.classifier = MLP(labeled_features, self.labels, self.mlp_network_params, self.learning_rate)
    
  def _optimizer(self):
    self.recon_loss = self.autoencoder.cost * self.loss_ratio
    self.class_loss = self.classifier.cost
    
    self.cost = tf.reduce_sum(self.recon_loss + self.class_loss)
    
    self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.cost)
  
  def accuracy(self):
    self.acc_update_op = self.classifier.acc_update_op
    self.acc = self.classifier.acc
    self.auc_update_op = self.classifier.auc_update_op
    self.auc = self.classifier.auc
    #self.acc = self.classifier.accuracy

