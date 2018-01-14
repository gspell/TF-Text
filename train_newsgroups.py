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
from sklearn.datasets import fetch_20newsgroups

from scipy.sparse import csr_matrix, vstack

import input_data

from MLP_classifier import MLP
from autoencoder import Autoencoder
from autoencoder_classifier import AutoencoderClassifier

tf.flags.DEFINE_string("data_path", ".", "data_path")
tf.flags.DEFINE_string("results_dir", ".", "results_directory")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


def train(mlp_network_params, AE_network_params, per, run_num,
          learning_rate=1e-4, batch_size=256, training_epochs=300, display_step=1):
          
  session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
  session_config.gpu_options.allow_growth=True
  sess = tf.Session(config=session_config)
  supervised=True
  default_train_size = int(0.8 * 1059)
  new_train_size = int(default_train_size * per)
  print("Loading Newsgroup data")
  data = input_data.read_newsgroups_data_sets(".", n_labeled = None, 
                                              run_num = run_num, one_hot=True, supervised=supervised)
  mlp_network_params['n_input'] = data.train.vocabulary_size
  #AE_network_params['n_input'] = data.train.vocabulary_size
  test_docs = data.test.documents.toarray()
  print("Beggining Training")
  with  sess.as_default():
    tf.set_random_seed(1)
    model = AutoencoderClassifier(mlp_network_params, AE_network_params, batch_size, learning_rate)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for epoch in range(training_epochs):
      avg_cost = 0.
      sess.run(tf.local_variables_initializer())
      total_batch = int(data.train.num_examples/batch_size)
      # Loop over all batches
      for i in range(total_batch):
      
        batch_x, batch_y = data.train.next_batch(batch_size)
        if not supervised:
          batch_x = np.vstack([batch_x[0][0].toarray(), batch_x[1][0].toarray()])
        else:
          batch_x = batch_x.toarray()
        #batch_x = np.squeeze(batch_x).toarray()
        #print(batch_x)
        batch_y = np.squeeze(batch_y)
        _, c, _= sess.run([model.optimizer, model.cost, model.acc_update_op], 
                                feed_dict={model.inputs: batch_x, model.labels: batch_y, model.num_labeled: batch_size})

        # Compute average loss
        avg_cost += c / total_batch
        #print("Accuracy on this batch: %.3f" % acc)
      # Display logs per epoch step
      if epoch % display_step == 0:
        train_acc = np.squeeze(sess.run([model.acc]))
        print("Epoch: %d cost= %.9f AUC: %.3f" % ((epoch+1), avg_cost, train_acc))
        sess.run(tf.local_variables_initializer())
        _ = sess.run([model.acc_update_op], feed_dict={model.inputs: test_docs, model.labels: data.test.labels,
                                                   model.num_labeled:  len(test_docs)})
        test_acc = np.squeeze(sess.run([model.acc]))
        print("Eval auc %.3f" % test_acc)
        sess.run(tf.local_variables_initializer())
    print("Optimization Finished!")
    
    # Test model by calculating accuracy

    _ = sess.run([model.acc_update_op], feed_dict={model.inputs: test_docs, model.labels: data.test.labels,
                                                   model.num_labeled:  len(test_docs)})
    test_acc = sess.run([model.acc])
    return np.squeeze(test_acc)
  sess.close()
  tf.reset_default_graph()
def main():

  n_input_auto = None
  n_hidden_1_auto = 256 # 1st layer number of features
  n_hidden_2_auto = 256 # 2nd layer number of features

  n_hidden_1_mlp = 256
  n_hidden_2_mlp = 256
  n_classes = 20 # number of newsgroups
  
  mlp_network_params = dict(n_input=n_hidden_2_auto, n_hidden_1=n_hidden_1_mlp, n_hidden_2=n_hidden_2_mlp, n_classes=n_classes)
  AE_network_params = dict(n_input=n_input_auto, n_hidden_1=n_hidden_1_auto, n_hidden_2=n_hidden_2_auto)
  percents = np.logspace(-2, 0, 20)
  num_runs = 10
  auc = train(mlp_network_params, AE_network_params, 1.0, 1)
  print("AUC %.4f" % auc)

if __name__ == '__main__':
  main()
