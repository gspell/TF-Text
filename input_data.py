"""Functions for downloading and reading MNIST data."""
from __future__ import print_function
import gzip
import os
import urllib

import collections
from collections import defaultdict
from collections import Counter


import csv
import numpy
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import string
import re
from string import punctuation, digits
import random
from random import shuffle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups


def extract_documents(filename):
  """Extract the documents into a 2D uint8 numpy array [index, x]."""
  print('Extracting', filename)
  with open(filename) as file:
    reader = csv.reader(file)
    docs = file.readlines()
  X = [doc.split()[:] for doc in docs]
  X = [' '.join(doc) for doc in X]
  data = []
  sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
  for doc in X:
      sentence = sent_detector.tokenize(doc)
      result = ''
      for s in sentence:
          tokens = word_tokenize(s)
          result += ' ' + ' '.join(tokens)
      data.append(result)
  return numpy.squeeze(numpy.array(data))


def dense_to_one_hot(labels_dense, num_classes=20):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def extract_labels(filename, one_hot=False):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  labels = []
  with open(filename) as file:
      reader = csv.reader(file, delimiter='\n')
      for row in reader:
          labels.extend([int(label) for label in row])
  labels = numpy .array(labels)
  if one_hot:
    return dense_to_one_hot(labels)
  return labels

def perform_split(data, percent_holdout):
  num_holdout = int(len(data) * percent_holdout)

  data = numpy.array(data, dtype=object)
  shuffle(data)
  train_set = data[:-num_holdout]
  test_set = data[-num_holdout:]
  return train_set, test_set

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):  #from Duke CS 290 course, lab 2
  stemmed = []
  for item in tokens:
      stemmed.append(stemmer.stem(item))
  return stemmed
    
def tokenize(text):
  tokens = nltk.word_tokenize(text)
  stems = stem_tokens(tokens, stemmer)
  return stems

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

class DataSet(object):

  def __init__(self, documents, labels, fake_data=False):
    if fake_data:
      self._num_examples = 10000
    else:
      self._num_examples = numpy.shape(documents)[0]
    
    
      """ Might want some sort of normalization in here at some point """
    self._documents = documents
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._vocab_size = numpy.shape(documents)[1]
  @property
  def vocabulary_size(self):
    return self._vocab_size
  @property
  def documents(self):
    return self._documents

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1.0 for _ in xrange(784)]
      fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._documents = self._documents[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._documents[start:end], self._labels[start:end]

class SemiDataSet(object):
    def __init__(self, documents, labels, n_labeled, supervised=False):
        self.n_labeled = n_labeled
        self.supervised = supervised
        # Unlabled DataSet
        self.unlabeled_ds = DataSet(documents, labels)

        # Labeled DataSet
        self.num_examples = self.unlabeled_ds.num_examples
        indices = numpy.arange(self.num_examples)
        shuffled_indices = numpy.random.permutation(indices)
        documents = documents[shuffled_indices]
        labels = labels[shuffled_indices]
        y = numpy.array([numpy.arange(2)[l==1][0] for l in labels])
        """
        idx = indices[y==0][:5]
        n_classes = y.max() + 1
        n_from_each_class = n_labeled / n_classes
        i_labeled = []
        for c in range(n_classes):
            i = indices[y==c][:n_from_each_class]
            i_labeled += list(i)
        l_docs = documents[i_labeled]
        l_labels = labels[i_labeled]

        """
        l_docs = documents[:n_labeled]
        l_labels = labels[:n_labeled]
        self.labeled_ds = DataSet(l_docs, l_labels)
        print(numpy.shape(self.labeled_ds.documents))
        
    def next_batch(self, batch_size):
      if not self.supervised:
        unlabeled_docs, _ = self.unlabeled_ds.next_batch(batch_size)
        if batch_size > self.n_labeled:
            labeled_docs, labels = self.labeled_ds.next_batch(self.n_labeled)
        else:
            labeled_docs, labels = self.labeled_ds.next_batch(batch_size)
        documents = numpy.vstack([labeled_docs, unlabeled_docs])
      else:
        documents, labels = self.labeled_ds.next_batch(batch_size)
      return documents, labels

def read_newsgroups_data_sets(train_dir, n_labeled=None, run_num = 1, 
                              fake_data=False, one_hot=False, supervised=True, do_tfidf=True):
  class DataSets(object):
    pass
  data_sets = DataSets()

  newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
  newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

  numpy.random.seed(1+run_num)
  random.seed(a=100+run_num)

  train_labels_one_hot = dense_to_one_hot(newsgroups_train.target)
  test_labels_one_hot = dense_to_one_hot(newsgroups_test.target)
  if do_tfidf:
    print("Vectorizing to TF-IDF")
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words = "english", min_df=5)
    newsgroups_train_vectors = tfidf.fit_transform(newsgroups_train.data)
    newsgroups_test_vectors = tfidf.transform(newsgroups_test.data)
  if not supervised:
    print("Creating Semi-Supervised Training Dataset")
    data_sets.train = SemiDataSet(newsgroups_train_vectors, train_labels_one_hot, n_labeled, supervised)
  else:
    print("Creating Supervised Training DataSet Object")
    data_sets.train = DataSet(newsgroups_train_vectors, train_labels_one_hot)
  print(data_sets.train.vocabulary_size)
  print("Creating test DataSet")
  data_sets.test = DataSet(newsgroups_test_vectors, test_labels_one_hot)

  return data_sets

def read_imdb_data_sets(train_dir, n_labeled=None, run_num = 1, 
                              fake_data=False, one_hot=False, supervised=True, do_tfidf=True):
  class DataSets(object):
    pass
  data_sets = DataSets()

  positive_examples = list(open("./rt-polaritydata/rt-polarity.pos", "r").readlines())
  positive_examples = [s.strip() for s in positive_examples]
  negative_examples = list(open("./rt-polaritydata/rt-polarity.neg", "r").readlines())
  negative_examples = [s.strip() for s in negative_examples]
  # Split by words
  x_text = positive_examples + negative_examples
  x_text = [clean_str(sent) for sent in x_text]
  # Generate labels
  positive_labels = [[0, 1] for _ in positive_examples]
  negative_labels = [[1, 0] for _ in negative_examples]
  #y = numpy.concatenate([positive_labels, negative_labels], 0)
  y = positive_labels + negative_labels
  
  numpy.random.seed(1+run_num)
  random.seed(a=100+run_num)
  
  train_set, test_set = perform_split(list(zip(x_text, y)), percent_holdout = 0.2)
  train_docs, train_labels = zip(*train_set)
  test_docs, test_labels = zip(*test_set)
  train_labels = numpy.array(train_labels)
  test_labels = numpy.array(test_labels)
  if do_tfidf:
    print("Vectorizing to TF-IDF")
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words = "english", min_df=5)
    imdb_train_vectors = tfidf.fit_transform(train_docs)
    imdb_test_vectors = tfidf.transform(test_docs)
    print("Creating Semi-Supervised Training Dataset")
    data_sets.train = SemiDataSet(imdb_train_vectors, train_labels, n_labeled, supervised)
  """
  if not supervised:
    print("Creating Semi-Supervised Training Dataset")
    data_sets.train = SemiDataSet(imdb_train_vectors, train_labels_one_hot, n_labeled, supervised)
  else:
    print("Creating Supervised Training DataSet Object")
    data_sets.train = DataSet(imdb_train_vectors, train_labels)
  """
  #print(data_sets.train.vocabulary_size)
  print("Creating test DataSet")
  data_sets.test = DataSet(imdb_test_vectors, test_labels)

  return data_sets
