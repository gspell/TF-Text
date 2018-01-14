# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for downloading data from 20 Newsgroups, tokenizing, vocabularies.
   Modified from the tutorial data_utils.py for downloading the WMT data set, 
   but there is no French dataset here."""
   
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile

from six.moves import urllib

from tensorflow.python.platform import gfile
from sklearn.datasets import fetch_20newsgroups

import csv
import nltk
from nltk.tokenize import word_tokenize
import string
import cPickle

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


def prepare_labels(filename):
    #print('Prepare Labels (putting as one-hot vector)')
    labels = []
    with open(filename) as file:
        reader = csv.reader(file, delimiter='\n')
        for row in reader:
            labels.extend([int(label) for label in row])
    labels = np.array(labels)
    num_delegate = np.sum(labels)
    percent_delegate = np.sum(labels)/len(labels)
    #print("Number delegating authority: ", num_delegate)
    one_hot = np.zeros((labels.size, labels.max()+1))
    one_hot[np.arange(labels.size),labels]=1
    return one_hot

def basic_tokenizer(sentence):
    """ Very basic tokenizer: split the sentence into a list of tokens.
        Actually just uses the tokenizer from NLTK. Does line by line
        what is done in the preprocess_cnn.py script """
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    X = sentence.split()[:]
    X = ' '.join(X)
    tokens = word_tokenize(X)
    return [t for t in tokens if t]

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with open(data_path) as f:
      counter = 0
      reader = csv.reader(f)
      for line in reader:
        counter += 1
        if counter % 100 == 0:
          print("  processing line %d" % counter)
        if line:
            tokens = tokenizer(line[0]) if tokenizer else basic_tokenizer(line[0])
        for w in tokens:
          word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=False):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=False):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_newsgroups_data(data_dir, vocab_size, tokenizer=None):
  """ We can just use sklearn to get the 20 Newsgroups data, so having an explicit
      data directory is not necessary, as well as the functions involved in downloading """

  # Already went through the download for 20 Newsgroups
  train_path = os.path.join(data_dir, "newsgroups_normalized_train.csv")
  test_path = os.path.join(data_dir, "newsgroups_normalized_test.csv")
  train_labels_path = os.path.join(data_dir, "newsgroups_train_labels.csv")
  test_labels_path = os.path.join(data_dir, "newsgroups_test_labels.csv")
  newsgroup_vocab_path = os.path.join(data_dir, "vocab%d.newsgroups" % vocab_size)
  create_vocabulary(newsgroup_vocab_path, train_path, vocab_size, tokenizer)
  
  # Create token ids for the training data
  newsgroup_train_ids_path = train_path + (".ids%d.newsgroups" % vocab_size)
  newsgroup_test_ids_path = test_path + (".ids%d.newsgroups" % vocab_size)
  data_to_token_ids(train_path, newsgroup_train_ids_path, newsgroup_vocab_path, tokenizer)
  data_to_token_ids(test_path, newsgroup_test_ids_path, newsgroup_vocab_path, tokenizer)
  
  return (newsgroup_train_ids_path, newsgroup_test_ids_path, train_labels_path, test_labels_path, newsgroup_vocab_path)

