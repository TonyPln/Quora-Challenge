# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np
from sklearn.cross_validation import train_test_split
import gensim

from helper import separate_faulty_entries

np.random.seed(0)

class BasePreprocessor:
  def __init__(self, dir):
    self.dir = dir
    self.train_url = os.path.join(dir, 'train.csv')
    self.test_url = os.path.join(dir, 'test.csv')
    
  def load_dataset(self, is_training):
    url = self.train_url if is_training else self.test_url
    dataset = pd.read_csv(url)
    processed_dataset = self.preprocess_dataset(dataset, is_training)
    features, targets = zip(*processed_dataset)
    if not is_training:
      return np.array(features), np.array(targets)
    return self.split_dataset(np.array(features), np.array(targets))
    
  def split_dataset(self, features, targets):
    training_features, testing_features, training_targets, testing_targets = train_test_split(features, targets, test_size=0.2)
    validation_features, testing_features, validation_targets, testing_targets = train_test_split(testing_features, testing_targets, test_size=0.5)
    return training_features, validation_features, testing_features, training_targets, validation_targets, testing_targets
  
  def transform_samples(self, dataset, is_training):
    transformed_samples = dataset.apply(
      lambda sample: self.transform_sample(sample, is_training),
      axis=1
    ).tolist()
    return list(filter(lambda line: line[0] is not None, transformed_samples)) if is_training else transformed_samples
  
  def transform_sample(self, sample, is_training):
    raise NotImplementedError
    
  def preprocess_dataset(self, dataset, is_training):
    transformed_samples = self.transform_samples(dataset, is_training)
    return self.reduce_dimensionality_wrapper(transformed_samples, is_training)
  
  def reduce_dimensionality_wrapper(self, samples, is_training):
    try:
      if not is_training:
        features, ids = zip(*samples)
        faulty_entries, good_entries = separate_faulty_entries(features, ids)
        return self.reduce_dimensionality(good_entries) + faulty_entries
      else:
        return self.reduce_dimensionality(samples)
    except TypeError:
      print(faulty_entries)
      raise TypeError

  def reduce_dimensionality(self, samples):
    raise NotImplementedError
    
class BaseWord2VecPreprocessor(BasePreprocessor):
  def __init__(self, dir):
    super(BaseWord2VecPreprocessor, self).__init__(dir)
    model_path = os.path.join(self.dir, 'GoogleNews-vectors-negative300.bin')
    self.model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

  def try_w2v_representation(self, word):
    try:
        if word[-1] == '?':
            word = word[:-1]
        return self.model[word]
    except:
        return None

  def sentence2vec(self, sentence):
    words = str(sentence).split(' ')
    features_list = [
        self.try_w2v_representation(word)
        for word in words
        if self.try_w2v_representation(word) is not None
    ]
    return np.mean(features_list, axis=0)

  def transform_sample(self, sample, is_training):
    question1 = sample.question1
    question2 = sample.question2
    
    in_vec = self.sentence2vec(question1)
    out_vec = self.sentence2vec(question2)
    
    target = sample.is_duplicate if is_training else sample.test_id
    
    if in_vec.size < 2 or out_vec.size < 2:
        return None, target
    else:
        return np.hstack([in_vec, out_vec]), target
      
  def reduce_dimensionality(self, samples):
    raise NotImplementedError

  def __del__(self):
    del self.model
