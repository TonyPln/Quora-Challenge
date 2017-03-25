# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np

from sklearn.cross_validation import train_test_split

#%%
estimators_range = (5,)
np.random.seed(0)

#%%
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
      return np.array(features)
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
    return filter(bool, transformed_samples)
  
  def transform_sample(self, sample, is_training):
    raise NotImplementedError
    
  def preprocess_dataset(self, dataset, is_training):
    transformed_samples = self.transform_samples(dataset, is_training)
    return self.reduce_dimensionality(transformed_samples)
 
  def reduce_dimensionality(self, samples):
    raise NotImplementedError
    

class BaseNaivePreprocessor(BasePreprocessor):
  def reduce_dimensionality(self, samples):
    return samples
