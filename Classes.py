#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:35:19 2017

@author: tng
"""

import pandas as pd
import os
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from functools import reduce




class Optimizer:

  def __init__(self, name):
    self.name = name
    return
    
  def optimize(self, classifier, features, targets, **estimators_range):
    return self.cross_validation(classifier, features, targets, **estimators_range)
    

  def evaluate_parameters_on_fold(classifier, current_fold_indices, features, targets):
    current_training_indices, current_testing_indices = current_fold_indices
    current_training_features = features[current_training_indices]
    current_training_targets = targets[current_training_indices]
    current_testing_features = features[current_testing_indices]
    current_testing_targets = targets[current_testing_indices]
    classifier.fit(current_training_features, current_training_targets)
    testing_classification_rate = log_loss(current_testing_targets,classifier.predict(current_testing_features))
    training_classification_rate = log_loss(current_training_targets,classifier.predict(current_training_features))
    
    return training_classification_rate, testing_classification_rate
  
  
  
  
  def evaluate_parameters(self, classifier, features, targets):
    np.random.seed(0)
    kf=KFold(n_splits=10)
    fold_indices = kf.split(
      X=features,
      y=targets
    )
    classification_rates = [
        self.evaluate_parameters_on_fold(classifier, fold, features, targets)
        for fold in fold_indices
    ]
    training_classification_rates, testing_classification_rates = list(zip(*classification_rates))
    training_avg_classification_rate = np.mean(training_classification_rates)
    testing_avg_classification_rate = np.mean(testing_classification_rates)
    
    print(time.ctime(), training_avg_classification_rate, testing_avg_classification_rate)
    return testing_avg_classification_rate
  
  
  
  
  def cross_validation(self, classifier, training_features, training_targets, **estimators_range):
    best_params={}
    for key in estimators_range.keys():
      perfs_params = [
          (self.evaluate_parameters(classifier.update_parameters({key:n_estimators}), training_features, training_targets), n_estimators)
          for n_estimators in estimators_range[key]
      ]
      best_params[key] = reduce(
        lambda first, second: first if first[0] > second[0] else second,
        perfs_params
      )[1]
    return best_params










class Classifier:
  """Main class for classifiers
  
            Currently available classifiers:
                'RandomForest'
  """
              
  available_classif={
      'RandomForest': RandomForestClassifier
      }
  
  default_params={
      'RandomForest': {
        'n_estimators': 10,
        'criterion': "gini",
        'n_jobs': -1
        }
      }
      
  optim_parameters={
      'RandomForest': ['n_estimators']
      }
  
  
  def __init__(self, name, **kwargs):
    self.name = name
    self.parameters = {**(Classifier.default_params[name]), **kwargs}
    self.classifier = Classifier.available_classif[name](**self.parameters)
    return
      
  
  def fit(self, features, targets):
    
    return self.classifier.fit(features, targets)
  
  
  def predict(self, features):
    
    return self.classifier.predict_proba(features)
  
  def optimize(self, optimizer, features, targets, **estimators_range): ## n_estimators = [5, 10, 20, 50, 100]
    best_params = optimizer.optimize(self, features, targets, **estimators_range)
    self.update_parameters(best_params)
    print("Optimization done")
    
    return
      
  def update_parameters(self, kwargs):
    self.parameters={**self.parameters, **kwargs}
    self.classifier = Classifier.available_classif[self.name](**self.parameters)
    return self
  
  