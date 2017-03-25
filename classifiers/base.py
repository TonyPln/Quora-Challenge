# -*- coding: utf-8 -*-
from sklearn.metrics import log_loss

class Classifier:
  def fit(self, training_features, training_targets):
    raise NotImplementedError
    
  def predict(self, testing_features):
    raise NotImplementedError
    
  def evaluate(self, features, targets):
    probabilities = self.predict(features)
    return log_loss(targets, probabilities)
