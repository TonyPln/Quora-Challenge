# -*- coding: utf-8 -*-

from classifiers.base import Classifier
from sklearn.linear_model import LogisticRegression

class RandomForest(Classifier):
  name = 'randomforest'
  def __init__(self, **kwargs):
    self.model = LogisticRegression(**kwargs, n_jobs=-1)
    
  def fit(self, features, targets):
    self.model.fit(features, targets)
    
  def predict(self, features):
    return self.model.predict_proba(features) + 1e-4