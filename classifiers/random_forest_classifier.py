# -*- coding: utf-8 -*-
from classifiers.base import Classifier
from sklearn.ensemble import RandomForestClassifier

class RandomForest(Classifier):
  def __init__(self, **kwargs):
    self.model = RandomForestClassifier(**kwargs, oob_score=True, n_jobs=-1)
    RandomForestClassifier()
    
  def fit(self, features, targets):
    self.model.fit(features, targets)
    
  def predict(self, features):
    return self.model.predict_proba(features) + 1e-4

#%%