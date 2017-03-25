# -*- coding: utf-8 -*-
from classifiers.base import Classifier
from sklearn.ensemble import RandomForestClassifier

class RandomForest(Classifier):
  def init(self, **kwargs):
    self.model = RandomForestClassifier(**kwargs)
    
  def fit(self, features, targets):
    self.model.fit(features, targets, n_jobs=-1)
    
  def predict(self, features):
    self.model.predict_proba(features)
    