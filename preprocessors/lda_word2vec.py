# -*- coding: utf-8 -*-
from preprocessors.base import BaseWord2VecPreprocessor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LDAWord2VecPreprocessor(BaseWord2VecPreprocessor):
  def __init__(self):
    super(LDAWord2VecPreprocessor, self).__init__()
    self.lda = LinearDiscriminantAnalysis(solver='svd', shrinkage='auto')
    
  def reduce_dimensionality(self, samples, is_training=True):
    features, targets = zip(*samples)
    if is_training:
      self.lda.fit(features, targets)
    reduced_features = self.pca.transform(features)
    return zip(reduced_features, targets)