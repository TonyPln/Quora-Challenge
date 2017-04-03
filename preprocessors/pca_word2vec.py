# -*- coding: utf-8 -*-
from preprocessors.base import BaseWord2VecPreprocessor
from sklearn.decomposition import PCA
import numpy as np

class PCAWord2VecPreprocessor(BaseWord2VecPreprocessor):
  def __init__(self, dir):
    super(PCAWord2VecPreprocessor, self).__init__(dir)
    self.pca = PCA(n_components=50, whiten=True)

  def reduce_dimensionality(self, samples, is_training=True):
    features, targets = zip(*samples)
    if is_training:
        self.pca.fit(features)
    reduced_features = self.pca.transform(np.array(features))
    return list(zip(reduced_features, targets))
