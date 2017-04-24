# -*- coding: utf-8 -*-
from preprocessors.base import BaseWord2VecPreprocessor
from sklearn.decomposition import PCA
import numpy as np

class PairPCAWord2VecPreprocessor(BaseWord2VecPreprocessor):
  def __init__(self, n_components):
    super(PairPCAWord2VecPreprocessor, self).__init__()
    self.pca = PCA(n_components=n_components, whiten=True)
    print(n_components)
    
  def reduce_dimensionality(self, samples, is_training=True):
    features, targets = zip(*samples)
    features_array = np.array(features)
    
    if not is_training:
      print(features_array.shape)
    
    n_features = int(features_array.shape[1] / 2)
    
    if is_training:
      question_features = np.vstack([
          features_array[:, range(n_features)],
          features_array[:, range(n_features, 2 * n_features)]
      ])
      self.pca.fit(question_features)

    first_questions = features_array[:, range(n_features)]
    second_questions = features_array[:, range(n_features, 2 * n_features)]
    
    reduced_first_questions = self.pca.transform(first_questions)
    reduced_second_questions = self.pca.transform(second_questions)
    
    reduced_features = np.hstack([
      reduced_first_questions,
      reduced_second_questions
    ])

    return list(zip(reduced_features, targets))
