# -*- coding: utf-8 -*-
from preprocessors.base import BaseWord2VecPreprocessor
from sklearn.decomposition import PCA

class PCAWord2VecPreprocessor(BaseWord2VecPreprocessor):
    def reduce_dimensionality(self, samples):
      pca = PCA(n_components=50, whiten=True)
      return pca.fit_transform(samples)
