# -*- coding: utf-8 -*-
from preprocessors.base import BaseWord2VecPreprocessor

class NaiveWord2VecPreprocessor(BaseWord2VecPreprocessor):
    def reduce_dimensionality(self, samples):
      return samples
