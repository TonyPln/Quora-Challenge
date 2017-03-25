# -*- coding: utf-8 -*-

class Model:
  def __init__(self, preprocessor, classifier, hyperoptimizer):
    self.preprocessor = preprocessor
    self.classifier = classifier
    self.hyperoptimizer = hyperoptimizer
    
  