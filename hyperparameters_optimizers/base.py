# -*- coding: utf-8 -*-
from functools import reduce
import time
import numpy as np

from sklearn.metrics import log_loss
from sklearn.model_selection import KFold


class BaseHyperOptimizer:
  def __init__(self, classifier_class):
    self.classifier_class = classifier_class
    
  def evaluate_parameters_on_fold(self, current_fold_indices, features, targets, parameters):
    current_training_indices, current_testing_indices = current_fold_indices
    current_training_features = features[current_training_indices]
    current_training_targets = targets[current_training_indices]
    current_testing_features = features[current_testing_indices]
    current_testing_targets = targets[current_testing_indices]

    classifier = self.classifier_class(parameters)

    classifier.fit(current_training_features, current_training_targets)
    testing_classification_rate = log_loss(current_testing_targets,classifier.predict(current_testing_features))
    training_classification_rate = log_loss(current_training_targets,classifier.predict(current_training_features))
    return training_classification_rate, testing_classification_rate

  def optimize(self, features, targets, parameters):
    raise NotImplementedError

  def evaluate_parameters(self, features, targets, parameters):
    np.random.seed(0)
    kf=KFold(n_splits=10)
    fold_indices = kf.split(
      X=features,
      y=targets
    )
    classification_rates = [
        self.evaluate_parameters_on_fold(parameters, fold, features, targets)
        for fold in fold_indices
    ]
    training_classification_rates, testing_classification_rates = list(zip(*classification_rates))
    training_avg_classification_rate = np.mean(training_classification_rates)
    testing_avg_classification_rate = np.mean(testing_classification_rates)
    
    print(time.ctime(), training_avg_classification_rate, testing_avg_classification_rate)
    return testing_avg_classification_rate

class GridSearch(BaseHyperOptimizer):
  def optimize(self, features, targets, parameter_grid):
    perfs_params = [
          (self.evaluate_parameters(features, targets, parameters), parameters)
          for parameters in parameter_grid
      ]

    best_params = reduce(
        lambda first, second: first if first[0] > second[0] else second,
        perfs_params
      )[1]
    return best_params
