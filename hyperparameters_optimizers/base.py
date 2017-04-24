# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

class BaseHyperOptimizer:
  def __init__(self, classifier_class, parameter_space):
    self.classifier_name = classifier_class.name
    self.classifier_class = classifier_class
    self.parameter_space = parameter_space

  def evaluate_parameters_on_fold(self, current_fold_indices, features, targets, parameter_set):
    current_training_indices, current_testing_indices = current_fold_indices
    current_training_features = features[current_training_indices]
    current_training_targets = targets[current_training_indices]
    current_testing_features = features[current_testing_indices]
    current_testing_targets = targets[current_testing_indices]

    classifier = self.classifier_class(**parameter_set)

    classifier.fit(current_training_features, current_training_targets)
    testing_classification_rate = log_loss(current_testing_targets,classifier.predict(current_testing_features))
    training_classification_rate = log_loss(current_training_targets,classifier.predict(current_training_features))
    return training_classification_rate, testing_classification_rate

  def optimize(self, features, targets):
    raise NotImplementedError

  def evaluate_parameters(self, features, targets, parameter_set):
    print("Evaluating parameters:", parameter_set, end=", ")
    np.random.seed(0)
    kf=KFold(n_splits=10)
    fold_indices = kf.split(
      X=features,
      y=targets
    )
    classification_rates = [
        self.evaluate_parameters_on_fold(fold, features, targets, parameter_set)
        for fold in fold_indices
    ]
    training_classification_rates, testing_classification_rates = list(zip(*classification_rates))
    testing_avg_classification_rate = np.mean(testing_classification_rates)
    print("Score:", testing_avg_classification_rate)
    return testing_avg_classification_rate