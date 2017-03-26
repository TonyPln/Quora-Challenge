# -*- coding: utf-8 -*-
from hyperparameters_optimizers.base import BaseHyperOptimizer
from functools import reduce

class GridSearch(BaseHyperOptimizer):
  def optimize(self, features, targets):
    perfs_params = [
          (self.evaluate_parameters(features, targets, parameters), parameters)
          for parameters in self.parameter_space
      ]

    best_params = reduce(
        lambda first, second: first if first[0] > second[0] else second,
        perfs_params
      )[1]
    return best_params