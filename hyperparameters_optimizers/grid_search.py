# -*- coding: utf-8 -*-
import pandas as pd
import os
from helper import working_dir
from hyperparameters_optimizers.base import BaseHyperOptimizer
from functools import reduce
import time

class GridSearch(BaseHyperOptimizer):
  name = 'grid_search'
  def optimize(self, features, targets):
    perfs_params = [
          (self.evaluate_parameters(features, targets, parameters), parameters)
          for parameters in self.parameter_space
      ]
    print(str(time.time()).split('.')[0])
    perfs, params = zip(*perfs_params)
    pd.DataFrame({
      'perf': perfs,
      'parameter_set': params
    }).to_csv(
      os.path.join(
        working_dir,
        'hyperopt_%s_%s_%s.csv' % (
          self.name,
          self.classifier_name,
          str(time.time()).split('.')[0],
        )
      ),
      index=False
    )

    best_params = reduce(
        lambda first, second: first if first[0] > second[0] else second,
        perfs_params
      )[1]
    return best_params