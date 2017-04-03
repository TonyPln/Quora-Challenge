from preprocessors import PCAWord2VecPreprocessor
from hyperparameters_optimizers.grid_search import GridSearch
from classifiers.random_forest_classifier import RandomForest
from model import Model
import pandas as pd
import os

preprocessor_class = PCAWord2VecPreprocessor
hyper_opt_class = GridSearch
classifier_class = RandomForest

parameter_space = [
  {'n_estimators': 30, 'max_features': 0.02},
]

model = Model(
  preprocessor_class=preprocessor_class,
  hyperoptimizer_class=hyper_opt_class,
  classifier_class=classifier_class,
  parameter_space=parameter_space
)

#%%
model.run(create_submission_file=True)