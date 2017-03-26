from preprocessors import NaiveWord2VecPreprocessor
from hyperparameters_optimizers.grid_search import GridSearch
from classifiers.random_forest_classifier import RandomForest
from model import Model

preprocessor_class = NaiveWord2VecPreprocessor
hyper_opt_class = GridSearch
classifier_class = RandomForest

parameter_space = [
  {'n_estimators': 30, 'max_features': 0.02},
  {'n_estimators': 30, 'max_features': 0.04},
  {'n_estimators': 30, 'max_features': 0.06},
  {'n_estimators': 30, 'max_features': 0.08},
  {'n_estimators': 30, 'max_features': 0.10},
  {'n_estimators': 45, 'max_features': 0.02},
  {'n_estimators': 45, 'max_features': 0.04},
  {'n_estimators': 45, 'max_features': 0.06},
  {'n_estimators': 45, 'max_features': 0.08},
  {'n_estimators': 45, 'max_features': 0.10},
]

#%%
model = Model(
  preprocessor_class=preprocessor_class,
  hyperoptimizer_class=hyper_opt_class,
  classifier_class=classifier_class,
  parameter_space=parameter_space
)

#%%
model.run(create_submission_file=True)