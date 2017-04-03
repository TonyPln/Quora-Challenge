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

model.run(create_submission_file=True)

#%%

#dir='~/Téléchargements'
#preprocessor = PCAWord2VecPreprocessor(dir)
#
##%%
#train = pd.read_csv(os.path.join(dir, 'train.csv'), nrows=2000)
#
##%%
#
#preprocessor.preprocess_dataset(train, is_training=True)
##%%
#test = pd.read_csv(os.path.join(dir, 'test.csv'), header=0, nrows=200, skiprows=range(1,8613))
#
##%%
#
#
#preprocessor.transform_samples(test,False)
#
##%%
#
#preprocessor.preprocess_dataset(test, is_training=False)





#%%

#best_classifier = RandomForest(parameter_space[0])
#best_classifier.fit(training_features, training_targets)
#
#model.compute_submission_dataframe(model.classifier_class)

