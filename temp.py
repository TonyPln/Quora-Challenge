from preprocessors.naive_word2vec import NaiveWord2VecPreprocessor
from hyperparameters_optimizers.base import GridSearch
from classifiers.random_forest_classifier import RandomForest

dir = '~/Téléchargements'
preprocessor = NaiveWord2VecPreprocessor(dir)
dataset = preprocessor.load_dataset(is_training=True)
training_features, validation_features, testing_features, training_targets, validation_targets, testing_targets = dataset

#%%
grid_search = GridSearch(RandomForest)
parameters_grid = [{'n_estimators': 5}]
grid_search.optimize()