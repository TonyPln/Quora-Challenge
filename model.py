# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from preprocessors import PairPCAWord2VecPreprocessor
from hyperparameters_optimizers.grid_search import GridSearch
from classifiers.random_forest_classifier import RandomForest
from helper import separate_faulty_entries, fill_faulty_results, working_dir


# Parameters
n_components=25
create_submission_file=True
parameter_space = [{
  'n_estimators': 70,
  'max_features': 0.14
}]

# Creation of the naive model
preprocessor = PairPCAWord2VecPreprocessor(n_components)
hyperoptimizer = GridSearch(RandomForest, parameter_space)

# K-fold cross-validation for selecting the best model
training_features,validation_features, testing_features, training_targets, validation_targets, testing_targets = preprocessor.load_dataset(is_training=True)
best_params = hyperoptimizer.optimize(training_features, training_targets)
best_classifier = RandomForest(**best_params)
best_classifier.fit(
  np.vstack([training_features, validation_features, testing_features]),
  np.hstack([training_targets, validation_targets, testing_targets])
)

# Create the submission file
if create_submission_file:
  features, ids = preprocessor.load_dataset(is_training=False)
  faulty_entries, good_entries = separate_faulty_entries(features, ids)
  faulty_entries_result = fill_faulty_results(faulty_entries)
  good_ids = list(map(lambda line: line[1], good_entries))
  good_features = np.array(list(map(lambda line: line[0], good_entries)))
  good_entries_result = pd.DataFrame({
    'test_id': good_ids,
    'is_duplicate': best_classifier.predict(good_features)[:, 1]
  })

  submission_df = pd.concat([faulty_entries_result, good_entries_result])
  submission_df.to_csv(os.path.join(working_dir, 'submission.csv'), index=False)