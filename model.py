# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from helper import separate_faulty_entries, fill_faulty_results

class Model:
  def __init__(self, preprocessor_class, classifier_class, hyperoptimizer_class, parameter_space, dir='~/Téléchargements'):
    self.preprocessor = preprocessor_class(dir)
    self.classifier_class = classifier_class
    self.hyperoptimizer = hyperoptimizer_class(classifier_class, parameter_space)
    self.dir = dir

  def load_dataset(self, is_training):
    return self.preprocessor.load_dataset(is_training)

  def evaluate(self, features, targets):
    return self.classifier.evaluate(features, targets)
 
  def compute_submission_dataframe(self, classifier):
    features, ids = self.load_dataset(is_training=False)
    faulty_entries, good_entries = separate_faulty_entries(features, ids)
    faulty_entries_result = fill_faulty_results(faulty_entries)
    
    good_ids = list(map(lambda line: line[1], good_entries))
    good_features = np.array(list(map(lambda line: line[0], good_entries)))
    good_entries_result = pd.DataFrame({
      'test_id': good_ids,
      'is_duplicate': classifier.predict(good_features)[:, 1]
    })
    return pd.concat([faulty_entries_result, good_entries_result])
    
  def run(self, create_submission_file=False):
    training_features,validation_features, testing_features, training_targets, validation_targets, testing_targets = self.load_dataset(is_training=True)
    best_params = self.hyperoptimizer.optimize(training_features, training_targets)
    best_classifier = self.classifier_class(**best_params)
    best_classifier.fit(training_features, training_targets)
    training_logloss = best_classifier.evaluate(training_features, training_targets)
    validation_logloss = best_classifier.evaluate(validation_features, validation_targets)
    testing_logloss = best_classifier.evaluate(testing_features, testing_targets)
    
    self.loglosses = {
      'training': training_logloss,
      'validation': validation_logloss,
      'testing': testing_logloss
    }
    
    if create_submission_file:
      best_classifier.fit(
        np.vstack([training_features, validation_features, testing_features]),
        np.hstack([training_targets, validation_targets, testing_targets])
      )
      
      submission_df = self.compute_submission_dataframe(best_classifier)
      submission_df.to_csv(os.path.join(self.dir, 'submission.csv'), index=False)