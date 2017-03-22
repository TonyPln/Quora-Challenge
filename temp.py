#%%
import pandas as pd
import os
import gensim
import numpy as np
import time

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

#%%
np.random.seed(0)

dir = '~/Téléchargements'
url = os.path.join(dir, 'train.csv')
model = os.path.join(dir, 'GoogleNews-vectors-negative300.bin')

#testdf=pd.read_csv(os.path.join(dir, 'test.csv'))
model = gensim.models.KeyedVectors.load_word2vec_format(model, binary=True)


#%%
def try_w2v_representation(word):
  try:
      if word[-1] == '?':
          word = word[:-1]
      return model[word]
  except:
      return None

def sentence2vec(sentence):
  words = str(sentence).split(' ')
  features_list = [
      try_w2v_representation(word)
      for word in words
      if try_w2v_representation(word) is not None
  ]
  return np.mean(features_list, axis=0)

def pair2vec(line):
  question1 = line.question1
  question2 = line.question2
  
  in_vec = sentence2vec(question1)
  out_vec = sentence2vec(question2)
  
  if in_vec.size < 2 or out_vec.size < 2:
      return None
  else:
      return np.hstack([in_vec, out_vec]), line.is_duplicate

def split_dataset(features, targets, test_ratio):
  return train_test_split(features, targets, test_size=test_ratio)

def load_dataset():
  training_df = pd.read_csv(url)
  processed_training_data = [
      line
      for line in training_df.apply(pair2vec, axis=1).tolist()
      if line is not None
  ]
  features, targets = zip(*processed_training_data)
  features = np.array(features)
  targets = np.array(targets)
  
  return split_dataset(features, targets, test_ratio=0.2)

def evaluate_parameters_on_fold(n_estimators, current_fold_indices, features, targets):
  current_training_indices, current_testing_indices = current_fold_indices
  current_training_features = features[current_training_indices]
  current_training_targets = targets[current_training_indices]
  current_testing_features = features[current_testing_indices]
  current_testing_targets = targets[current_testing_indices]
  rfc = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
  rfc.fit(current_training_features, current_training_targets)
  classification_rate = np.sum(rfc.predict(current_testing_features)==current_testing_targets)/len(current_testing_targets)
  return classification_rate 

def evaluate_parameters(n_estimators, features, targets):
  kf=KFold(n_splits=10)
  fold_indices = kf.split(
    X=features,
    y=targets
  )
  classification_rates = [
      evaluate_parameters_on_fold(n_estimators, fold, features, targets)
      for fold in fold_indices
  ]
  avg_classification_rate = np.mean(classification_rates)
  print(time.ctime(), n_estimators, avg_classification_rate)
  return avg_classification_rate 

def cross_validation(estimators_range, training_features, training_targets):
  return [
      (n_estimators, evaluate_parameters(n_estimators, training_features, training_targets))
      for n_estimators in estimators_range
  ]

#%%
training_features, testing_features, training_targets, testing_targets = load_dataset()
estimators_range = (5, 10, 20, 50, 100)
cross_validation(estimators_range, training_features, training_targets)
