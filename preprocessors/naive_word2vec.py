# -*- coding: utf-8 -*-
import numpy as np
import os
import gensim

from preprocessors.base import BaseNaivePreprocessor

class NaiveWord2VecPreprocessor(BaseNaivePreprocessor):
  def __init__(self, dir):
    super(NaiveWord2VecPreprocessor, self).__init__(dir)
    model_path = os.path.join(self.dir, 'GoogleNews-vectors-negative300.bin')
    self.model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

  def try_w2v_representation(self, word):
    try:
        if word[-1] == '?':
            word = word[:-1]
        return self.model[word]
    except:
        return None

  def sentence2vec(self, sentence):
    words = str(sentence).split(' ')
    features_list = [
        self.try_w2v_representation(word)
        for word in words
        if self.try_w2v_representation(word) is not None
    ]
    return np.mean(features_list, axis=0)

  def transform_sample(self, sample, is_training):
    question1 = sample.question1
    question2 = sample.question2
    
    in_vec = self.sentence2vec(question1)
    out_vec = self.sentence2vec(question2)
    
    target = sample.is_duplicate if is_training else None
    
    if in_vec.size < 2 or out_vec.size < 2:
        return None
    else:
        return np.hstack([in_vec, out_vec]), target
