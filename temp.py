from preprocessors.naive_word2vec import NaiveWord2VecPreprocessor


dir = '~/Téléchargements'
preprocessor = NaiveWord2VecPreprocessor(dir)
preprocessor.load_dataset(is_training=True)

#%%