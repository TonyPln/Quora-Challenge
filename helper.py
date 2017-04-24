import pandas as pd


def separate_faulty_entries(features, ids):
  test_set = list(zip(features, ids))
  faulty_entries = list(filter(lambda line: line[0] is None, test_set))
  good_entries = list(filter(lambda line: line[0] is not None, test_set))
  
  return faulty_entries, good_entries

def fill_faulty_results(faulty_entries):
  faulty_ids = list(map(lambda line: line[1], faulty_entries))
  faulty_entries_result = pd.DataFrame({
    'test_id': faulty_ids,
    'is_duplicate': [0.37] * len(faulty_ids)
  })
  return faulty_entries_result

def chunks(df, size):
  for i in range(0, len(df), size):
    yield df.iloc[i:min(i + size,len(df))]

working_dir = '~/kaggle/quora/data'