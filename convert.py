import pandas as pd
df = pd.read_parquet('wiki_nlp.parquet')
df.to_csv('wiki1.csv')