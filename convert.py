import pandas as pd
df = pd.read_parquet('wiki_prompt.parquet')
df.to_csv('wiki.csv')