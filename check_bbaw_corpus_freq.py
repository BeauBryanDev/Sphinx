import pandas as pd
from collections import Counter

df = pd.read_parquet('artifacts/bbaw_clean.parquet')
all_codes = ' '.join(df['gardiner_seq'].tolist()).split()
freq = Counter(all_codes)

for code in ['U21', 'E10','C2','C12','P3','N23','M4', 'F34', 'O29', 'M20', 'M16', 'E7', 'S19']:
    print(f"{code:6s} aparece {freq.get(code, 0):>5} veces en el corpus BBAW")
