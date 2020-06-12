import pandas as pd
import os

out_files = []
for f in os.listdir('.'):
    if 'output' and '.csv' in f and 'F' not in f:
        out_files.append(f)

print(out_files)

l = []
for filename in out_files:
    df = pd.read_csv(filename, sep='\t', index_col=None, header=0)
    l.append(df)

frame = pd.concat(l, axis=0, ignore_index=True)
frame.to_csv('FILENAME.csv', sep=';', na_rep='NA', decimal='.')
