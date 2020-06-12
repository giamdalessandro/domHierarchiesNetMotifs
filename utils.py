import os
import pandas as pd
import configparser


# Reading the number of individuas from the config file
def individuals_number(cfg_file):
	with open(cfg_file) as f:
		file_content = '[default]\n' + f.read()
	f.close()

	cp = configparser.ConfigParser()
	cp.read_string(file_content)

	return int(cp['default']['NumFemales'][0]) + int(cp['default']['NumMales'][0]) 

# unify output files of different runs output files
def unify_runs_output(f_name):
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
    frame.to_csv(f_name, sep=';', na_rep='NA', decimal='.')
