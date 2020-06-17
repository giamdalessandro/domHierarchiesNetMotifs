import os
import pandas as pd
import configparser


# Reading the number of individuas from the config file
def set_domWorld_cfg(filename,  params):
	new_cfg = []
	with open(filename, 'r') as f:
		for row in f.readlines():
			for k in params.keys():
				if k in row:
					row = k + ' = {}'.format(params[k][0]) + '\t' + row[row.find('#'):]
					break

			new_cfg.append(row)
		f.close()

	with open(filename, 'w') as f:
		for item in new_cfg:
			f.write(item)
		f.close()

	return 

# unify output files of different runs output filescin f_name file
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


# Reading the number of individuas from the config file
def individuals_number(cfg_file):
	with open(cfg_file) as f:
		file_content = '[default]\n' + f.read()
	f.close()

	cp = configparser.ConfigParser()
	cp.read_string(file_content)

	return int(cp['default']['NumFemales'][0]) + int(cp['default']['NumMales'][0])