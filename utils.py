import os
import platform
import numpy as np
import pandas as pd
import configparser
import matplotlib.pyplot as plt
import networkx as nx


# setting domWorld config file
def set_domWorld_cfg(filename,  params):
	new_cfg = []
	with open(filename, 'r') as f:
		for row in f.readlines():
			for k in params.keys():
				if k in row:
					row = k + ' = {}'.format(params[k]) + '\t' + row[row.find('#'):]
					break

			new_cfg.append(row)
		f.close()

	with open(filename, 'w') as f:
		for item in new_cfg:
			f.write(item)
		f.close()

	return 


# run domWorld model
def run_domWorld_model(cfg_file):
	if platform.system() == 'Windows':
		os.system('DomWorld_Legacy.exe .\{}'.format(cfg_file))
	elif platform.system() == 'Linux':
		os.system('wine DomWorld_Legacy.exe ./{}'.format(cfg_file))
	else:
		print('System not supported')

	return	


# unify output files of different runs output filescin f_name file
def unify_runs_output(f_name):
	out_files = []
	for f in os.listdir('.'):
		if ('output' in f) and ('.csv' in f) and ('F' not in f):
			out_files.append(f)

	print(out_files)

	l = []
	for filename in out_files:
		df = pd.read_csv(filename, sep='\t', index_col=None, header=0)
		l.append(df)

	frame = pd.concat(l, axis=0, ignore_index=True)
	frame.to_csv(f_name, sep=';', na_rep='NA', decimal='.')

	return


# calculate the David's Score given the contest matrix.
# The David's score for an individual i is given by:
#	DS = w + w_2 - l - l_2 
def davids_score(contest_mat):
	n_ind = len(contest_mat[0])
	P_mat = np.zeros((n_ind,n_ind))    # win proportion matrix
	w = []                             # w values list
	l = []                             # l values list
	for i in range(n_ind):
		P_list = [] 
		for j in range(n_ind):
			if i == j:
				P_list.append(0)
				continue
			else:
				a_ij = contest_mat[i][j]                      # no. times i defeat j
				n_ij = a_ij + contest_mat[j][i]               # no. interactions between i and j
				P_ij = (a_ij/n_ij if n_ij != 0 else 0)        # proportion of wins by i with j
				P_mat[i][j] = P_ij
				P_list.append(P_ij)

		w_i = sum(P_list)
		l_i = sum([1 - P_list[idx] for idx in range(n_ind) if idx != i])
		w.append(w_i)
		l.append(l_i)

	w_2 = []     # w2 values list
	l_2 = []     # l2 values list
	DS = []      # David's scores
	for i in range(n_ind):	
		for j in range(n_ind):	
			if j != i:
				w_2_i = P_mat[i][j]*w[j]
				l_2_i = P_mat[i][j]*l[j] 
				w_2.append(w_2_i)
				l_2.append(l_2_i)

		DS_i = w[i] + w_2[i] - l[i] - l_2[i]
		DS.append(DS_i) 

	res = {}
	res['w'] = w
	res['w2'] = w_2
	res['l'] = l
	res['l2'] = l_2
	res['DS'] = DS
	return res


# Reading the number of individuas from the config file
def individuals_number(cfg_file):
	with open(cfg_file) as f:
		file_content = '[default]\n' + f.read()
	f.close()

	cp = configparser.ConfigParser()
	cp.read_string(file_content)

	return int(cp['default']['NumFemales']) + int(cp['default']['NumMales'])


# plot the dominance network as a graph
def plot_network(net_graph):
	nx.draw(net_graph, with_labels=True, font_weight='bold')
	plt.show()
	return