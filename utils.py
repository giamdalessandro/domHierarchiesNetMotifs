import os
import platform
import configparser

from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


# setting domWorld config file
def setDomWorldCfg(filename, params):
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
def runDomWorldModel(cfg_file):
	if platform.system() == 'Windows':
		os.system('DomWorld_Legacy.exe .\{}'.format(cfg_file))
	elif platform.system() == 'Linux':
		os.system('wine DomWorld_Legacy.exe ./{}'.format(cfg_file))
	else:
		print('System not supported')

	return	


# unify output files of different runs output filescin f_name file
def unifyRunsOutput(f_name):
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


# Reading the number of individuas from the config file
def individualsNumber(cfg_file):
	with open(cfg_file) as f:
		file_content = '[default]\n' + f.read()
	f.close()

	cp = configparser.ConfigParser()
	cp.read_string(file_content)

	return int(cp['default']['NumFemales']) + int(cp['default']['NumMales'])


# calculate the David's Score given the contest matrix.
# The David's score for an individual i is given by:
#	DS = w + w_2 - l - l_2 
def davidsScore(contest_mat):
	# compute win/fights ratio
	n_ind = len(contest_mat[0])
	P_mat = np.zeros((n_ind,n_ind))    # win proportion matrix
	w = []                             # win ratio list
	for i in range(n_ind):
		P_list = [] 
		for j in range(n_ind):
			if i == j:
				P_list.append(0)
				continue
			else:
				a_ij = contest_mat[i][j]                      # no. times i defeat j
				n_ij = a_ij + contest_mat[j][i]               # no. interactions between i and j
				P_ij = (a_ij/n_ij if n_ij != 0 else 0)        # proportion of wins by i over j
				P_mat[i][j] = P_ij
				P_list.append(P_ij)

		w_i = sum(P_list)              # i win rate 
		w.append(w_i)

	# compute loss/fights ratio
	l = []                             # loss ratio list
	for j in range(n_ind):
		l_list = []
		for i in range(n_ind):
			l_list.append(P_mat[i][j])
		
		l_i = sum(l_list) 
		l.append(l_i)

	# compute term w_2 and l_2 to calculate David's Score
	DS = []      # David's scores
	w_2 = []     # w_2 values list
	l_2 = []     # l_2 values list
	for i in range(n_ind):	
		w_2_i = []     
		l_2_i = []     	
		for j in range(n_ind):	
			if i == j:
				w_2_i.append(0)
				l_2_i.append(0)
			else:
				w_2_ij = P_mat[i][j]*w[j]
				l_2_ij = P_mat[j][i]*l[j] 
				w_2_i.append(w_2_ij)
				l_2_i.append(l_2_ij)

		w_2.append(sum(w_2_i))
		l_2.append(sum(l_2_i))

		DS_i = w[i] + w_2[i] - l[i] - l_2[i]
		DS.append(DS_i) 

	d_score = {}
	d_score['w'] = w
	d_score['w2'] = w_2
	d_score['l'] = l
	d_score['l2'] = l_2
	d_score['DS'] = DS
	return d_score


# Compute hierarchy steepness
def hierarchySteepness(d_score):
	# normalize the DS to ensure that steepness values 
	# varies between 0 and 1  
	NormDS = []
	DS = d_score['DS']
	n_ind = len(DS)
	for i in range(n_ind):
		NormDS_i = (DS[i] + (n_ind*(n_ind-1))/2)/n_ind
		NormDS.append(NormDS_i)

	#print(NormDS)
	tmp = NormDS.copy()
	NormDS.sort(reverse=True)
	ind_ids = []
	for pos in range(n_ind):
		for i in range(n_ind):
			if NormDS[pos] == tmp[i]:
				ind_ids.append('{}'.format(i+1))

	ticks = [i for i in range(0,n_ind)]
	x = np.array(ticks) #.reshape((-1, 1))
	y = np.array(NormDS)

	slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
	print('\nhierarchy steepness: %.4f' % slope)
	print('intercept: %.4f' % intercept)
	print('r_value: %.4f' % r_value)

	plotHierarchy(x,y,ind_ids,intercept,slope)

	return abs(slope)


# plot hierarchy ranking and its steepness
def plotHierarchy(x, y, ind_ids, intercept, slope):
	n_ind = len(ind_ids)
	fig, ax = plt.subplots()
	ax.plot(x, y, 'o-')
	ax.plot(x, intercept + slope*x, '-', label='fitted line')
	ax.set_xticks(np.arange(0, n_ind, 1))
	ax.set_xticklabels(ind_ids)
	ax.set(xlabel='individuals id', ylabel='normalized DS',
		   title='steepness')

	plt.legend()
	plt.show()


# plot the dominance network as a digraph
def plotNetwork(net_graph):
	nx.draw(net_graph, with_labels=True, font_weight='bold')
	plt.show()
	return