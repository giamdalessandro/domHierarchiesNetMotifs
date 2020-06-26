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
	# compute win proportion matrix
	n_ind = len(contest_mat[0])
	P_mat = np.zeros((n_ind,n_ind))    # win proportion matrix
	w = []                             
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

	# compute l term to calculate David's Score
	l = []
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

	#plotHierarchy(x,y,ind_ids,intercept,slope)

	return abs(slope)


# plot hierarchy ranking and its steepness
def plotHierarchy(x, y, ind_ids, intercept, slope):
	n_ind = len(ind_ids)
	fig, ax = plt.subplots()
	ax.plot(x, y, 'o-', label='NormDS')
	ax.plot(x, intercept + slope*x, '-', label='fitted line')

	ax.set_yticks(np.arange(0, round(y[0])+2, 2))
	ax.set_xticks(np.arange(0, n_ind, 1))
	ax.set_xticklabels(ind_ids)
	ax.set(xlabel='individuals id', ylabel='normalized DS',
		   title='hierarchy steepness')

	plt.legend()
	plt.show()


# plot the dominance network as a digraph
def plotNetwork(net_graph):
	nx.draw(net_graph, with_labels=True, font_weight='bold')
	plt.show()
	return
	

# plot hierarchy steepnees w.r.t group sizes, for mild and fierce species
def plotsAggrIntensity():
	sizes = [8, 12, 18, 24, 30, 36, 42, 48]
	steep_mild = [0.2566, 0.3419, 0.3306, 0.2964, 0.3189, 0.3513, 0.3001, 0.2761]
	steep_fierce = [0.4298, 0.3478, 0.4544, 0.3787, 0.4979, 0.4144, 0.4905, 0.4487]

	fig, ax = plt.subplots()
	ax.plot(sizes, steep_mild, 'o-', label='mild')
	ax.plot(sizes, steep_fierce, 'o-', label='fierce')

	ax.set_yticks(np.arange(0, 1, 0.2))
	ax.set_xticks(sizes)
	#ax.set_xticklabels(ind_ids)
	ax.set(xlabel='group size', ylabel='hierarchy steepness',
		   title='Intesity of aggression')

	plt.legend()
	plt.show()


# plot occurences of triadic patterns for different group sizes
def plotstTriadicPatterns(aggr):   
	# aggr: 'mild' or 'fierce'
	sizes = [8, 12, 18, 24, 30, 36, 42, 48]
	if aggr == 'mild':
		pat_8 =  [0, 0,	0,	0,	0,	0,	56]
		pat_12 = [0, 0,	0,	0,	0,	7, 193]
		pat_18 = [0, 0,	0,	0,	0,	34,	750]
		pat_24 = [0, 0,	0,	0,	0,	147, 1426]
		pat_30 = [0, 0,	0,	0,	0,	323, 2901] 
		pat_36 = [0, 0,	22,	22,	45,	495, 4728]
		pat_42 = [0, 14,	305, 231, 119, 1153, 5937] 
		pat_48 = [5, 144, 917, 565, 498, 1655, 7794]
	else:
		pat_8 =  [0, 0,	0,	0,	0,	0,	56]
		pat_12 = [0, 0,	0,	0,	0,	10, 210]
		pat_18 = [0, 0,	0,	0,	0,	39, 730]
		pat_24 = [0, 0,	0,	0,	0,	196, 1699]
		pat_30 = [0, 0,	0,	0,	0,	321, 3102] 
		pat_36 = [0, 1,	61,	36,	43,	678, 4721]
		pat_42 = [0, 17, 289, 161, 205, 962, 7512]
		pat_48 = [3, 131, 745, 628, 692, 1166, 9230]
	
	fig, ax = plt.subplots()
	ax.plot(np.arange(0, 7, 1),  pat_8, 'o-', label='8')
	ax.plot(np.arange(0, 7, 1), pat_12, 'o-', label='12')
	ax.plot(np.arange(0, 7, 1), pat_18, 'o-', label='18')
	ax.plot(np.arange(0, 7, 1), pat_24, 'o-', label='24')
	ax.plot(np.arange(0, 7, 1), pat_30, 'o-', label='30')
	ax.plot(np.arange(0, 7, 1), pat_36, 'o-', label='36')
	ax.plot(np.arange(0, 7, 1), pat_42, 'o-', label='42')
	ax.plot(np.arange(0, 7, 1), pat_48, 'o-', label='48')

	ax.set_yticks(np.arange(0, pat_48[-1], 500))
	ax.set_xticks(np.arange(0, len(pat_8), 1))
	ax.set_xticklabels(['Null','Single-edge','Double-dominant','Double-subordinate','Pass-along','Transitive','Cycle'])
	ax.set(ylabel='pattern occurrences',
		   title='Triadic patterns - {} species'.format(aggr))

	plt.legend()
	plt.show()


# plot occurences of triadic patterns for different fleeing distances
def plotFleeDist():
	cols = ['group-size','flee-dist','aggr-intensity','steepness','Null','Single-edge',
	        'Double-dominant','Double-subordinate','Pass-along','Transitive','Cycle']
	data = pd.read_csv('results.csv', usecols=cols, sep=',')

	m_data = data.query('`group-size` == 24 & `aggr-intensity` == "mild"')
	f_data = data.query('`group-size` == 24 & `aggr-intensity` == "fierce"')

	fig, ax = plt.subplots()
	for idx, row in m_data.iterrows():
		ax.plot(np.arange(0, 7, 1),  row[['Null','Single-edge','Double-dominant',
		                        'Double-subordinate','Pass-along','Transitive','Cycle']], 'o-', label=row['flee-dist'])

	ax.set_yticks(np.arange(0, 1800, 200))
	ax.set_xticks(np.arange(0, 8, 1))
	ax.set_xticklabels(['Null','Single-edge','Double-dominant','Double-subordinate','Pass-along','Transitive','Cycle'])
	ax.set(ylabel='pattern occurrences',
		   title='Triadic patterns - fleeing distance (mild)')

	plt.legend()
	plt.show()