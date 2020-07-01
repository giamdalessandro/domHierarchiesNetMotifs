import os
import platform
import configparser

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from graph_plot import plotHierarchy

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


# Compute hierarchy steepness as linear fit of the ranked David's scores
def hierarchySteepness(d_score):
	# normalize the DS to ensure that steepness varies between 0 and 1  
	NormDS = []
	DS = d_score['DS']
	n_ind = len(DS)
	for i in range(n_ind):
		NormDS_i = (DS[i] + (n_ind*(n_ind-1))/2)/n_ind
		NormDS.append(NormDS_i)

	tmp = NormDS.copy()
	NormDS.sort(reverse=True)
	ind_ids = []
	for pos in range(n_ind):
		for i in range(n_ind):
			if NormDS[pos] == tmp[i]:
				ind_ids.append('{}'.format(i+1))

	ticks = [i for i in range(0,n_ind)]
	x = np.array(ticks)
	y = np.array(NormDS)

	slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
	print('\nhierarchy steepness: %.4f' % slope)
	print('intercept: %.4f' % intercept)
	print('r_value: %.4f' % r_value)

	#plotHierarchy(x,y,ind_ids,intercept,slope)

	return abs(slope)


# maps nx.triadic_census() subgraph codes to explicit to 
# triadic patterns names
def mapTriadCodes(census, rand_census, triad_cfg):
	real = {}
	for k,v in sorted(census.items()):
		if k in triad_cfg:
			real[triad_cfg[k]] = v

	random = {}
	for rc in rand_census:
		for k,v in sorted(rc.items()):
			if k in triad_cfg:
				if triad_cfg[k] not in random.keys():
					random[triad_cfg[k]] = []
				
				random[triad_cfg[k]].append(v)
	
	return (real, random)


# compute the significance profile of the patterns mapped in 
# triad_cfg, inside directed graph G
def triadSignificanceProfile(G, triad_cfg):
	# G: directed graph representing the network 
	# triads_cfg: dict mapping interesting triadic patterns 
	#       codes, as in nx.triadic_census(), with explicit names. 
	# 		(e.g. triad_cfg = {'003' : 'Null', '012' : 'Single-edge'})
	census = nx.triadic_census(G)
	in_degree_sequence = [d for n, d in G.in_degree()]  # in degree sequence
	out_degree_sequence = [d for n, d in G.out_degree()]  # out degree sequence
	print("In_degree sequence %s" % in_degree_sequence)
	print("Out_degree sequence %s" % out_degree_sequence)

	random_nets_census = []
	for i in range(100):
		rand_G = nx.directed_configuration_model(in_degree_sequence, out_degree_sequence, create_using=nx.DiGraph, seed=i)
		random_nets_census.append(nx.triadic_census(rand_G))

	real_census, random_census = mapTriadCodes(census,random_nets_census,triad_cfg)
	#print(real_census)
	#print(random_census)

	z_score = []
	for p in real_census.keys():
		N_real_p = real_census[p]
		N_rand_p = np.mean(random_census[p])
		std = np.std(random_census[p])

		z_p =  ((N_real_p - N_rand_p)/std if std != 0 else 0)
		z_score.append(z_p)

	SP = []
	for i in range(len(z_score)):
		z_norm = np.linalg.norm(z_score)
		norm_z_score = (z_score[i]/z_norm if z_norm != 0 else z_score[i])
		SP.append(norm_z_score)

	print(SP)
	return SP