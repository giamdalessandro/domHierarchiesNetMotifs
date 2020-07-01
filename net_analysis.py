import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from graph_plot import plotNetwork
from dom_utils import setDomWorldCfg, runDomWorldModel, davidsScore, \
				  unifyRunsOutput, hierarchySteepness, triadSignificanceProfile



if len(sys.argv) > 1:
	from json import loads
	CONFIG_FILE = sys.argv[1]
	OUTPUT_FILE = sys.argv[2]
	params = loads(sys.argv[3])

else:
	CONFIG_FILE = 'Config_domHierarchies.ini'
	OUTPUT_FILE = 'FILENAME.csv'
	params = {
		"Periods" : 260,
		"firstDataPeriod" : 200,
		"InitialDensity" :  1.7,
		"NumFemales" : 4,                         # 4, 6, 9, 12, 15, 18, 21, 24
		"NumMales" : 4,                           # 4, 6, 9, 12, 15, 18, 21, 24
		"Rating.Dom.female.Intensity" : 0.1,       # eg: 0.1  desp: 0.8
		"Rating.Dom.male.Intensity" : 0.2,         # eg: 0.2  desp: 1.0
		"female.PersSpace" : 2.0,
		"female.FleeDist" : 2.0,
		"male.PersSpace" : 2.0,
		"male.FleeDist" : 2.0
	}


#setDomWorldCfg(CONFIG_FILE,params)
#runDomWorldModel(CONFIG_FILE)

# Reading data from DomWorld output 
unifyRunsOutput(OUTPUT_FILE)  # unify different runs output files
data = pd.read_csv(OUTPUT_FILE, usecols=['run','period','actor.id','actor.sex','actor.behavior','actor.score',
                                              'receiver.id','receiver.sex','receiver.behavior','receiver.score'], sep=';')

# selecting the rows representing dominance interactions
df_attacks = data.query('`actor.behavior` == "Fight" | `actor.behavior` == "Flee"')
#print(df_attacks)

N_IND = int(params['NumFemales']) + int(params['NumMales'])
dom_mat = np.zeros((N_IND,N_IND))

# Create contest matrix from raw interaction data
# counting the number of wins in each dyad:
#     dom_mat[r][c] <- n. of times r wins over c 
for idx in df_attacks.index:
	act_idx = int(df_attacks['actor.id'][idx]) - 1                   # domMatrix attacker index (row)
	recv_idx = int(df_attacks['receiver.id'][idx]) - 1               # domMatrix receiver index (col)

	if df_attacks['actor.behavior'][idx] == "Fight":                 # attacker wins
		dom_mat[act_idx][recv_idx] += 1
	elif df_attacks['actor.behavior'][idx] == "Flee":                # receiver wins
		dom_mat[recv_idx][act_idx] += 1

print('\nContest matrix:')
print(dom_mat)

'''
N_IND = 7
dom_mat = [
	[0, 0, 1, 2, 10, 63, 8], 
	[0, 0, 2, 3, 0, 88, 4], 
	[0, 0, 0, 4, 65, 84, 3],
	[0, 0, 0, 0, 0, 80, 10],
	[0, 0, 0, 0, 0, 4, 1],
	[0, 1, 5, 0, 10, 0, 6], 
	[0, 0, 0, 0, 0, 2, 0]
]'''

# Compute hierarchy ranking with the David's score measure
ds = davidsScore(dom_mat)
steep = hierarchySteepness(ds)

# create dominance matrix, dom_mat[r][c] is equal to:
#     - 1 -> r dominates c
#     - 0 -> c dominates r
#     - 0.5 -> equal number of wins
for r in range(N_IND):
	for c in range(r, N_IND):
		if r == c:
			dom_mat[r][c] = 0                          # no fight against itself
			continue

		if dom_mat[r][c] > dom_mat[c][r]:              # r wins over c
			dom_mat[r][c] = 1
			dom_mat[c][r] = 0

		elif dom_mat[r][c] < dom_mat[c][r]:            # c wins over r
			dom_mat[r][c] = 0
			dom_mat[c][r] = 1
		
		else:          								   # deuce
			tmp = (0.5 if dom_mat[r][c] != 0 else 0)
			dom_mat[r][c] = tmp
			dom_mat[c][r] = tmp

print('\nDominance matrix:')
print(dom_mat)



# triadic census of the dominance network represented as a digraph,
# individuals are the nodes, and edges their dominance relationship
triad_cfg = {
	'003' : 'Null',
	'012' : 'Single-edge',
	'021D': 'Double-dominant',
	'021U': 'Double-subordinate',
	'021C': 'Pass-along',
	'030T': 'Transitive',
	'030C': 'Cycle'
}

net_G = nx.from_numpy_matrix(dom_mat, create_using=nx.DiGraph)
census = nx.triadic_census(net_G)

sp = triadSignificanceProfile(net_G, triad_cfg)

f_census = {}
f_census['group-size'] = [N_IND]
f_census['flee-dist'] = [params['female.FleeDist']]
f_census['aggr-intensity'] = [('mild' if params['Rating.Dom.female.Intensity'] == 0.1 else 'fierce')]
f_census['steepness'] = round(steep,4)

print('\nNetwork Triadic Census:')
for k,v in sorted(census.items()):
	if k in triad_cfg:
		f_census[triad_cfg[k]] = [v]
		print('  ' + triad_cfg[k] + ': ' + str(v))
	

res = pd.DataFrame.from_dict(f_census, orient='columns')
res.to_csv('results.csv', mode='a', sep=';', header=False)

#plotNetwork(net_G)