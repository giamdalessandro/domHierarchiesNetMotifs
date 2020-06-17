import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from utils import set_domWorld_cfg, unify_runs_output

params = {
	'InitialDensity' :  [1.7],
	'NumFemales' : [6],
	'NumMales' : [6],
	'Rating.Dom.female.Intensity' : [0.1],      # eg: 0.1  desp: 0.8
	'Rating.Dom.male.Intensity' : [0.2],        # eg: 0.2  desp: 1.0
	'female.PersSpace' : [2.0],
	'female.FleeDist' : [2.0],
	'male.PersSpace' : [2.0],
	'male.FleeDist' : [2.0]
}

set_domWorld_cfg('Config_domHierarchies.ini',params)
os.system('DomWorld_Legacy.exe .\Config_domhierarchies.ini')


# Reading data from DomWorld output 
unify_runs_output('FILENAME.csv')  # unify different runs output files
data = pd.read_csv('FILENAME.csv', usecols=['run','period','actor.id','actor.sex','actor.behavior','actor.score',
                                              'receiver.id','receiver.sex','receiver.behavior','receiver.score'], sep=';')

# selecting the rows representing dominance interactions
df_attacks = data.query('`actor.behavior` == "Fight" | `actor.behavior` == "Flee"')
#print(df_attacks)


N_IND = int(params['NumFemales'][0]) + int(params['NumMales'][0])
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
		
		else:                                          # deuce
			dom_mat[r][c] = 0.5
			dom_mat[c][r] = 0.5

print('\nDominance matrix:')
print(dom_mat)



# triadic census of the network
net_G = nx.from_numpy_matrix(dom_mat, create_using=nx.DiGraph)
census = nx.triadic_census(net_G)

triad_cfg = {
	'003' : 'Null',
	'012' : 'Single-edge',
	'021D': 'Double-dominant',
	'021U': 'Double-subordinate',
	'021C': 'Pass-along',
	'030T': 'Transitive',
	'030C': 'Cycle'
}

print('\nNetwork Triadic Census:')
f_census = {}
f_census['group-size'] = [N_IND]
f_census['flee-dist'] = params['female.FleeDist']
f_census['aggr-intensity'] = [('mild' if params['Rating.Dom.female.Intensity'][0] == 0.1 else 'fierce')]

for k,v in sorted(census.items()):
	if k in triad_cfg:
		f_census[triad_cfg[k]] = v
		print('  ' + triad_cfg[k] + ': ' + str(v))
	

#nx.draw(net_G, with_labels=True, font_weight='bold')
#plt.show()

res = pd.DataFrame.from_dict(f_census, orient='columns')
print(res)
res.to_csv('results.csv', mode='a', sep=';', header=False)