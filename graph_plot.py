import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from networkx import draw


# plot the dominance network as a digraph
def plotNetwork(net_graph):
	draw(net_graph, with_labels=True, font_weight='bold')
	plt.show()
	return


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
	

# plot hierarchy steepnees w.r.t group sizes, for mild and fierce species
def plotAggrIntensity():
	cols = ['group-size','flee-dist','aggr-intensity','steepness']
	data = pd.read_csv('results.csv', usecols=cols, sep=',')

	mild_data = data.query('`flee-dist` == 2 & `aggr-intensity` == "mild"')
	fierce_data = data.query('`flee-dist` == 2 & `aggr-intensity` == "fierce"')
	
	sizes = [8, 12, 18, 24, 30, 36, 42, 48]
	steep_mild = np.array(mild_data['steepness'], dtype=float)
	steep_fierce = np.array(fierce_data['steepness'], dtype=float)

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
def plotTriadicPatterns(aggr):   
	# aggr: 'mild' or 'fierce'
	patterns = ['Null','Single-edge','Double-dominant','Double-subordinate','Pass-along','Transitive','Cycle']
	cols = ['group-size','flee-dist','aggr-intensity','Null','Single-edge',
	        'Double-dominant','Double-subordinate','Pass-along','Transitive','Cycle']

	data = pd.read_csv('results.csv', usecols=cols, sep=',')
	pat_data = data.query('`flee-dist` == 2 & `aggr-intensity` == "{}"'.format(aggr))
	
	fig, ax = plt.subplots()
	for idx, row in pat_data.iterrows():
		ax.plot(np.arange(0, 7, 1),  row[patterns], 'o-', label=row['group-size'])

	ytick = np.amax(np.array(pat_data[['Transitive']])) + 200
	ax.set_yticks(np.arange(0, ytick, 400))
	ax.set_xticks(np.arange(0, 8, 1))
	ax.set_xticklabels(patterns)
	ax.set(ylabel='pattern occurrences',
		   title='Triadic patterns - {} species'.format(aggr))

	plt.legend()
	plt.show()


# plot occurences of triadic patterns for different fleeing distances
def plotFleeDist(size,aggr):
	# size: different flee distances have been tested only on 24 or 36 group
	# aggr: 'mild' or 'fierce'
	patterns = ['Null','Single-edge','Double-dominant','Double-subordinate','Pass-along','Transitive','Cycle']
	cols = ['group-size','flee-dist','aggr-intensity','steepness','Null','Single-edge',
	        'Double-dominant','Double-subordinate','Pass-along','Transitive','Cycle']
	data = pd.read_csv('results.csv', usecols=cols, sep=',')

	flee_data = data.query('`group-size` == {} & `aggr-intensity` == "{}"'.format(size,aggr))
	
	fig, ax = plt.subplots()
	for idx, row in flee_data.iterrows():
		ax.plot(np.arange(0, 7, 1),  row[patterns], 'o-', label=row['flee-dist'])

	ytick = (1800 if size == 24 else 5000)
	ax.set_yticks(np.arange(0, ytick, 200))
	ax.set_xticks(np.arange(0, 8, 1))
	ax.set_xticklabels(patterns)
	ax.set(ylabel='pattern occurrences',
		   title='Triadic patterns - fleeing distance ({})'.format(aggr))

	plt.legend()
	plt.show()



def autolabel(axes, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        axes.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# plot occurences of triadic patterns for different group sizes
def plotFreqPatterns():   
	# aggr: 'mild' or 'fierce'
	patterns = ['Null','Single-edge','Double-dominant','Double-subordinate','Pass-along','Transitive','Cycle']
	cols = ['group-size','flee-dist','aggr-intensity','Null','Single-edge',
	        'Double-dominant','Double-subordinate','Pass-along','Transitive','Cycle']

	data = pd.read_csv('results.csv', usecols=cols, sep=',')
	mild_data = data.query('`flee-dist` == 2 & `aggr-intensity` == "mild"')
	fierce_data = data.query('`flee-dist` == 2 & `aggr-intensity` == "fierce"')

	sizes = ['8', '12', '18', '24', '30', '36', '42', '48']
	tot_fierce = []
	for idx, row in fierce_data.iterrows():
		tot_fierce.append(sum(row[patterns]))

	freq_mild = {}
	freq_fierce = {}
	width = 0.12  # the width of the bars
	for p in patterns:
		freq_m = []
		freq_f = []		
		for idx, row in mild_data.iterrows():      # for each group size
			tot_mild = sum(row[patterns])
			freq_m.append(round(row[p]/tot_mild, 2))
			freq_f.append(round(fierce_data[p][idx+8]/tot_fierce[idx], 2))
	
		freq_mild[p] = freq_m
		freq_fierce[p] = freq_f 
		#rec1 = ax.bar(np.arange(0, len(sizes), 1) - width/2, freq_mild, width, label='mild')
		#rec2 = ax.bar(np.arange(0, len(sizes), 1) + width/2, freq_fierce, width, label='fierce')

	fig, ax = plt.subplots()
	plt.grid(True, axis='y', linestyle='-.', linewidth=0.5)
	for s in range(len(sizes)):
		to_plot = []
		for p in range(len(patterns)):
			to_plot.append(freq_fierce[patterns[p]][s])

		step = [i + s*0.12 - 0.48 for i in range(len(patterns))]
		ax.bar(step, to_plot, width, label=sizes[s])

	ax.set_yticks(np.arange(0, 1.1, 0.1))
	ax.set_xticks(np.arange(0, len(patterns), 1))
	ax.set_xticklabels(patterns)
	ax.set(ylabel='frequency', title='Triadic pattern frequencies - fierce species')

	plt.legend(title='group size')
	plt.show()