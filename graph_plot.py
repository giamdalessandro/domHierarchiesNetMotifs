import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from networkx import draw


def plotNetwork(net_graph):
	"""
	Plots the dominance network as a digraph.
	"""
	draw(net_graph, with_labels=True, font_weight='bold')
	plt.show()
	return


def plotHierarchy(x, y, ind_ids, intercept, slope):
	"""
	Plots hierarchy ranking and its steepness.
	"""
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
	

def plotAggrIntensity():
	"""
	Plots hierarchy steepnees w.r.t group sizes, for mild and fierce species.
	"""
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


def plotTriadicPatterns(aggr):
	"""
	Plot occurences of triadic patterns for different group sizes.
		- aggr : 'mild' or 'fierce'
	"""   
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


def plotFleeDist(size, aggr):
	"""
	Plots occurences of triadic patterns for different fleeing distances.
		- size : different flee distances have been tested only on 24 or 36 group size;
		- aggr : 'mild' or 'fierce'.
	"""
	patterns = ['Null','Single-edge','Double-dominant','Double-subordinate','Pass-along','Transitive','Cycle']
	cols = ['group-size','flee-dist','aggr-intensity','steepness','Null','Single-edge',
	        'Double-dominant','Double-subordinate','Pass-along','Transitive','Cycle']
	data = pd.read_csv('results.csv', usecols=cols, sep=',')

	flee_data = data.query('`group-size` == {} & `aggr-intensity` == "{}"'.format(size,aggr))
	prop = {}
	for p in patterns:
		prop_p = []
		for idx, row in flee_data.iterrows():      # for each group size
			tot_mild = sum(row[patterns])
			prop_p.append(round(row[p]/tot_mild, 2))
			
		prop[p] = prop_p

	fig, ax = plt.subplots()
	#for idx, row in flee_data.iterrows():
	#	ax.plot(np.arange(0, 7, 1),  row[patterns], 'o-', label=row['flee-dist'])

	for fd in range(5):
		plot = []
		for p in patterns:
			plot.append(prop[p][fd])

		ax.plot(np.arange(0, 7, 1),  plot, 'o-', label=str(2.0 * (fd + 1)))

	#ytick = (1800 if size == 24 else 5000)
	#ax.set_yticks(np.arange(0, ytick, 200))
	#ax.set_xticks(np.arange(0, 8, 1))
	ax.set_yticks(np.arange(0, 1.1, 0.1))
	ax.set_xticks(np.arange(0, 8, 1))
	ax.set_xticklabels(patterns)
	ax.set(ylabel='pattern proportion',
		   title='Triadic patterns - fleeing distance ({})'.format(aggr))

	plt.legend(title='fleeing dist')
	plt.show()


def _autolabel(axes, rects):
    """
	Attach a text label above each bar in *rects*, displaying its height.
	"""
    for rect in rects:
        height = rect.get_height()
        axes.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def _plotFreqPatterns():
	"""
	Plots occurences of triadic patterns for different group sizes.
		- aggr : 'mild' or 'fierce'.
	"""   
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
			to_plot.append(freq_mild[patterns[p]][s])

		step = [i + s*0.12 - 0.48 for i in range(len(patterns))]
		ax.bar(step, to_plot, width, label=sizes[s])

	ax.set_yticks(np.arange(0, 1.1, 0.1))
	ax.set_xticks(np.arange(0, len(patterns), 1))
	ax.set_xticklabels(patterns)
	ax.set(ylabel='pattern proportion', title='Triadic patterns - mild species')

	plt.legend(title='group size')
	plt.show()


def _plotSignificance(aggr):
	"""
	Plots hierarchy ranking and its steepness.
	"""
	sizes = ['8', '12', '18', '24', '30', '36', '42', '48']
	patterns = ['Null','Single-edge','Pass-along','Double-dominant','Double-subordinate','Transitive','Cycle']
	sp_m = [
		[-0.0944, -0.2326, -0.189, -0.3178, -0.3306,  0.8302, 0.0419],  # 8
		[-0.1028, -0.2673, -0.1794, -0.2757, -0.3094,  0.845, 0.0046],   # 12
		[-0.1249, -0.2799, -0.2341, -0.3026, -0.2995, 0.8057, 0.1444],  # 18
		[-0.1423, -0.3133, -0.2355, -0.2837, -0.2945, 0.7952, 0.1631],  # 24
		[-0.1519, -0.3301, -0.2502, -0.2832, -0.302,  0.7096, 0.3612],   # 30
		[-0.1528, -0.3018, -0.2889, -0.311, -0.2941,  0.6661, 0.4186],   # 36
		[-0.1536, -0.4038, -0.2718, -0.3095, -0.2865, 0.6476, 0.3771],  # 42
		[-0.1869, -0.4248, -0.2331, -0.2389, -0.2625,  0.667, 0.3993]     # 48
	]

	sp_f = [
		[-0.0949, -0.253, -0.173, -0.3001, -0.3164,   0.8384, 0.0621],    # 8
		[-0.112, -0.2819, -0.1952, -0.2727, -0.2813,  0.8326, 0.1521],   # 12
		[-0.1456, -0.2934, -0.2224, -0.2648, -0.3164, 0.7646, 0.2975],  # 18
		[-0.1305, -0.2387, -0.209, -0.2707, -0.2554,  0.7802, 0.3676],   # 24
		[-0.134, -0.3369, -0.227, -0.2839, -0.2682,   0.7402, 0.3414],    # 30
		[-0.1314, -0.3035, -0.2394, -0.262, -0.2777,   0.678, 0.4773],    # 36
		[-0.1652, -0.376, -0.2018, -0.2874, -0.2408,  0.7214, 0.3601],   # 42
		[-0.1727, -0.3873, -0.2315, -0.2838, -0.2749, 0.7043, 0.3382]   # 48
	]
	sp = (sp_m if aggr == 'mild' else sp_f)

	# to plot both mild and fierce in the same graph
	avg_sp_m = []
	avg_sp_f = []
	for c in range(7):
		tmp_m = []
		tmp_f = []
		for r in range(8):
			tmp_m.append(sp_m[r][c])
			tmp_f.append(sp_f[r][c])

		avg_sp_m.append(np.mean(tmp_m))
		avg_sp_f.append(np.mean(tmp_f)) 	


	fig, ax = plt.subplots()
	ax.axhline(xmax=10, color='gray', linestyle='--', linewidth=0.5)
	#for s in range(len(sizes)):
	#	ax.plot(np.arange(0, 7, 1), sp[s], 'o-', label=sizes[s])

	ax.plot(np.arange(0, 7, 1), avg_sp_m, 'o-', label='mild')
	ax.plot(np.arange(0, 7, 1), avg_sp_f, 'o-', label='fierce')

	ax.set_yticks(np.arange(-1.0, 1.1, 0.25))
	ax.set_xticks(np.arange(0, len(patterns), 1))
	ax.set_xticklabels(patterns)
	ax.set(ylabel='normalized Z-score', title='Significance Profile'.format(aggr))

	plt.legend(title='aggression intensity')
	plt.show()