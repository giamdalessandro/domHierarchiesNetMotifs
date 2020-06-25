import os
import json
import platform

CONFIG_FILE = 'Config_domHierarchies.ini'
OUTPUT_FILE = 'FILENAME.csv'

for f_dist in range(2,11,2):
    params = {
        "Periods" : 260,
        "firstDataPeriod" : 200,
        "InitialDensity" :  1.7,
        "NumFemales" : 12,                         # 4, 6, 9, 12, 15, 18, 21, 24
        "NumMales" : 12,                           # 4, 6, 9, 12, 15, 18, 21, 24
        "Rating.Dom.female.Intensity" : 0.1,       # eg: 0.1  desp: 0.8
        "Rating.Dom.male.Intensity" : 0.2,         # eg: 0.2  desp: 1.0
        "female.PersSpace" : 2.0,
        "female.FleeDist" : f_dist,
        "male.PersSpace" : 2.0,
        "male.FleeDist" : f_dist
    }

    if platform.system() == 'Windows':
        os.system("python net_analysis.py {} {} '{}'".format(CONFIG_FILE,OUTPUT_FILE,json.dumps(params)))
    elif platform.system() == 'Linux':
        os.system("python3 net_analysis.py {} {} '{}'".format(CONFIG_FILE,OUTPUT_FILE,json.dumps(params)))
    else:
        print('System not supported')