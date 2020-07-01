import os
import json
import platform

CONFIG_FILE = 'Config_domHierarchies.ini'
OUTPUT_FILE = 'FILENAME.csv'

for size in [4, 6, 9, 12, 15, 18, 21, 24]:
    params = {
        "Periods" : 260,
        "firstDataPeriod" : 200,
        "InitialDensity" :  1.7,
        "NumFemales" : size,                         # 4, 6, 9, 12, 15, 18, 21, 24
        "NumMales" : size,                           # 4, 6, 9, 12, 15, 18, 21, 24
        "Rating.Dom.female.Intensity" : 0.8,       # eg: 0.1  desp: 0.8
        "Rating.Dom.male.Intensity" : 1.0,         # eg: 0.2  desp: 1.0
        "female.PersSpace" : 2.0,
        "female.FleeDist" : 2.0,
        "male.PersSpace" : 2.0,
        "male.FleeDist" : 2.0
    }

    if platform.system() == 'Windows':
        os.system("python net_analysis.py {} {} '{}'".format(CONFIG_FILE,OUTPUT_FILE,json.dumps(params)))
    elif platform.system() == 'Linux':
        os.system("python3 net_analysis.py {} {} '{}'".format(CONFIG_FILE,OUTPUT_FILE,json.dumps(params)))
    else:
        print('System not supported')