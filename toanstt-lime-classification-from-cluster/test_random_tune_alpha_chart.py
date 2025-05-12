from main import ExecuteExplain
import random
import argparse
import copy
from itertools import combinations
import json
import os
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)
basename =os.path.basename(__file__)

jsonFiles = ['test_random_tune_alpha.py_breast_cancer.json', 'test_random_tune_alpha.py_covtype.json', 'test_random_tune_alpha.py_iris.json']
dataset = ['breast_cancer','covtype','iris']
fig = plt.figure()

for  i in range(len(jsonFiles)):
    jsonFile = jsonFiles[i]
    jsonFile = dirname + "/Results/" + jsonFile
    with open(jsonFile, 'r') as file:
        data_ = json.load(file)
    x = []
    y =[]
    for k,v in data_.items():
        x.append(float(k))
        y.append(v)
    plt.plot(x, y, label=dataset[i])


plt.xlabel(r'$\alpha$')
plt.ylabel("Feature coefficient ratio")
plt.legend()
plt.tight_layout()

plt.show()
    



outputName = dirname + "/Figures/" + basename + ".pdf"

fig.savefig(outputName)
        

  






