import os
import sys
import inspect
import matplotlib.pyplot as plt
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

sys.path.insert(0, parentdir+'/lime')

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#import lime
#import lime.lime_tabular


#data = load_breast_cancer()
data = load_iris()
X = data.data
y = data.target
label =1
feature_names = data.feature_names

df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

X_train, X_test, y_train, y_test = train_test_split(
    df[feature_names], df['target'], test_size=0.2, random_state=42)

# Train a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

i = 25  # Index of the instance in the test set
instance = X_test.iloc[i]


from toanstt.TabularExplainer import TabularExplainer
explainer = TabularExplainer(np.array(X_train),feature_names,class_names=feature_names,mode='classification',
                        training_labels=None)

i = 25  # Index of the instance in the test set
instance = X_test.iloc[i]

# Generate explanation for the instance
exp = explainer.explain_instance(
    data_row=instance,
    predict_fn=rf.predict_proba, labels=(label,)
)

print('Prediction probability:', rf.predict_proba([instance])[0])
print('True class:', y_test.iloc[i])
print('Explanation:', exp.as_list(label=label))

fig = exp.as_pyplot_figure(label=label)
plt.show()
input()
asd=123
