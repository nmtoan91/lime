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
import argparse

from MTools import LoadDataSet, SelectClassifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data",type=str,  default='covtype', help='data')
    parser.add_argument("-i", "--index",type=int,  default=0, help='index')
    parser.add_argument("-l", "--label",type=int,  default=None, help='index')
    parser.add_argument("-m", "--method",type=str,  default='KNeighborsClassifier', help='method')

    args = parser.parse_args()

    data = LoadDataSet(args.data)
    X = data.data
    y = data.target
    
    feature_names = data.feature_names

    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_names], df['target'], test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    classifier = SelectClassifier(args.method)
    classifier.fit(X_train, y_train)

    
    instance = X_test.iloc[args.index]
    if args.label == None: args.label = classifier.predict([instance])[0]
    

    from toanstt.TabularExplainer import TabularExplainer
    explainer = TabularExplainer(np.array(X_train),feature_names,class_names=feature_names,mode='classification',
                            training_labels=None)

    

    # Generate explanation for the instance
    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=classifier.predict_proba, labels=(args.label,)
    )

    print('Prediction probability:', classifier.predict_proba([instance])[0])
    print('True class:', y_test.iloc[args.index])
    print('Explanation:', exp.as_list(label=args.label))

    fig = exp.as_pyplot_figure(label=args.label)
    plt.show()
    input()
    asd=123

