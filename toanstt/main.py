from TabularExplainer import TabularExplainer
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
import lime
import lime.lime_tabular
import json

from MTools import LoadDataSet, SelectClassifier,ExtractExplnationData,CheckArgs
from sklearn.metrics import accuracy_score
from tqdm import tqdm

dirname = os.path.dirname(__file__)
basename =os.path.basename(__file__)

def ExecuteExplain(args):
    CheckArgs(args)

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

    # if True:
    #     y_predict = classifier.predict(X_test)
    #     accuracy = accuracy_score(y_predict, y_test)
    #     correct_indices = np.where(y_predict == y_test)[0]
    #     wrong_indices = np.where(y_predict != y_test)[0]
    #     pass
    instance = X_test.iloc[args.index]
    if args.label == None:
        #args.label = classifier.predict([instance])[0]
        args.label = np.argmax(classifier.predict_proba([instance])[0])
        asd=123
    asd=123


    if args.explainer == 'dst-lime':
        from toanstt.TabularExplainer import TabularExplainer
        explainer = TabularExplainer(np.array(X_train),feature_names,class_names=feature_names,
                                     mode='classification',
                                training_labels=None)
        exp = explainer.explain_instance(
            data_row=instance,
            predict_fn=classifier.predict_proba, labels=(args.label,),
            alpha = args.alpha, num_features = args.num_features
        )

    elif args.explainer == 'lime':
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=feature_names,
            class_names=feature_names,
            mode='classification',
            training_labels=None)

        exp = explainer.explain_instance(
            data_row=instance,
            predict_fn=classifier.predict_proba,
            labels=(args.label,), num_features = args.num_features)

    print('Prediction probability:', classifier.predict_proba([instance])[0])
    print('True class:', y_test.iloc[args.index])
    print('Explanation:', exp.as_list(label=args.label))

    fig = exp.as_pyplot_figure(label=args.label)

    outputName = f"{args.method}_{args.data}_i{args.index}_l{args.label}_a{args.alpha}_{args.explainer}"

    fig.savefig(dirname+"/Figures/" + outputName + ".pdf")
    results = ExtractExplnationData(exp,args.label)
    with open(dirname+"/Results/" + outputName + ".json", "w") as file:
        json.dump(results, file, indent=4)
    
    #plt.show()
    return exp,fig 


def Analyze(args):
    CheckArgs(args)

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

    y_predict = classifier.predict(X_test)
    accuracy = accuracy_score(y_predict, y_test)
    accuracy_list = list(y_predict == y_test)
    correct_indices = np.where(y_predict == y_test)[0]
    wrong_indices = np.where(y_predict != y_test)[0]
        
    
    y_test = y_test.to_numpy()
    n_class = len(np.unique(y_train))
    data = np.zeros((len(X_test),n_class,3 +  3 +1))
    explainer = TabularExplainer(np.array(X_train),feature_names,class_names=feature_names,mode='classification',training_labels=None)
    for i in tqdm(range(len(X_test))):
        instance = X_test.iloc[args.index]
        for l in range(n_class):
            exp = explainer.explain_instance(
                data_row=instance,
                predict_fn=classifier.predict_proba, labels=(l,),
                alpha = args.alpha, num_features = args.num_features
            )
            data[i,l,0] = y_test[i]
            data[i,l,1] = y_predict[i]
            data[i,l,2] = accuracy_list[i]
            data[i,l,3] = l
            data[i,l,4] =  exp.intercept_conflict[l]
            data[i,l,5] =  exp.local_pred_conflict[l]
            data[i,l,6] =  exp.score_conflict[l]
            pass
    pass
    np.save('toanstt/data/data.npy', data)

   


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument("-d", "--data",type=str,  default='covtype', help='data')
    parser.add_argument("-d", "--data",type=str,  default='breast_cancer', help='data')
    parser.add_argument("-i", "--index",type=int,  default=3, help='index')
    parser.add_argument("-l", "--label",type=int,  default=None, help='index')
    parser.add_argument("-m", "--method",type=str,  default='KNeighborsClassifier', help='method')
    parser.add_argument("-a", "--alpha",type=float,  default=0.1, help='alpha')
    parser.add_argument("-e", "--explainer",type=str,  default='dst-lime', help='alpha')
    #parser.add_argument("-e", "--explainer",type=str,  default='lime', help='alpha')
    parser.add_argument("-nf", "--num_features",type=int,  default=10, help='num_features')
    args = parser.parse_args()
    #ExecuteExplain(args)
    Analyze(args)



