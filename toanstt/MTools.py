from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_covtype
import numpy as np

def LoadDataSet(datasetName):
    if datasetName == 'iris':
        return load_iris()
    if datasetName == 'breast_cancer':
        return load_breast_cancer()
    if datasetName == 'covtype':
        return fetch_covtype()
    
    print('Cannot find any dataset with dataset name', datasetName)

def SelectClassifier(classifierName):
    if classifierName == 'RandomForestClassifier':
        return RandomForestClassifier(n_estimators=100, random_state=42)
    if classifierName == 'KNeighborsClassifier':
        return KNeighborsClassifier(n_neighbors=5)
    if classifierName =='AdaBoostClassifier':
        return AdaBoostClassifier(n_estimators=50)
    if classifierName == 'MLPClassifier':
        return MLPClassifier(hidden_layer_sizes=(100,))
    if classifierName == 'DecisionTreeClassifier':
        return DecisionTreeClassifier()
    print('Cannot find any dataset with name', classifierName)


# def convert_to_serializable(data):
#     if isinstance(data, dict):
#         return {convert_to_serializable(key): convert_to_serializable(value) for key, value in data.items()}
#     elif isinstance(data, list):
#         return [convert_to_serializable(item) for item in data]
#     elif isinstance(data, (np.int64, np.int32, np.int_, np.integer)):
#         return int(data)
#     elif isinstance(data, (np.float64, np.float32, np.float_, np.floating)):
#         return float(data)
#     else:
#         return data
    

def ExtractExplnationData(exp,label):
    label = int(label)
    data={}
    data['label'] = label
    data['score'] = float(exp.score[label])
    #data['local_exp'] = exp.local_exp[label]
    #data['local_exp_conflict'] = exp.local_exp_conflict
    data['predict_proba'] = [float(i) for i in exp.predict_proba]
    r = []
    for (fid,val) in exp.local_exp[label]:
        rdata = {'fid':int(fid), 'cof': float(val)}
        
        if label in exp.local_exp_conflict:
            for (fid2,val2) in exp.local_exp_conflict[label]:
                if fid2!= fid: continue
                rdata['conflict'] = float(val2)
                break
        r.append(rdata)
    data['result'] = r
    return data
        
        
def CheckArgs(args):
    if not hasattr(args,'data'): args.data = 'iris'
    if not hasattr(args,'index'): args.index = 0
    if not hasattr(args,'label'): args.label = None
    if not hasattr(args,'method'): args.method = 'KNeighborsClassifier'
    if not hasattr(args,'alpha'): args.alpha = 0.1
    if not hasattr(args,'explainer'): args.explainer = 'dst-lime'
    if not hasattr(args,'num_features'): args.num_features = 10

    asd=123

    #return convert_to_serializable(data)