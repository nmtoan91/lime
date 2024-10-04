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