from main import ExecuteExplain
import random
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument("-d", "--data",type=str,  default='covtype', help='data')
    parser.add_argument("-d", "--data",type=str,  default='iris', help='data')
    parser.add_argument("-i", "--index",type=int,  default=3, help='index')
    parser.add_argument("-l", "--label",type=int,  default=None, help='index')
    parser.add_argument("-m", "--method",type=str,  default='KNeighborsClassifier', help='method')
    parser.add_argument("-a", "--alpha",type=float,  default=0.1, help='alpha')
    parser.add_argument("-e", "--explainer",type=str,  default='dst-lime', help='alpha')
    #parser.add_argument("-e", "--explainer",type=str,  default='lime', help='alpha')
    parser.add_argument("-nf", "--num_features",type=int,  default=10, help='num_features')
    args = parser.parse_args()
    
    datas = ['iris','breast_cancer','covtype']
    methods = ['RandomForestClassifier','KNeighborsClassifier','AdaBoostClassifier','MLPClassifier', 'DecisionTreeClassifier']

    for i in range(1000):

        args.data = random.choice(datas)
        args.method = random.choice(methods)
        args.index = random.randint(0,100)
        try:
            asd=123
            args.explainer = 'lime'
            ExecuteExplain(args)
            args.explainer = 'dst-lime'
            ExecuteExplain(args)
        except:
            asd=123



