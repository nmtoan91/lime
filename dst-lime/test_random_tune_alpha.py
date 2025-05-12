from main import ExecuteExplain
import random
import argparse
import copy
from itertools import combinations
import json
import os
dirname = os.path.dirname(__file__)
basename =os.path.basename(__file__)

def order_of_pairs_similarity(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")

    # Create all possible pairs from list1 and list2
    pairs1 = list(combinations(list1, 2))
    pairs2 = list(combinations(list2, 2))

    # Count matching pairs in the same relative order
    matching_pairs = sum(1 for pair in pairs1 if pair in pairs2)

    # Calculate percentage of matching pairs
    similarity_percentage = (matching_pairs / len(pairs1)) * 100

    return similarity_percentage
import math

def gaussian_kernel(x, y, sigma=1.0):
    return math.exp(-((x - y) ** 2) / (2 * sigma ** 2))

def dictionary_similarity(dict1, dict2, sigma=1.0, mismatch_penalty_weight=0.5):
    # Find common, only-in-dict1, and only-in-dict2 keys
    common_keys = set(dict1.keys()) & set(dict2.keys())
    only_in_dict1 = set(dict1.keys()) - set(dict2.keys())
    only_in_dict2 = set(dict2.keys()) - set(dict1.keys())

    # Compare the values of the common keys using the Gaussian kernel
    total_similarity = sum(gaussian_kernel(dict1[key], dict2[key], sigma) for key in common_keys)

    # Calculate the penalty based on the number of mismatched keys
    total_mismatched_keys = len(only_in_dict1) + len(only_in_dict2)
    total_keys = len(dict1) + len(dict2)
    
    # Apply penalty based on the mismatch keys ratio
    penalty = mismatch_penalty_weight * (total_mismatched_keys / total_keys)

    # Calculate the final similarity percentage
    if common_keys:  # Prevent division by zero
        similarity_percentage = ((total_similarity / len(common_keys)) * (1 - penalty)) * 100
    else:
        similarity_percentage = 0.0

    # Return the similarity percentage and other details
    return {
        "similarity_percentage": max(similarity_percentage, 0),  # Ensure no negative percentage
        "total_keys_dict1": len(dict1),
        "total_keys_dict2": len(dict2),
        "matching_keys": len(common_keys),
        "total_similarity": total_similarity,
        "total_mismatched_keys": total_mismatched_keys,
        "penalty": penalty
    }

if __name__ == '__main__':
    datas = ['iris','breast_cancer','covtype']
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument("-d", "--data",type=str,  default='covtype', help='data')
    parser.add_argument("-d", "--data",type=str,  default='covtype', help='data')
    parser.add_argument("-i", "--index",type=int,  default=3, help='index')
    parser.add_argument("-l", "--label",type=int,  default=None, help='index')
    parser.add_argument("-m", "--method",type=str,  default='KNeighborsClassifier', help='method')
    parser.add_argument("-a", "--alpha",type=float,  default=0.1, help='alpha')
    parser.add_argument("-e", "--explainer",type=str,  default='dst-lime', help='alpha')
    #parser.add_argument("-e", "--explainer",type=str,  default='lime', help='alpha')
    parser.add_argument("-nf", "--num_features",type=int,  default=10, help='num_features')
    args = parser.parse_args()

    args.index = 0
    args.explainer = 'lime'
    random.seed(1)
    exp,_ = ExecuteExplain(args)
    exp0 = copy.deepcopy(exp)
    label = list(exp0.score.keys())[0]
    features0 = [i[0] for i in exp0.local_exp[label]]
    dict_ = copy.deepcopy(exp0.local_exp[label])
    dict0 = {}
    for (k,v) in dict_: dict0[k] = v
    similarities = []
    data = {}
    for alpha_i in range(0,100,5):
        alpha = alpha_i/100
        args.explainer = 'dst-lime'
        args.alpha = alpha
        random.seed(1)
        exp,_ = ExecuteExplain(args)
        exp1 = copy.deepcopy(exp)
        label = list(exp0.score.keys())[0]
        features1 = [i[0] for i in exp1.local_exp[label]]
        dict_ = copy.deepcopy(exp1.local_exp[label])
        dict1 = {}
        for (k,v) in dict_: dict1[k] = v
        #similarity = order_of_pairs_similarity(features0,features1)
        similarity = dictionary_similarity(dict0,dict1)

        #print(similarity)
        similarities.append(similarity['similarity_percentage'])
        data[alpha] = similarity['similarity_percentage']
        print(similarities)

        outputName = dirname + "/Results/" + basename + "_" + args.data  + ".json"

        with open(outputName, 'w') as file:
            json.dump(data, file, indent=4)


  






