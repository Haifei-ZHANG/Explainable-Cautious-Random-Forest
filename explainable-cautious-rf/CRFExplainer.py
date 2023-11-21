# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:38:37 2023

@author: zhanghai
"""

import numpy as np
import pandas as pd
import time
import copy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from BinaryCautiousRandomForest import WCRF

class CRFCFExplainer:
    def __init__(self, crf, train_set, feature_names, feature_cons=None):
        self.model = crf
        self.rf = crf.model
        self.s = crf.s
        self.classes = crf.model.classes_
        self.n_classes = len(self.classes)
        self.train_set = train_set
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.immutable_features = feature_cons['immutable']
        self.increasing_features = feature_cons['increasing']
        self.decreasing_features = feature_cons['decreasing']


    def fit(self):
        # extract decision rules [decision paths, probabilityies]
        # extract split values and split intervals
        decision_paths = np.zeros(2 * self.n_features)
        probability_intervals = np.zeros(2*len(self.classes))
        
        for tree in self.rf.estimators_:
            
            n_nodes = tree.tree_.node_count
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            feature = tree.tree_.feature
            threshold = tree.tree_.threshold
            value = tree.tree_.value.reshape((-1, self.n_classes))
            
            dp_stack = [[[-0.0001, 1] for i in range(self.n_features)]]

            for i in range(n_nodes):
                is_internal_node = (children_left[i] != children_right[i])
                if is_internal_node:
                    parent_dp = dp_stack.pop()
                    left_dp = copy.deepcopy(parent_dp)
                    right_dp = copy.deepcopy(parent_dp)
                    dp_stack.append(right_dp)
                    dp_stack.append(left_dp)
                    dp_stack[-1][feature[i]][1] = threshold[i]
                    dp_stack[-2][feature[i]][0] = threshold[i]
                    

                else:
                    N_S = sum(value[i]) + self.s
                    n_samples = value[i].repeat(2)
                    n_samples[1:2*len(self.classes):2] = n_samples[1:2*self.n_classes:2] + self.s
                    probabilities = (n_samples/N_S).round(4)
                    
                    dp_to_add = np.array(dp_stack.pop()).flatten()
                    decision_paths = np.vstack((decision_paths, dp_to_add))
                    probability_intervals = np.vstack((probability_intervals, probabilities))
            
        self.decision_paths = decision_paths[1:,:]
        self.probability_intervals = probability_intervals[1:,:]
        
        self.split_points = {}
        self.split_intervals = {}
        for d in range(self.n_features):
            split_points_d = np.unique(self.decision_paths[:,2*d:2*d+2].reshape((1,-1)))
            split_points_d.sort()
            self.split_points[d] = split_points_d
            split_intervals_d = np.zeros((len(split_points_d)-1,2))
            split_intervals_d[:,0] = split_points_d[:-1]
            split_intervals_d[:,1] = split_points_d[1:]
            self.split_intervals[d] = split_intervals_d
            
        # construct association between split intervals and leaves
        
        


if __name__ == "__main__":
    data = pd.read_csv('pima.csv')
    feature_names = list(data.columns)[:-1]
    feature_cons = {'immutable':None,
                    'increasing':None,
                    'decreasing':None}
    X = np.array(data.iloc[:,:-1])
    y = np.array(data.iloc[:,-1]) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    wcrf = WCRF(n_trees=100, s=3, gamma=10, labda=10, tree_max_depth=7, combination=1)
    wcrf.fit(X_train, y_train)
    e = wcrf.evaluate(X_test, y_test)
    print(e)
    explainer = CRFCFExplainer(wcrf, X_train, feature_names, feature_cons)
    explainer.fit()