# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:38:37 2023

@author: zhanghai
"""

import numpy as np
import pandas as pd
import time
import copy
from math import ceil, floor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from BinaryCautiousRandomForest import WCRF

class CRFCFExplainer:
    def __init__(self, crf, train_set, feature_names, feature_cons=None, dist_type='L2'):
        self.model = crf
        self.rf = crf.model
        self.s = crf.s
        self.n_trees = crf.n_trees
        self.classes = crf.model.classes_
        self.n_classes = len(self.classes)
        self.train_set = train_set
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.immutable_features = feature_cons['immutable']
        self.increasing_features = feature_cons['increasing']
        self.decreasing_features = feature_cons['decreasing']
        self.feature_types = feature_cons['data types']
        self.dist_type = dist_type
        

    def fit(self):
        # calculate the distance adjustment terms
        self.dist_std = np.std(self.train_set, axis=0)
        medians_abs_diff = abs(self.train_set - np.median(self.train_set, axis=0))
        self.dist_mad = np.mean(medians_abs_diff, axis=0)
        self.dist_mad[self.dist_mad==0] = 1
        
        # extract decision rules [decision paths, probabilityies]
        # extract split values and split intervals
        decision_paths = np.zeros(2 * self.n_features)
        probability_intervals = np.zeros(2*len(self.classes))
        decision_paths_dict = {}
        t = -1
        for tree in self.rf.estimators_:
            t += 1
            decision_paths_dict[t] = {}
            n_nodes = tree.tree_.node_count
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            feature = tree.tree_.feature
            threshold = tree.tree_.threshold
            value = tree.tree_.value.reshape((-1, self.n_classes))
            
            dp_stack = [[[-0.0001, 1] for i in range(self.n_features)]]
            for d in range(self.n_features):
                dp_stack[0][d][1] = self.train_set[:,d].max()

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
                    decision_paths_dict[t][i] = dp_to_add
                    decision_paths = np.vstack((decision_paths, dp_to_add))
                    probability_intervals = np.vstack((probability_intervals, probabilities))
            
        self.decision_paths = decision_paths[1:,:]
        self.probability_intervals = probability_intervals[1:,:]
        # extrat split points and split intervals of each feature
        # construct association between split intervals and leaves
        self.split_points = {}
        self.split_intervals = {}
        self.intervals_with_leaves = {}
        for d in range(self.n_features):
            split_points_d = np.unique(self.decision_paths[:,2*d:2*d+2].reshape((1,-1)))
            split_points_d.sort()
            self.split_points[d] = split_points_d
            split_intervals_d = np.zeros((len(split_points_d)-1,2))
            split_intervals_d[:,0] = split_points_d[:-1]
            split_intervals_d[:,1] = split_points_d[1:]
            self.split_intervals[d] = split_intervals_d
            
            self.intervals_with_leaves[d] = {}
            decision_path_d = self.decision_paths[:,2*d:2*d+2]
            for i in range(len(split_intervals_d)):
                split_interval = (split_intervals_d[i,0], split_intervals_d[i,1])
                self.intervals_with_leaves[d][split_interval] = np.where(np.maximum(split_interval[0], decision_path_d[:,0]) 
                                                                       < np.minimum(split_interval[1], decision_path_d[:,1]))[0]
        leaves_index = self.rf.apply(self.train_set)
        leaves_index = np.unique(leaves_index, axis=0)
        self.live_regions = np.zeros((len(leaves_index), 2 * self.n_features))
        # construct live regions
        for i in range(len(leaves_index)):
            current_live_region = np.zeros((self.n_trees, 2 * self.n_features))
            for t in range(self.n_trees):
                current_live_region[t] = decision_paths_dict[t][leaves_index[i,t]]

            self.live_regions[i,0:-1:2] = current_live_region[:,0:-1:2].max(axis=0)
            self.live_regions[i,1::2] = current_live_region[:,1::2].min(axis=0)

        self.live_regions_predictions = self.model.predict((self.live_regions[:,0:-1:2]+self.live_regions[:,1::2])/2)
        
     
    def __instance_dist(self, x, X, dist_type):
        if dist_type == 'L1':
            dist = (abs(X - x)/self.dist_mad).sum(axis=1)
        elif dist_type == 'L0':
            dist = (X != x).sum(axis=1)
        else:
            dist = np.sqrt(((X - x)**2/self.dist_std).sum(axis=1))
        return dist
    
    
    def __generate_cf_in_regions(self, x, regions):
        candidates = x.reshape((1,-1)).repeat(len(regions), axis=0)
        for d in range(self.n_features):
            take_inf = regions[:,2*d] >= x[d]
            take_sup = regions[:,2*d+1] < x[d]
            if self.feature_types[d] == 'int':
                inf_values = np.ceil(regions[take_inf,2*d] + 1e-4)
                sup_values = np.floor(regions[take_sup,2*d+1])
            else:
                inf_values = regions[take_inf,2*d] + 1e-4
                sup_values = regions[take_sup,2*d+1]
            
            candidates[take_inf,d] = inf_values
            candidates[take_sup,d] = sup_values
            
        return candidates
            
                
    def mo(self, x, target, dist_type=None):
        if target not in self.classes:
            print("Your input target dose not existe!")
            return None, None
        
        if dist_type is None and self.dist_type is None:
            dist_type = 'L1'
                
        predictions = self.model.predict(self.train_set)
        remain_instances = self.train_set[predictions==target].copy()
        
        for d in range(self.n_features):
            feature = self.feature_names[d]
            
            if self.immutable_features is not None:
                if feature in self.immutable_features:
                    index = (x[d] == remain_instances[:,d])
                    remain_instances = remain_instances[index]
                    continue
            elif self.increasing_features is not None:
                if feature in self.increasing_features:
                    index = (x[d] <= remain_instances[:,d])
                    remain_instances = remain_instances[index]
                    continue
            elif self.decreasing_features is not None:
                if feature in self.decreasing_features:
                    index = (x[d] >= remain_instances[:,d])
                    remain_instances = remain_instances[index]
                    continue
            else:
                continue
            
        if len(remain_instances) == 0:
            print("Your feature constrains are too strict for this instance! Can't generate satisfied counterfactual example!")
            return None, None
        dists = self.__instance_dist(x, remain_instances, dist_type)
        cf_index = np.argmin(dists)
        min_dist = dists[cf_index]
        cf = remain_instances[cf_index]
        
        return cf, round(min_dist,4)
        

    def ofcc(self, x, target, dist_type=None):
        if target not in self.classes:
            print("Your input target dose not existe!")
            return None, None
        
        if dist_type is None and self.dist_type is None:
            dist_type = 'L1'
        
        cf = None
        min_dist = 1e8
        for d in range(self.n_features):
            feature = self.feature_names[d]
            split_points_d = self.split_points[d]
            if self.immutable_features is not None:
                if feature in self.immutable_features:
                    continue
            elif self.increasing_features is not None:
                if feature in self.increasing_features:
                    split_points_d = split_points_d[split_points_d > x[d]]
                    candidates = x.reshape((1,-1)).repeat(len(split_points_d), axis=0)
                    if self.feature_types[d] == 'int':
                        candidates[:,d] = np.ceil(split_points_d + 1e-4)
                    else: 
                        candidates[:,d] = split_points_d + 1e-4
            elif self.decreasing_features is not None:
                if feature in self.decreasing_features:
                    split_points_d = split_points_d[split_points_d < x[d]]
                    candidates = x.reshape((1,-1)).repeat(len(split_points_d), axis=0)
                    if self.feature_types[d] == 'int':
                        candidates[:,d] = np.floor(split_points_d)
                    else:
                        candidates[:,d] = split_points_d
            else:
                candidates = x.reshape((1,-1)).repeat(len(split_points_d), axis=0)
                index1 = split_points_d > x[d]
                index2 = split_points_d < x[d]
                if self.feature_types[d] == 'int':
                    candidates[index1,d] = np.ceil(split_points_d[index1] + 1e-4)
                    candidates[index2,d] = np.floor(split_points_d[index2])
                else:
                    candidates[index1,d] = split_points_d[index1] + 1e-4
                    candidates[index2,d] = split_points_d[index2]
            
            predictions = self.model.predict(candidates)
            remain_instances = candidates[predictions==target]
            
            if len(remain_instances) == 0:
                continue
            else:
                dists = self.__instance_dist(x, remain_instances, dist_type)
                min_index = np.argmin(dists)
                
                if dists[min_index] < min_dist:
                    min_dist = dists[min_index]
                    cf = remain_instances[min_index]
                else:
                    continue
        
        if cf is None:
            print("Can't get one-feature-changed counterfactual example!")
            return None, None
        else:
            return cf, round(min_dist,4)
        
    
    def discern(self, x, target, dist_type=None):
        pass
            
    
    def lire(self, x, target, dist_type=None):
        if target not in self.classes:
            print("Your input target dose not existe!")
            return None, None
        
        live_regions = self.live_regions[self.live_regions_predictions==target].copy()
        
        for d in range(self.n_features):
            feature = self.feature_names[d]
            
            if self.immutable_features is not None:
                if feature in self.immutable_features:
                    index = (live_regions[:,2*d] < x[d]) * (x[d] <= live_regions[:,2*d+1])
                    live_regions = live_regions[index]
                    continue
            elif self.increasing_features is not None:
                if feature in self.increasing_features:
                    index = live_regions[:,2*d+1] > x[d]
                    live_regions = live_regions[index]
                    continue
            elif self.decreasing_features is not None:
                if feature in self.decreasing_features:
                    index = live_regions[:,2*d] < x[d]
                    live_regions = live_regions[index]
                    continue
            else:
                continue
        
        if len(live_regions)==0:
            print("Your feature constrains are too strict for this instance! Can't generate satisfied counterfactual example!")
            return None, None
        else:
            candidates = self.__generate_cf_in_regions(x, live_regions)
            dists = self.__instance_dist(x, candidates, dist_type)
            cf_index = np.argmin(dists)
            min_dist = dists[cf_index]
            cf = candidates[cf_index]
            
            return cf, round(min_dist,4)
        
    
    
    def eece(self, x, target, dist_type=None):
        regions = np.concatenate((self.decision_paths.copy(), self.live_regions.copy()), axis=0)
        for d in range(self.n_features):
            feature = self.feature_names[d]
            
            if self.immutable_features is not None:
                if feature in self.immutable_features:
                    index = (regions[:,2*d] < x[d]) * (x[d] <= regions[:,2*d+1])
                    regions = regions[index]
                    continue
            elif self.increasing_features is not None:
                if feature in self.increasing_features:
                    index = regions[:,2*d+1] > x[d]
                    regions = regions[index]
                    continue
            elif self.decreasing_features is not None:
                if feature in self.decreasing_features:
                    index = regions[:,2*d] < x[d]
                    regions = regions[index]
                    continue
            else:
                continue
        
        if len(regions)==0:
            print("Your feature constrains are too strict for this instance! Can't generate satisfied counterfactual example!")
            return None, None
        else:
            candidates = self.__generate_cf_in_regions(x, regions)
            predictions = self.model.predict(candidates)
            candidates = candidates[predictions==target]
            if len(candidates)==0:
                print("Your feature constrains are too strict for this instance! Can't generate satisfied counterfactual example!")
                return None, None
            else:
                dists = self.__instance_dist(x, candidates, dist_type)
                cf_index = np.argmin(dists)
                min_dist = dists[cf_index]
                cf = candidates[cf_index]
                
                return cf, round(min_dist,4)
        
        
                

if __name__ == "__main__":
    data = pd.read_csv('pima.csv')
    feature_names = list(data.columns)[:-1]
    feature_cons = {'immutable':None,
                    'increasing':None,
                    # 'increasing':['Pregnancies', 'Age'],
                    'decreasing':None,
                    'data types':['int', 'int', 'int', 'int', 'int', 'float', 'float', 'int']}
    dist_type = 'L2'
    X = np.array(data.iloc[:,:-1])
    y = np.array(data.iloc[:,-1]) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    wcrf = WCRF(n_trees=10, s=3, gamma=10, labda=10, tree_max_depth=7, combination=1)
    wcrf.fit(X_train, y_train)
    predictions = wcrf.predict(X_test)
    indeterminate_index = np.where(predictions==-1)[0]
    print(indeterminate_index)
    
    explainer = CRFCFExplainer(wcrf, X_train, feature_names, feature_cons, dist_type)
    explainer.fit()
    x = X_test[indeterminate_index[-1]]
    print(x)
    target = 1
    print("MO")
    cf, dist = explainer.mo(x, target,'L0')
    print(dist, wcrf.predict(cf))
    cf, dist = explainer.mo(x, target,'L1')
    print(dist, wcrf.predict(cf))
    cf, dist = explainer.mo(x, target,'L2')
    print(dist, wcrf.predict(cf))
    
    print("OFCC")
    cf, dist = explainer.ofcc(x, target,'L0')
    print(dist, wcrf.predict(cf))
    cf, dist = explainer.ofcc(x, target,'L1')
    print(dist, wcrf.predict(cf))
    cf, dist = explainer.ofcc(x, target,'L2')
    print(dist, wcrf.predict(cf))
    
    print("Lire")
    cf, dist = explainer.lire(x, target,'L0')
    print(dist, wcrf.predict(cf))
    cf, dist = explainer.lire(x, target,'L1')
    print(dist, wcrf.predict(cf))
    cf, dist = explainer.lire(x, target,'L2')
    print(dist, wcrf.predict(cf))
    
    print("EECE")
    cf, dist = explainer.eece(x, target,'L1')
    print(dist, wcrf.predict(cf))
    cf, dist = explainer.eece(x, target,'L2')
    print(dist, wcrf.predict(cf))
    cf, dist = explainer.eece(x, target,'L0')
    print(dist, wcrf.predict(cf))