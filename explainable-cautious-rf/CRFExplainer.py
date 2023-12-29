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
        self.epsilon = 1e-4
        self.lof = LocalOutlierFactor(n_neighbors=5, novelty=True)
        

    def fit(self):
        # get feature importance to determin feature order
        feature_importance = self.rf.feature_importances_
        self.feature_order = np.argsort(feature_importance)
        
        # calculate the distance adjustment terms
        self.dist_std = np.std(self.train_set, axis=0)
        medians_abs_diff = abs(self.train_set - np.median(self.train_set, axis=0))
        self.dist_mad = np.mean(medians_abs_diff, axis=0)
        self.dist_mad[self.dist_mad==0] = 1
        
        # train the local outlier factor
        self.lof.fit(self.train_set)
        
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
            dists = (abs(X - x)/self.dist_mad).sum(axis=1)
        elif dist_type == 'L0':
            dists = (X != x).sum(axis=1)
        else:
            dists = np.sqrt(((X - x)**2/self.dist_std).sum(axis=1))
        return dists
    
    
    def __interval_dist(self, d, x_d, intervals, dist_type):
        in_index = (intervals[:,0] < x_d) * (x_d <= intervals[:,1])
        left_index = intervals[:,1] < x_d
        right_index = intervals[:,0] >= x_d
        dists = np.zeros(len(intervals))
        dists[in_index] = 0
        if self.feature_types[d] == 'int':
            dists[left_index] = x_d - np.floor(intervals[left_index,1])
            dists[right_index] = np.ceil(intervals[right_index,1] + self.epsilon) - x_d
        else:
            dists[left_index] = x_d - intervals[left_index,1]
            dists[right_index] = intervals[right_index,1] - x_d + self.epsilon
        if dist_type == 'L1':
            dists = dists/self.dist_mad[d]
        elif dist_type == 'L0':
            dists = (dists!=0)*1
        else:
            dists = dists**2/self.dist_std[d] # not squared
        
        return dists
    
    
    def __generate_cf_in_regions(self, x, regions):
        candidates = x.reshape((1,-1)).repeat(len(regions), axis=0)
        for d in range(self.n_features):
            take_inf = regions[:,2*d] >= x[d]
            take_sup = regions[:,2*d+1] < x[d]
            if self.feature_types[d] == 'int':
                inf_values = np.ceil(regions[take_inf,2*d] + self.epsilon)
                sup_values = np.floor(regions[take_sup,2*d+1])
            else:
                inf_values = regions[take_inf,2*d] + self.epsilon
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
    
    
    def discern(self, x, target, dist_type=None):
        init_cf, init_min_dist = self.mo(x, target, dist_type)
        cf = x.copy()
        if init_cf is None:
            return init_cf, init_min_dist
        else:
            for d in self.feature_order:
                cf[d] = init_cf[d]
                prediction = self.model.predict(cf.reshape((1,-1)))[0]
                if prediction ==  target:
                    min_dist = self.__instance_dist(x, cf.reshape((1,-1)), dist_type)[0]
                    return cf, round(min_dist, 4)
        

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
                        candidates[:,d] = np.ceil(split_points_d + self.epsilon)
                    else: 
                        candidates[:,d] = split_points_d + self.epsilon
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
                    candidates[index1,d] = np.ceil(split_points_d[index1] + self.epsilon)
                    candidates[index2,d] = np.floor(split_points_d[index2])
                else:
                    candidates[index1,d] = split_points_d[index1] + self.epsilon
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
            
    
    def lire(self, x, target, dist_type=None):
        if target not in self.classes:
            print("Your input target dose not existe!")
            return None, None
        
        if dist_type is None and self.dist_type is None:
            dist_type = 'L1'
        
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
        if target not in self.classes:
            print("Your input target dose not existe!")
            return None, None
        
        if dist_type is None and self.dist_type is None:
            dist_type = 'L1'
            
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
    
            
    def __filter_split_intervals(self, x, init_dist, dist_type):
        split_intervals = copy.deepcopy(self.split_intervals)
        n_intervals = np.zeros(self.n_features)
        split_interval_dists = {}
        for d in range(self.n_features):
            feature = self.feature_names[d]
            split_intervals_d = split_intervals[d]
            
            if self.immutable_features is not None:
                if feature in self.immutable_features:
                    index = (split_intervals_d[:,0] < x[d]) * (x[d] <= split_intervals_d[:,1])
                    split_intervals[d] = split_intervals_d[index]
                    n_intervals[d] = 1
                    continue
            elif self.increasing_features is not None:
                if feature in self.increasing_features:
                    index = split_intervals_d[:,1] >= x[d]
                    split_intervals_d = split_intervals_d[index]
                    
                    dists = self.__interval_dist(d, x[d], split_intervals_d, dist_type)
                    interval_order = np.argsort(dists)
                    dists = dists[interval_order]
                    split_intervals_d = split_intervals_d[interval_order]
                    
                    index = dists <= init_dist
                    n_intervals[d] = sum(index)
                    split_interval_dists[d] = dists[index]
                    split_intervals[d] = split_intervals_d[index]
                    continue
            elif self.decreasing_features is not None:
                if feature in self.decreasing_features:
                    index = split_intervals_d[:,0] < x[d]
                    split_intervals_d = split_intervals_d[index]
                    
                    dists = self.__interval_dist(d, x[d], split_intervals_d, dist_type)
                    interval_order = np.argsort(dists)
                    dists = dists[interval_order]
                    split_intervals_d = split_intervals_d[interval_order]
                    
                    index = dists <= init_dist
                    n_intervals[d] = sum(index)
                    split_interval_dists[d] = dists[index]
                    split_intervals[d] = split_intervals_d[index]
                    continue
            else:
                dists = self.__interval_dist(d, x[d], split_intervals_d, dist_type)
                interval_order = np.argsort(dists)
                dists = dists[interval_order]
                split_intervals_d = split_intervals_d[interval_order]
                
                index = dists <= init_dist
                n_intervals[d] = sum(index)
                split_interval_dists[d] = dists[index]
                split_intervals[d] = split_intervals_d[index]
        
        return split_intervals, split_interval_dists, n_intervals
    
    
    def exact_cf(self,  x, target, dist_type=None):
        if target not in self.classes:
            print("Your input target dose not existe!")
            return None, None
        
        if dist_type is None and self.dist_type is None:
            dist_type = 'L1'
            
        init_cf, init_dist = self.eece(x, target, dist_type)
        if dist_type == 'L2':
            init_dist = init_dist ** 2
        if init_dist is not None:
            # filter split intervals
            split_intervals, split_interval_dists, n_intervals = self.__filter_split_intervals(x, init_dist, dist_type)
        else:
            print("Your feature constrains are too strict for this instance! Can't generate satisfied counterfactual example!")
            return None, None

        n_checked_intervals = np.zeros(self.n_features, 'int64')
        current_level = 0
        current_region = np.zeros(2*self.n_features)
        feature_dists= np.zeros(self.n_features)
        leaf_index = {}
        cf = init_cf.copy()
        min_dist = copy.deepcopy(init_dist)
        start_time = time.time()
        while True:
            end_time = time.time()
            d = self.feature_order[current_level]
            if current_level < 0 or (end_time - start_time) > 10:
                if dist_type=='L2':
                    min_dist = np.sqrt(min_dist)
                return cf, round(min_dist,4)
            elif n_checked_intervals[d]==n_intervals[d]:
                n_checked_intervals[d] = 0
                feature_dists[d] = 0
                current_level -= 1
            else:
                feature_dists[d] = split_interval_dists[d][n_checked_intervals[d]]
                
                # confirm cumulative distance
                if sum(feature_dists) >= min_dist:
                    n_checked_intervals[d] = 0
                    feature_dists[d] = 0
                    current_level -= 1
                else:
                    interval = split_intervals[d][n_checked_intervals[d]]
                    interval_key = (interval[0],interval[1])
                    n_checked_intervals[d] += 1
                    current_region[2*d:2*d+2] = interval
                    
                    # calculate leaf index
                    if current_level == 0:
                        leaf_index[current_level] = self.intervals_with_leaves[d][interval_key]
                    else:
                        leaf_index[current_level] = np.intersect1d(leaf_index[current_level-1], 
                                                                   self.intervals_with_leaves[d][interval_key])
                    # check if arrived in a leaf
                    if current_level == self.n_features-1:
                        prbability_intervals = self.probability_intervals[leaf_index[current_level]]
                        probabilities_aggr = np.zeros_like(prbability_intervals, dtype='bool')
                        probabilities_aggr[:,:2*self.n_classes:2] = prbability_intervals[:,:2*self.n_classes:2] >= 0.5
                        probabilities_aggr[:,1:2*self.n_classes:2] = prbability_intervals[:,1:2*self.n_classes:2] > 0.5
                        probabilities_aggr = np.mean(probabilities_aggr,axis=0)
                        if probabilities_aggr[2] >= 0.5:
                            current_prediction = 1
                        elif probabilities_aggr[3] <= 0.5:
                            current_prediction = 0
                        else:
                            current_prediction = -1
                        
                        # check if reach target
                        if current_prediction == target:
                            candidate = self.__generate_cf_in_regions(x, current_region.reshape((1,-1)))
                            # check plausibility
                            if self.lof.predict(candidate)[0] == 1:
                                cf = candidate[0]
                                min_dist = sum(feature_dists)
                    else:
                        current_level += 1
        
        
                

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
    wcrf = WCRF(n_trees=50, s=3, gamma=10, labda=10, tree_max_depth=7, combination=1)
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
    
    print("DisCERN")
    cf, dist = explainer.discern(x, target,'L0')
    print(dist, wcrf.predict(cf))
    cf, dist = explainer.discern(x, target,'L1')
    print(dist, wcrf.predict(cf))
    cf, dist = explainer.discern(x, target,'L2')
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
    cf, dist = explainer.eece(x, target,'L0')
    print(dist, wcrf.predict(cf))
    cf, dist = explainer.eece(x, target,'L1')
    print(dist, wcrf.predict(cf))
    cf, dist = explainer.eece(x, target,'L2')
    print(dist, wcrf.predict(cf))
    
    print("Exact CF")
    cf, dist = explainer.exact_cf(x, target,'L0')
    print(dist, wcrf.predict(cf))
    cf, dist = explainer.exact_cf(x, target,'L1')
    print(dist, wcrf.predict(cf))
    cf, dist = explainer.exact_cf(x, target,'L2')
    print(dist, wcrf.predict(cf))