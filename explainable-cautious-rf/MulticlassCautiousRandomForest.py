# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:38:37 2023

@author: zhanghai
"""


import numpy as np
import itertools
from sklearn.utils import check_X_y
from sklearn.ensemble import RandomForestClassifier


class CautiousRandomForest:
    def __init__(self, n_trees=100, s=2, min_samples_leaf=1,combination='cdm-vote', discount_type='u65', random_state=42):
        self.n_trees = n_trees
        self.s = s
        self.combination = combination
        self.discount_type = discount_type
        self.__discount_ratio = self.__define_discount_ratio(discount_type)
        self.classes = None
        self.n_class = None
        self.__intervals_leaves = None
        self.rf = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=min_samples_leaf, random_state=random_state)
        
        
    def fit(self, train_X, train_y):
        check_X_y(train_X, train_y)
        
        self.rf.fit(train_X, train_y)
        
        self.classes = self.rf.classes_
        self.n_class = len(self.rf.classes_)

        n_sample_leaves = {}
        intervals_leaves = {}

        for t in range(self.n_trees):
            n_sample_leaves[t] = {}
            intervals_leaves[t] = {}
            tree = self.rf.estimators_[t]

            n_nodes = tree.tree_.node_count
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            sample_count = tree.tree_.value.reshape((-1, self.n_class))

            for i in range(n_nodes):
                is_leaf_node = (children_left[i] == children_right[i])
                if is_leaf_node:
                    n_sample_leaves[t][i] = sample_count[i]
                    n_total_sample = sum(sample_count[i])
                    intervals = sample_count[i].repeat(2).reshape(self.n_class, 2)
                    intervals[:,0] = intervals[:,0]/(n_total_sample + self.s)
                    intervals[:,1] = (intervals[:,1] + self.s)/(n_total_sample + self.s)
                    intervals_leaves[t][i] = intervals
                    
        self.__intervals_leaves = intervals_leaves
        
        
    def __define_discount_ratio(self,i):
        if self.discount_type == 'f1':
            def discount_ratio(i):
                return 2/(1+i)
        elif self.discount_type == 'u80':
            def discount_ratio(i):
                return -1.2/(i**2) + 2.2/i
        else:
            def discount_ratio(i):
                return -0.6/(i**2) + 1.6/i
            self.discount_type = 'u65'
        return discount_ratio
        
    
    def __ndc(self, probabilities):
        class_order = np.argsort(-probabilities)
        max_eu = 0
        top_k = 0
        for k in range(1,self.n_class+1):
            discount_ratio = self.__discount_ratio(k)
            if discount_ratio < max_eu:
                break
            else:
                probability_sum = np.sum(probabilities[class_order[0:k]])
                eu = discount_ratio * probability_sum
                if eu > max_eu:
                    max_eu = eu
                    top_k = k
        
        return list(self.classes[class_order[0:top_k]])
    
    
    def __instance_interval_dominance(self, instance_intervals):
        n_class = len(instance_intervals)
        decision = []
        for k in range(n_class):
            other_classes = np.setdiff1d(np.arange(n_class), np.array([k]))
            if np.any(instance_intervals[k, 1] < instance_intervals[other_classes, 0]):
                continue
            else:
                decision.append(k)
                
        return decision
    
    
    def __mva(self, intervals):
        # intervals here is numpy array of shape (T, n_class, 2)
        vote_against = np.zeros(self.n_class)
        for t in range(self.n_trees):
            t_non_dominated_class = self.__instance_interval_dominance(intervals[t])
            t_dominated_class = np.setdiff1d(np.arange(self.n_class), np.array(t_non_dominated_class))
            for c in t_dominated_class:
                vote_against[c] += 1
        mva = vote_against.min()
        predictions_index = np.where(vote_against==mva)[0]
        
        return list(self.classes[predictions_index])
                
    
    def __ave(self, intervals):
        # intervals here is numpy array of shape (T, n_class, 2)
        ave_intervals = intervals.mean(axis=0)
        predictions_index = self.__instance_interval_dominance(ave_intervals)
        
        return list(self.classes[predictions_index])
    
    
    def __cdm_ave(self, intervals):
        # intervals here is numpy array of shape (T, n_class, 2)
        ave_intervals = intervals.mean(axis=0)
        bels = ave_intervals[:,0]
        class_order = np.argsort(-bels)
        max_leu = 0
        top_k = 0
        for k in range(1,self.n_class+1):
            discount_ratio = self.__discount_ratio(k)
            if discount_ratio < max_leu:
                break
            else:
                if k == self.n_class:
                    bel = 1
                else: 
                    bel = np.sum(bels[class_order[0:k]])
                leu = discount_ratio * bel
                if leu > max_leu:
                    max_leu = leu
                    top_k = k
        return list(self.classes[class_order[0:top_k]])
    
    
    def __cdm_slow_vote(self, intervals):
        # intervals here is numpy array of shape (T, n_class, 2)
        mass_function = {}
        for t in range(self.n_trees):
            t_non_dominated_class = tuple(self.__instance_interval_dominance(intervals[t]))
            if t_non_dominated_class not in list(mass_function.keys()):
                mass_function[t_non_dominated_class] = 0
            mass_function[t_non_dominated_class] += 1/self.n_trees

        max_leu = 0
        prediction_index = None
        focal_elements = list(mass_function.keys())
        for k in range(1, self.n_class+1):
            discount_ratio = self.__discount_ratio(k)
            if discount_ratio < max_leu or k > 5:
                break
                
            for subset_of_omega in itertools.combinations(np.arange(self.n_class), k):
                if len(subset_of_omega) == 0:
                    continue
                
                bel = 0
                for focal_element in focal_elements:
                    if set(focal_element).issubset(subset_of_omega):
                        bel += mass_function[focal_element]
                        
                leu = discount_ratio * bel

                if leu > max_leu:
                    max_leu = leu
                    prediction_index = subset_of_omega
                    
        return self.classes[list(prediction_index)]
    
    
    def __cdm_vote(self, intervals):
        # intervals here is numpy array of shape (T, n_class, 2)
        mass_function = {}
        focal_elements = {}
        considering_class_flag = np.zeros((self.n_class+1, self.n_class+1))
        for t in range(self.n_trees):
            t_non_dominated_class = tuple(self.__instance_interval_dominance(intervals[t]))
            cardinality = len(t_non_dominated_class)
            if cardinality not in list(mass_function.keys()):
                mass_function[cardinality] = {}
                focal_elements[cardinality] = []
            if t_non_dominated_class not in list(mass_function[cardinality].keys()):
                mass_function[cardinality][t_non_dominated_class] = 0
                focal_elements[cardinality].append(t_non_dominated_class)
                for c in t_non_dominated_class:
                    considering_class_flag[cardinality, c] = 1
            mass_function[cardinality][t_non_dominated_class] += 1/self.n_trees
            
        considering_class = {}
        for k in range(1, self.n_class+1):
            flag = considering_class_flag[:k+1].sum(axis=0)
            considering_class[k] = np.where(flag>0)[0]
            
        max_leu = 0
        prediction_index = None
        for k in range(1, self.n_class+1):
            discount_ratio = self.__discount_ratio(k)
            if discount_ratio < max_leu or k > 5:
                break
            
            possible_subsets = itertools.combinations(considering_class[k], k)
            for subset in possible_subsets:
                bel = 0
                for i in range(1, k+1):
                    if i not in list(focal_elements.keys()):
                        continue
                    else:
                        for focal_element in focal_elements[i]:
                            if set(focal_element).issubset(subset):
                                bel += mass_function[i][focal_element]
                        
                leu = discount_ratio * bel

                if leu > max_leu:
                    max_leu = leu
                    prediction_index = subset
                    
        return self.classes[list(prediction_index)]
    
        
    def predict(self, X, dacc=None):
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        predictions = []
        n_instance = X.shape[0]
        leaves_index = self.rf.apply(X)
        
        if self.combination == 'ndc':
            all_proabilities = self.rf.predict_proba(X)
            for i in range(n_instance):
                predictions.append(self.__ndc(all_proabilities[i]))
            return predictions
        
        # get all [bel, pl] intervals for all instances, shape of (n_instance, T, n_class, 2)
        all_intrvals = np.zeros((n_instance, self.n_trees, self.n_class, 2))
        for i in range(n_instance):
            for t in range(self.n_trees):
                all_intrvals[i, t] = self.__intervals_leaves[t][leaves_index[i,t]]
                
        if self.combination == 'mva':
            # MVA
            for i in range(n_instance):
                predictions.append(self.__mva(all_intrvals[i]))
            return predictions
        
        elif self.combination == 'ave':
            # AVE
            for i in range(n_instance):
                predictions.append(self.__ave(all_intrvals[i]))
            return predictions
        
        elif self.combination == 'cdm-ave':
            # generalized ave
            for i in range(n_instance):
                predictions.append(self.__cdm_ave(all_intrvals[i]))
            return predictions
        
        else:
            # default cdm-vote
            for i in range(n_instance):
#                 predictions.append(self.__cdm_vote(all_intrvals[i]))
                predictions.append(self.__cdm_slow_vote(all_intrvals[i]))
            return predictions
        
        
    def score(self, X_test, y_test):
        # get both imprecise and precise predictions 
        predictions = self.predict(X_test)
        determinacy = 0
        single_accuracy = 0
        set_accuracy = 0
        set_size = 0
        u65 = 0
        f1 = 0
        u80 = 0
        for i in range(len(y_test)):
            prediction = predictions[i]
            if len(prediction) == 1:
                determinacy += 1
                
                if prediction[0] == y_test[i]:
                    single_accuracy += 1
                    u65 += 1
                    f1 += 1
                    u80 += 1
            else:
                set_size += len(prediction)
                if y_test[i] in prediction:
                    set_accuracy += 1
                    u65 += (-0.6/(len(prediction)**2) + 1.6/len(prediction))
                    f1 += 2/(1+len(prediction))
                    u80 += (-1.2/(len(prediction)**2) + 2.2/len(prediction))
                    
        n_determinate = determinacy
        n_indeterminate = len(y_test) - determinacy
        
        determinacy /= len(y_test)
        if n_determinate == 0:
            single_accuracy = None
        else:
            single_accuracy /= n_determinate
        if n_indeterminate == 0:
            set_accuracy = None
            set_size = None
        else:
            set_accuracy /= n_indeterminate
            set_size /= n_indeterminate
        u65 /= len(y_test)
        f1 /= len(y_test)
        u80 /= len(y_test)
                
        return {'determinacy': round(determinacy,4),
                 'single accuracy': None if single_accuracy is None else round(single_accuracy,4),
                 'set accuracy': None if set_accuracy is None else round(set_accuracy,4),
                 'set size': None if set_size is None else round(set_size,4),
                 'u65 score': round(u65,4),
                 'f1 score': round(f1,4),
                 'u80 score': round(u80,4)}