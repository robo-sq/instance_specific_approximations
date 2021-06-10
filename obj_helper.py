""" This file contains code to get objective functions
that can be used as input to submodular maximization algorithms
and approximation algorithms. """

import pickle
import numpy as np
import pandas as pd
from data_helper import DATA_PATH
from utils import *

def entropy(Y):
    """
    Also known as Shanon Entropy
    Reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    unique, count = np.unique(Y, return_counts=True, axis=0)
    prob = count/len(Y)
    en = np.sum((-1)*prob*np.log2(prob))
    return en

def jEntropy(Y,X):
    """
    H(Y;X)
    Reference: https://en.wikipedia.org/wiki/Joint_entropy
    """
    YX = np.c_[Y,X]
    return entropy(YX)

def mi(X, Y):
    return entropy(X) + entropy(Y) - jEntropy(X,Y)


def get_obj(objective, dataset, **kwargs):
    if objective == 'influence_maximization':
        # for youtube and citation netowrks
        def obj_wrapper(G, return_set=False):
    
            def obj_fun(el_selected):
                
                # for each node not in S, sum weight
                neigh_list = []
                for j in el_selected:
                    neigh_list += list(np.where(G[j,:] == 1)[0])
                if return_set:
                    return len(set(neigh_list)), set(neigh_list)
                else:
                    return len(set(neigh_list))
            
            return obj_fun
    elif objective == 'car_dispatch':
        # for uber dataset
        def obj_wrapper(G, weights=None, return_set=False):

            if weights is None:
                weights = np.ones(len(G))
            
            def obj_fun(el_selected):
                
                # for each node not in S, sum weight
                neigh_list = []
                for j in el_selected:
                    neigh_list += list(np.where(G[j,:] == 1)[0])
                if return_set:
                    return np.sum([weights[n] for n in set(neigh_list)]), set(neigh_list)
                else:
                    return np.sum([weights[n] for n in set(neigh_list)])
            
            return obj_fun
    elif objective == 'movie_recommendation':
        # for movielens
        def obj_wrapper(dataset):
            
            # grab high ratings
            row_idx, col_idx = np.where(dataset>=4.5)
            
            high_rat = np.zeros(dataset.shape)
            for idx, i in enumerate(row_idx):
                high_rat[i,col_idx[idx]] = 1
            
            def obj_fun(el_selected):
                
                # ratings
                tot_rat = np.sum(dataset[:,el_selected])

                # compute diversity
                user_list = []
                for idx_m in el_selected:
                    user_list += list(np.where(high_rat[:,idx_m] == 1)[0])
                    
                cov = len(set(user_list))

                return tot_rat + cov
            
            return obj_fun
    elif objective == 'revenue_maximization':
        # for facebook data
        def obj_wrapper(G):
            alpha = 0.9
            
            def obj_fun(el_selected):
                el_selected = list(set(el_selected))
                
                # for each node not in S, sum weight
                tot_weight = 0
                G_new = G[:, el_selected]
                for i in G_new: 
                    tot_weight += (np.sum(i))**alpha
                
        
                return tot_weight
            
            return obj_fun
    elif objective == 'feature_selection':
        # for census data
        def obj_wrapper(dataset):
            
            X = dataset[:,:-1]
            y = dataset[:,-1]
            
            def obj_fun(el_selected):
                
                if len(el_selected) == 0:
                    return 0
                else:
                    return jEntropy(y, X[:, el_selected])

            
            return obj_fun
    elif objective == 'facility_location':
        # for movielens
        def obj_wrapper(dataset):
            
            m = len(dataset)
            
            def obj_fun(el_selected):

                if len(el_selected) == 0:
                    return 0

                return 1/m*np.sum(np.max(dataset[:,el_selected], axis=1))
            
            return obj_fun
    elif objective == 'concave_over_modular':
        # for movielens
        def obj_wrapper(dataset):

            m = dataset.shape[0]
            print(m)
            
            def obj_fun(el_selected):
                el_selected = list(set(el_selected))
                
                if len(el_selected) == 0:
                    return 0
                elif round((1/m*np.sum(dataset[:,el_selected])), 2) == 0:
                    return 0
                else:
                    tot_rat = (1/m*np.sum(dataset[:,el_selected]))**0.8

                    return tot_rat
            
            return obj_fun
        
    elif objective == 'sensor_placement':
        # for sensor data
        def obj_wrapper(dataset):
            
            def obj_fun(el_selected):

                if len(el_selected) == 0:
                    return 0
                else:
                    return entropy(dataset[:, el_selected])
            
            return obj_fun
    obj = obj_wrapper(dataset, **kwargs)        
    return obj