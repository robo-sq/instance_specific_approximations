""" This file contains code for submodular maximization algorithms:
Greedy, Local Search, Random Greedy, Lazier then Lazy Greedy.
In addition, it contains two functions to compute OPT, one by brute
force and another specifically for coverage objectives that uses
integer programming.
"""

import numpy as np
import pickle
from itertools import combinations 
from mip import Model, xsum, maximize, BINARY
from utils import *

def run_greedy(k_to_select, ground_set, obj_fun, fname='tmp', print_log=True, save_iter=10, restart=None):

    filename = fname + '_greedy.p'
    # get starting set of data
    if restart is None:
        el_selected = [([], 0)]
    else:
        el_selected = pickle.load(open(RESULTS_PATH + restart, "rb" ))

    # loop through predictors and at each step,
    # add one predictor that increases obj the most
    if restart is None:
        best_k_plus_1 = el_selected[-1][0] # should be []
    else:
        # need to aggregate previous ones
        x_list = [i[0] for i in el_selected]
        best_k_plus_1 = [j for i in x_list for j in i]
    
    # loop through k to select (remaining) el for a total of 
    # k_to_select 
    for k in range(len(best_k_plus_1), k_to_select):
        
        best_k_el = best_k_plus_1
        obj_val = []

        el_list = list(ground_set - set(best_k_el))
        for el in el_list:
            k_plus_1 = list(best_k_el + [el])

            obj_val.append(obj_fun(k_plus_1))

        best_el = [el_list[np.argmax(obj_val)]]
        best_k_plus_1 = best_k_el + best_el
        
        el_selected.append((best_el, np.max(obj_val)))
        if k%save_iter == 0:
            if print_log:
                print(k)
                print(el_selected[-1])

            save_results(el_selected, filename)
    return el_selected

def run_random_greedy(k_to_select, ground_set, obj_fun, fname='tmp', rep_num=1, every_iter=10, k_to_start=1):

    filename = fname + '_greedy_random' + str(rep_num) + '.p'
    el_selected = [([], 0)]
    
    k_to_check = [i for i in list(range(k_to_start, k_to_select+1)) if (i-k_to_start)%every_iter == 0]
    
    for ki in k_to_check:
        best_k_plus_1 = []

        # for each ki, we have to run the random greedy algorithm
        for k in range(1, ki+1):

            best_k_el = best_k_plus_1
            obj_val = []

            el_list = list(ground_set - set(best_k_el))
            for el in el_list:
                k_plus_1 = list(best_k_el + [el])

                obj_val.append(obj_fun(k_plus_1))

            # get top k 
            top_idx = np.argsort(obj_val)[-ki:]
            top_el = [el_list[i] for i in top_idx]

            if np.random.uniform() < (1-len(top_el)/ki):
                best_el = []
            else:
                # select random el from top k
                best_el = [random.choice(top_el)]

            best_k_plus_1 = best_k_el + best_el
        el_selected.append((best_k_plus_1, obj_fun(best_k_plus_1)))

        save_results(el_selected, filename)

    return el_selected

def run_lazier_greedy(k_to_select, ground_set, obj_fun, eps, fname='tmp', rep_num=1, every_iter=10, k_to_start=1):

    filename = fname + '_greedy_lazier' + str(rep_num) + '.p'
    n = len(ground_set)
    el_selected = [([], 0)]
    
    k_to_check = [i for i in list(range(k_to_start, k_to_select+1)) if (i-k_to_start)%every_iter == 0]
    
    for ki in k_to_check:
        best_k_plus_1 = []
        sample_size = int(np.ceil(n*np.log(1/eps)/ki))
        sample_size = np.min([sample_size, n])

        # for each ki, we have to run the lazier greedy algorithm
        for k in range(1, ki+1):

            best_k_el = best_k_plus_1
            obj_val = []
            sample_size = np.min([len(list(ground_set - set(best_k_el))), sample_size])
            el_list = np.random.choice(list(ground_set - set(best_k_el)), size=sample_size, replace=False)
            for el in el_list:
                k_plus_1 = list(best_k_el + [el])

                obj_val.append(obj_fun(k_plus_1))

            best_el = [el_list[np.argmax(obj_val)]]
            best_k_plus_1 = best_k_el + best_el
        
        assert obj_fun(best_k_plus_1) == np.max(obj_val), "inconsistent"
        el_selected.append((best_k_plus_1, obj_fun(best_k_plus_1)))

        save_results(el_selected, filename)
    return el_selected

def run_local_search(k_to_select, ground_set, obj_fun, fname='tmp', k_to_start=1, every_iter=10):

    filename = fname + '_local_search.p'
    n = len(ground_set)
    el_selected = [([], 0)]

    k_to_check = [i for i in list(range(k_to_start, k_to_select+1)) if (i-k_to_start)%every_iter == 0]

    for ki in k_to_check:
        # get greedy result to start
        res = run_topk(ki, ground_set, obj_fun)
        S = [r[0][0] for r in res[1:]]
        S_not = [i for i in ground_set if i not in S]
        fS = obj_fun(S)
        
        change_bool = True
        num_changes = 0
        while change_bool:
            change_bool = False # no change has been made
            for el1 in S:
                for el2 in S_not:
                    S_prime = [s for s in S if s != el1] + [el2]
                    assert len(S_prime) == len(S), "len should be equal"

                    fS1 = obj_fun(S_prime)

                    if fS1 > fS:
                        S = S_prime
                        fS = obj_fun(S)
                        S_not = [i for i in ground_set if i not in S]
                        change_bool = True # change made
                        num_changes += 1
                        break # the for loop
        el_selected.append((S, obj_fun(S)))
        save_results(el_selected, filename)
    return el_selected

def run_opt_brute_force(k_to_select, ground_list, obj_fun, fname='tmp'):
    opt_val = [([], 0)]
    filename = fname + '_opt.p'

    for k in range(1,k_to_select+1):
        obj_val, combo = compute_opt(k, obj_fun, ground_list)
        opt_val.append((combo, obj_val))
        save_results(obj_val, filename)
    return opt_val

def run_opt_ip(k_to_select, graph, weights, fname='tmp'):
    opt_val = [([], 0)]
    filename = fname + '_ip.p'

    for k in range(1,k_to_select+1):
        selected_set, selected_el, status = ip_solver_coverage(graph, weights, k)
        opt_val.append((selected_set, selected_el))
        save_results(opt_val, filename)
    return opt_val

def run_topk(k_to_select, ground_set, obj_fun):
    # get starting set of data
    el_selected = [([], 0)]
    
    # marg contrib
    el_val = []
    el_list = list(ground_set)
    
    # compute marg contrib of all elements
    for el in el_list:
        k_plus_1 = [el]

        el_val.append(obj_fun(k_plus_1))

    el_selected = [([], 0)]
    for k in range(1, k_to_select+1):
        top_k = np.argsort(el_val)[-k:]

        el_selected.append(([top_k[0]],obj_fun(top_k)))

    return el_selected

### Helper functions. 

def compute_opt(k_to_select, obj, ground_set):
    res_opt = [([], 0)]
    for k in range(1,k_to_select+1):
        # compute all combos
        combo_list = combinations(ground_set, k)

        # loop through and compute obj_val and return highest
        max_obj_val = 0
        best_combo = []
        for combo in combo_list:
            obj_val = obj(combo)
            if obj_val > max_obj_val:
                max_obj_val = obj_val 
                best_combo = combo
        res_opt.append((list(best_combo), max_obj_val))
    return res_opt

# used for youtube data
def ip_solver_coverage(graph, weights, k):
        # I is the list of all nodes in the graph
    n = len(graph)
    d = graph.shape[1]
    N = set(range(n))
    D = set(range(d))

    m = Model()

    x = [m.add_var(var_type=BINARY) for i in N] # sets

    y = [m.add_var(var_type=BINARY) for i in D] # elements covered 

    m.objective = maximize(xsum(weights[i]*y[i] for i in D))

    m += xsum(x[i] for i in N) <= k

    for j in D:
        m += xsum(x[i]*graph[i][j] for i in N) >= y[j]

    status = m.optimize()

    selected_set = [i for i in N if x[i].x >= 0.99]

    selected_el = [i for i in D if y[i].x >= 0.99]
    
    return selected_set, len(selected_el), status
