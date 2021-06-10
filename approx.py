""" This file contains novel algorithms from the paper and benchmark algorithms 
that were used for experiments on large and small instances. 
For large instances, benchmarks are Marginal, upper bound on Curvature, Top-k. 
For small instances, benchmarks are dynamic submodular sharpness and 
curvature (brute force). """

import numpy as np
from itertools import combinations 

####################################################
########     METHODS PROPOSED IN PAPER.     ########
####################################################

# compute approximation for greedy given method 1
def compute_method1(k, obj, ground_set, res_gr):
    V = construct_ordering(obj, ground_set)

    # rearrange ordering
    V_cov = []
    for i in V:
        V_cov += int(i[1])*[i[0]]

    opt_cov_list = []
    approx = []
    idx = 0
    for ki in range(1, k+1):
        i = 0
        idx = 0
        while (i < ki) and (idx < len(V_cov)):
            i += round(1/V_cov[idx], 5)
            idx += 1 # the fraction to be added next
        opt_cov_list.append(idx)
        approx.append(res_gr[ki][1]/idx)
    return opt_cov_list, approx

# compute approximation for greedy given method 3
def compute_method3(k, obj, ground_set, res_gr):
    V = construct_ordering(obj, ground_set)
    opt_ub_list = []
    approx = []
    idx = 0
    istar = 0
    u = 0
    res_val = 0
    for ki in range(1,k+1):
        while (istar < len(V)) and (res_val + np.sum([v[1] for v in V[idx:istar+1]]) < V[istar][0]):
            istar += 1
        if (istar == len(V)):
            u += res_val + np.sum([v[1] for v in V[idx:istar+1]])
            res_val = 0 
        elif (res_val + np.sum([v[1] for v in V[idx:istar]])) > V[istar][0]:
            u += res_val + np.sum([v[1] for v in V[idx:istar]])
            res_val = 0
        else:
            u += V[istar][0]
            res_val =  (res_val + np.sum([v[1] for v in V[idx:istar+1]]) - V[istar][0])
            istar += 1
        idx = istar
        opt_ub_list.append(u)
        approx.append(res_gr[ki][1]/u)
    return opt_ub_list, approx

# compute approximation for greedy given dual
def compute_dual(k, obj, ground_set, res_gr, S_idx):

    o_bound_arr = np.zeros((len(S_idx), k))
    # first compute upper bound of opt across all sets in S_idx
    for idx in S_idx:

        # get greedy solution
        S = [i[0][0] for i in res_gr[1:idx+1]]

        # get new obj function
        obj_marg = obj_marg_wrapper(obj, S)

        # get a scalar bound for f_S(O), O is of size k and S is of size idx
        idx_v = S_idx.index(idx)

        o_bound_list, _ = compute_method3(k, obj_marg, ground_set, res_gr)
        o_bound_arr[idx_v] = o_bound_list
        
    # get the min upper bound of opt for each k
    approx = []
    opt_ub_list = []
    for ki in range(1,k+1):
        o_bound_list = []
        fSO = o_bound_arr[:, ki-1]
        for i in S_idx:
            idx_v = S_idx.index(i)
            # get greedy solution S of size i and marginal contribution
            # O to S, f_S(O)
            o_bound_list.append(res_gr[int(i)][1] + fSO[idx_v])
        opt_ub_list.append(np.min(o_bound_list))
        approx.append(res_gr[ki][1]/np.min(o_bound_list))
    return opt_ub_list, approx


###############################################
########      BENCHMARK METHODS.       ########
###############################################

# compute curvature approximation via brute force
def compute_curvature_brute_k(k_to_select, ground_set, obj):
    
    approx = []
    for k in range(1, k_to_select):
        max_curv = 0
        for el in ground_set:
            denom = obj([el])
            ground_set_new = [i for i in ground_set if i != el]
            if denom > 0:
                combo_list = combinations(ground_set_new, k)
                for combo in combo_list:
                    S = list(combo)
                    
                    curv = 1-(obj(S + [el])-obj(S))/denom
                    
                    if curv > max_curv:
                        max_curv = curv
        approx.append(1-np.e**(-max_curv)/max_curv)

    return approx

# compute curvature upper bound 
def compute_curvature_ub(k, ground_set, obj):

    # first find a* by 
    el, _ = find_a_star(ground_set, obj)
    denom = obj([el])
    ground_set_new = [i for i in ground_set if i != el]
    
    res_curv = []
    S = []
    for _ in range(k):
        
        # find max 
        max_c = 0
        for i in ground_set_new:
            new_S = S + [i]
            
            
            c = 1-(obj(new_S + [el]) - obj(new_S))/denom
            
            if c > max_c:
                max_el = i
                max_c = c
        S = S + [max_el]
        res_curv.append((S, (1 - np.e**(-max_c))/max_c))
            
        ground_set_new = [i for i in ground_set_new if i != max_el]
        if max_c > 0.99:
            break
    
    d = k - len(S)
    if d > 0:
        curv_add_on = d*[res_curv[-1]]
        res_curv += curv_add_on
            
    return res_curv

# compute dynamic submodular sharpness
def compute_s_sharpness_dyn(k, O, OPT, ground_set, obj):
    
    ci_list = []
    thetai_list = []
    for ki in range(1,k+1):
        c_list = np.linspace(1,3,201)
        max_apx = 0
        c_cache =1
        theta_cache = 0
        for c in c_list:
            theta = compute_theta_k_sub(c, ki, k, O, OPT, ground_set, obj)
            apx = 1-(1-theta/c)**(1/theta)
            if apx >= max_apx:
                max_apx = apx
                c_cache = c
                theta_cache = theta
            else:
                break
        ci_list.append(c_cache)
        thetai_list.append(theta_cache)
    # compute apx
    prod_acc = 1
    for idx, theta in enumerate(thetai_list):
        prod_acc -= theta/(ci_list[idx]*k)
        if idx == len(thetai_list) - 1:
            prod_acc = prod_acc**(1/theta)
        else:
            prod_acc = prod_acc**(thetai_list[idx+1]/theta)
    
    return 1-prod_acc

# compute marginal benchmark
def compute_marginal(k, res_gr):
    
    approx = []
    opt_ub_list = []
    for ki in range(1,k+1):
        opt_ub_k = []
        for i in range(0, len(res_gr)):
            for j in range(i+1, len(res_gr)):
                opt_ub_k.append((res_gr[j][1] - ((1-1/ki)**(j-i))*res_gr[i][1])/(1-(1-1/ki)**(j-i)))
        opt_ub_list.append(np.min(opt_ub_k))
        approx.append(res_gr[ki][1]/np.min(opt_ub_k))
    return opt_ub_list, approx

# compute top-k benchmark
def compute_topk(k, obj, ground_set, res_gr):
    fa = []
    for i in ground_set:
        fa.append(obj([i]))
    fa.sort(reverse=True)

    approx = []
    opt_ub_list = []
    for ki in range(1,k+1):
        approx.append(res_gr[ki][1]/np.sum(fa[:ki]))
        opt_ub_list.append(np.sum(fa[:ki]))
    return opt_ub_list, approx


####################################################
########          HELPER FUNCTIONS.         ########
####################################################

# compute ordering of singletons
def construct_ordering(obj, ground_set):
    values = []
    for a in ground_set:
        values.append(obj([a]))
    T = np.argsort(values)[::-1]
    S = []
    V = []
    for i in range(len(ground_set)):
        if i == 0:
            w = obj([T[i]])
        else:
            w = obj(S + [T[i]]) - obj(S)
        S.append(T[i])
        if w > 0:
            V.append((obj([T[i]]), w))
    return V

# get new objective function that is the 
# marginal contribution function wrt to S (el_selected)
def obj_marg_wrapper(obj, el_selected):

    def obj_fun(new_el):

        return obj(new_el + el_selected) - obj(el_selected)

    return obj_fun

# helper function for dynamic submodular sharpness
def compute_theta_k_sub(c, ki, k, O, OPT, ground_set, obj):
    min_theta = 1
    combo_list = combinations(ground_set, ki)

    for combo in combo_list:
        S = list(combo)
        E = [o for o in O if o not in S]
        fS = obj(S)
        if len(E) > 0:
            W2 = np.max([obj(S + [e])-fS for e in E])
            W = OPT - fS
            if W > 0:
                theta = np.log(k*c*W2/W)/np.log(OPT/W)
                if theta < min_theta:
                    min_theta = theta
    return min_theta

def find_a_star(ground_set, obj):
    
    min_frac = 1
    min_el = 0
    fN = obj(list(ground_set))
    for el in ground_set:
        N_no_i = [i for i in ground_set if i != el]
        new_frac = (fN - obj(N_no_i))/obj([el])
        
        if new_frac < min_frac:
            min_frac = new_frac
            min_el = el
        
    return min_el, 1-min_frac