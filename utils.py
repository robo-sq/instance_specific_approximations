from data_helper import DATA_PATH
import pickle
import numpy as np
import random

RESULTS_PATH = 'results/'


def save_results(var, path):
    pickle.dump(var, open(RESULTS_PATH + path, "wb" ))

def open_results(path):
	return pickle.load(open(RESULTS_PATH + path, "rb" ))

def open_data(path):
    return pickle.load(open(DATA_PATH + path, "rb" ))

def save_data(var, path):
    pickle.dump(var, open(DATA_PATH + path, "wb" ))

def sample_data(data, n):
    # return n data points
    sampled_idx = random.sample(range(data.shape[0]), n)
    
    return data[sampled_idx, :]