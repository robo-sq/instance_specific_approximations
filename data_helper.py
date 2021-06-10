""" This file contains code to read in raw datasets. """

import pickle
import csv
import math
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from scipy.io import mmread 

DATA_PATH = 'data/'

def distance_meters(dat1, dat2):
    lat1 = dat1[0]
    lon1 = dat1[1]
    
    lat2 = dat2[0]
    lon2 = dat2[1]
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat/2))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2))**2
    c = 2 * math.atan2( np.sqrt(a), np.sqrt(1-a) )
    d = 6373 * c
    return d

def load_cifar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')
        
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
        
    return features, labels

def get_data(data):
    if data == 'youtube':
        youtube_g = open_data('youtube_graph.p')
        return youtube_g
    elif data == 'movie':
        # read in file
        ratings_list = [i.strip().split("::") for i in open(DATA_PATH + 'ratings.dat', 'r').readlines()]
        ratings_df = pd.DataFrame(ratings_list, dtype = int)
        R_df = ratings_df.pivot(index = 0, columns = 1, values = 2).fillna(0)

        # normalize by user
        R = R_df.as_matrix()
        user_ratings_mean = np.mean(R, axis = 1)
        R_demeaned = R - user_ratings_mean.reshape(-1, 1)

        # matrix completion
        U, sigma, Vt = svds(R_demeaned, k = 50)
        sigma = np.diag(sigma)
        movie_mat = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        movie_mat = np.where(movie_mat<0, 0, movie_mat)
        
        return movie_mat
    elif data == 'facebook':
        fb_mat = mmread(DATA_PATH + 'socfb-Caltech36.mtx')
        fb_arr = np.array(fb_mat.todense())
        for i in range(len(fb_arr)):
            for j in range(len(fb_arr)):
                if (i < j) and (fb_arr[i,j] != 0):
                    val = np.random.uniform(1,2)
                    fb_arr[i,j] = val
                    fb_arr[j,i] = val
        return fb_arr
    elif data == 'citation':
        file_reader = csv.reader(open(DATA_PATH + 'ca-HepTh.txt'), delimiter='\t')
        edge_list = []
        for fi in file_reader:
            if fi[0][0] != '#':
                edge_list.append([int(i) for i in fi])

        # get unique list of ids in node_list
        node_list = []
        for i in edge_list:
            node_list += i
        node_set = set(node_list)
        node_set = sorted(node_set)
        node_list = list(node_set)


        collab_g = np.zeros((len(node_list), len(node_list)))
        # capture edges

        for edge in edge_list:
            collab_g[node_list.index(edge[0]), node_list.index(edge[1])] = 1
            collab_g[node_list.index(edge[1]), node_list.index(edge[0])] = 1

        return collab_g
    elif data == 'census':
        df = pd.read_csv(DATA_PATH + 'adult.data', header=None)
        
        # drop question marks
        df = df.replace(to_replace=' ?', value=np.nan).dropna()
        
        # convert continuous to categorical
        df['Age'] = ''
        df['Hours'] = ''
        df['Fin1'] = ''
        df['Fin2'] = ''
        df['Y'] = 0
        
        df.loc[df[0]<=20, ['Age']] = '<=20'
        df.loc[(df[0]>20) & (df[0]<=30), ['Age']] = '20-30'
        df.loc[(df[0]>30) & (df[0]<=40), ['Age']] = '30-40'
        df.loc[(df[0]>40) & (df[0]<=50), ['Age']] = '40-50'
        df.loc[(df[0]>50), ['Age']] = '>50'

        df.loc[df[12]<=40, ['Hours']] = '<=40'
        df.loc[df[12]>40, ['Hours']] = '>40'

        df.loc[df[10]<=1075, ['Fin1']] = '<=1075'
        df.loc[df[10]>1075, ['Fin1']] = '>1075'

        df.loc[df[11]<=85, ['Fin2']] = '<=85'
        df.loc[df[11]>85, ['Fin2']] = '>85'

        df.loc[df[14] == ' >50K', ['Y']] = 1
        
        df_cat = df.drop([0, 2, 4, 10, 11, 12, 14, 'Y'], axis=1)
        df_cat = pd.get_dummies(df_cat)
        
        y = df.Y.values
        X = df_cat.as_matrix()
    elif data == 'uber':
        df = pd.read_csv(DATA_PATH + 'uber-pickups-in-new-york-city/uber-raw-data-apr14.csv')

        # just first 1k rows
        n_samples = 1000
        df = df.head(n_samples)

        dat = np.zeros((n_samples, 2))
        dat[:,0] = df['Lat']
        dat[:,1] = df['Lon']

        K = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = np.exp(-distance_meters(dat[i], dat[j])**2/500**2)
        
        return K
    elif data =='sensor':
        arr = open_data('sensor_temp_bin.p')
        return arr
