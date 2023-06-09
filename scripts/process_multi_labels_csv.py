"""Aggregate cluster labels
Either based on the mode label, or probabalistic label aggregation, based on
this paper:
https://dl.acm.org/doi/abs/10.1145/1081870.1081890
Available here:
https://www.researchgate.net/publication/221654189_Combining_partitions_by_probabilistic_label_aggregation
"""

import pandas as pd
import numpy as np
import datetime as dt
import sys
sys.path.append('../')  #Add parent folder to path so imports work

from keras_dec.functions import align_cluster_labels, cluster_acc, modal_labels
from multi.comparison import prob_lab_agg, accuracy_arr

import warnings

#Load multiple cluster labels
output_folder = "../output_unbalanced_10000_5000/"
#An array of the number of copies of each digit to use
n_digits_array = [100,250,500,750,1000,2500,5000,7500,10000,25000,50000,70000]

columns = ['acc_mean','acc_stdev','acc_mode','acc_pla','time_mode_s','time_pla_s']

#Process results for 10k iterations
df_agg_data_10k = pd.DataFrame(index=n_digits_array,columns=columns)
labels_mode_10k = []
labels_pla_10k = []

for n_digits in df_agg_data_10k.index:
    print(f"Running for {n_digits} digits",flush=True)
    csv_path = output_folder + f"dec_{n_digits}.csv"
    labels_path = output_folder + f"dec_{n_digits}_labels.csv"
    
    try:
        #Load just the first 25 columns
        df_dec = pd.read_csv(csv_path,index_col=0).iloc[:,0:25]
        df_labels = pd.read_csv(labels_path,index_col=0).iloc[:,0:25]        
    except:
        print(f"Unable to load for {n_digits} digits",flush=True)
        continue
    true_labels = df_labels['labels_1']

    #Calc PLA labels
    t_start = dt.datetime.now()
    dec_labels_pla = prob_lab_agg(df_dec)
    t_end = dt.datetime.now()
    df_agg_data_10k.loc[n_digits]['time_pla_s'] = (t_end-t_start).total_seconds()
    df_agg_data_10k.loc[n_digits]['acc_pla'] = cluster_acc(true_labels, dec_labels_pla)[0]
    labels_pla_10k.append(dec_labels_pla)
    
    #Calc mode labels
    t_start = dt.datetime.now()
    df_dec_aligned = pd.DataFrame(index=df_dec.index)
    #Ignore a fragmentation performance warning, it was quicker doing it this way
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=Warning)
        for col in df_dec.columns:
            df_dec_aligned[col] = align_cluster_labels(df_dec.iloc[:,0],df_dec[col])

    #Mode labels
    dec_labels_mode = modal_labels(df_dec_aligned)
    t_end = dt.datetime.now()
    df_agg_data_10k.loc[n_digits]['time_mode_s'] = (t_end-t_start).total_seconds()
    df_agg_data_10k.loc[n_digits]['acc_mode'] = cluster_acc(true_labels, dec_labels_mode)[0]
    labels_mode_10k.append(dec_labels_mode)
    
    #Mean and stdev of single runs
    acc_arr = accuracy_arr(df_dec,true_labels)
    df_agg_data_10k.loc[n_digits]['acc_mean'] = np.mean(acc_arr)
    df_agg_data_10k.loc[n_digits]['acc_stdev'] = np.std(acc_arr)

#Export data
df_agg_data_10k.to_csv(output_folder + 'df_agg_data_10k.csv')
df_labels_mode_10k = pd.DataFrame(labels_mode_10k).T
df_labels_mode_10k.columns = n_digits_array
df_labels_mode_10k.index = df_dec.index
df_labels_mode_10k.to_csv(output_folder + 'df_labels_mode_10k.csv')
df_labels_pla_10k = pd.DataFrame(labels_pla_10k).T
df_labels_pla_10k.columns = n_digits_array
df_labels_pla_10k.index = df_dec.index
df_labels_pla_10k.to_csv(output_folder + 'df_labels_pla_10k.csv')

#%%Repeat for 100 iterations in the clustering
output_folder = "../output_unbalanced_100_50/"

df_agg_data_100 = pd.DataFrame(index=n_digits_array,columns=columns)
labels_mode_100 = []
labels_pla_100 = []

for n_digits in df_agg_data_100.index:
    print(f"Running for {n_digits} digits",flush=True)
    csv_path = output_folder + f"dec_{n_digits}.csv"
    labels_path = output_folder + f"dec_{n_digits}_labels.csv"
    
    try:
        #Load just the first 25 columns
        df_dec = pd.read_csv(csv_path,index_col=0).iloc[:,0:25]
        df_labels = pd.read_csv(labels_path,index_col=0).iloc[:,0:25]        
    except:
        print(f"Unable to load for {n_digits} digits",flush=True)
        continue
    true_labels = df_labels['labels_1']
    
    #Calc PLA labels
    t_start = dt.datetime.now()
    dec_labels_pla = prob_lab_agg(df_dec)
    t_end = dt.datetime.now()
    df_agg_data_100.loc[n_digits]['time_pla_s'] = (t_end-t_start).total_seconds()
    df_agg_data_100.loc[n_digits]['acc_pla'] = cluster_acc(true_labels, dec_labels_pla)[0]
    labels_pla_100.append(dec_labels_pla)
    
    #Calc mode labels
    t_start = dt.datetime.now()
    df_dec_aligned = pd.DataFrame(index=df_dec.index)
    #Ignore a fragmentation performance warning, it was quicker doing it this way
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=Warning)
        for col in df_dec.columns:
            df_dec_aligned[col] = align_cluster_labels(df_dec.iloc[:,0],df_dec[col])

    #Mode labels
    dec_labels_mode = modal_labels(df_dec_aligned)
    t_end = dt.datetime.now()
    df_agg_data_100.loc[n_digits]['time_mode_s'] = (t_end-t_start).total_seconds()
    df_agg_data_100.loc[n_digits]['acc_mode'] = cluster_acc(true_labels, dec_labels_mode)[0]
    labels_mode_100.append(dec_labels_mode)
    
    #Mean and stdev of single runs
    acc_arr = accuracy_arr(df_dec,true_labels)
    df_agg_data_100.loc[n_digits]['acc_mean'] = np.mean(acc_arr)
    df_agg_data_100.loc[n_digits]['acc_stdev'] = np.std(acc_arr)


#Export data
df_agg_data_100.to_csv(output_folder + 'df_agg_data_100.csv')
df_labels_mode_100 = pd.DataFrame(labels_mode_100).T
df_labels_mode_100.columns = n_digits_array
df_labels_mode_100.index = df_dec.index
df_labels_mode_100.to_csv(output_folder + 'df_labels_mode_100.csv')
df_labels_pla_100 = pd.DataFrame(labels_pla_100).T
df_labels_pla_100.columns = n_digits_array
df_labels_pla_100.index = df_dec.index
df_labels_pla_100.to_csv(output_folder + 'df_labels_pla_100.csv')



#%%Repeat for kmeans
output_folder = "../output_unbalanced_kmeans/"

df_agg_data_kmeans = pd.DataFrame(index=n_digits_array,columns=columns)
labels_mode_kmeans = []
labels_pla_kmeans = []

for n_digits in df_agg_data_kmeans.index:
    print(f"Running for {n_digits} digits",flush=True)
    csv_path = output_folder + f"kmeans_{n_digits}.csv"
    labels_path = output_folder + f"kmeans_{n_digits}_labels.csv"
    
    try:
        #Load just the first 25 columns
        df_kmeans = pd.read_csv(csv_path,index_col=0).iloc[:,0:25]
        df_labels = pd.read_csv(labels_path,index_col=0).iloc[:,0:25]        
    except:
        print(f"Unable to load for {n_digits} digits",flush=True)
        continue
    true_labels = df_labels['labels_1']
    
    #Calc PLA labels
    t_start = dt.datetime.now()
    kmeans_labels_pla = prob_lab_agg(df_kmeans)
    t_end = dt.datetime.now()
    df_agg_data_kmeans.loc[n_digits]['time_pla_s'] = (t_end-t_start).total_seconds()
    df_agg_data_kmeans.loc[n_digits]['acc_pla'] = cluster_acc(true_labels, kmeans_labels_pla)[0]
    labels_pla_kmeans.append(kmeans_labels_pla)
    
    #Calc mode labels
    t_start = dt.datetime.now()
    df_kmeans_aligned = pd.DataFrame(index=df_kmeans.index)
    #Ignore a fragmentation performance warning, it was quicker doing it this way
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=Warning)
        for col in df_kmeans.columns:
            df_kmeans_aligned[col] = align_cluster_labels(df_kmeans.iloc[:,0],df_kmeans[col])

    #Mode labels
    kmeans_labels_mode = modal_labels(df_kmeans_aligned)
    t_end = dt.datetime.now()
    df_agg_data_kmeans.loc[n_digits]['time_mode_s'] = (t_end-t_start).total_seconds()
    df_agg_data_kmeans.loc[n_digits]['acc_mode'] = cluster_acc(true_labels, kmeans_labels_mode)[0]
    labels_mode_kmeans.append(kmeans_labels_mode)
    
    #Mean and stdev of single runs
    acc_arr = accuracy_arr(df_kmeans,true_labels)
    df_agg_data_kmeans.loc[n_digits]['acc_mean'] = np.mean(acc_arr)
    df_agg_data_kmeans.loc[n_digits]['acc_stdev'] = np.std(acc_arr)


#Export data
df_agg_data_kmeans.to_csv(output_folder + 'df_agg_data_kmeans.csv')
df_labels_mode_kmeans = pd.DataFrame(labels_mode_kmeans).T
df_labels_mode_kmeans.columns = n_digits_array
df_labels_mode_kmeans.index = df_dec.index
df_labels_mode_kmeans.to_csv(output_folder + 'df_labels_mode_kmeans.csv')
df_labels_pla_kmeans = pd.DataFrame(labels_pla_kmeans).T
df_labels_pla_kmeans.columns = n_digits_array
df_labels_pla_kmeans.index = df_dec.index
df_labels_pla_kmeans.to_csv(output_folder + 'df_labels_pla_kmeans.csv')