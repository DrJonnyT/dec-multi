"""Testing probabalistic labeling
Based on this paper
https://dl.acm.org/doi/abs/10.1145/1081870.1081890
Available here
https://www.researchgate.net/publication/221654189_Combining_partitions_by_probabilistic_label_aggregation
"""

import pandas as pd
import numpy as np
import datetime as dt

from keras_dec.functions import align_cluster_labels, cluster_acc, modal_labels
from multi.comparison import prob_lab_agg, accuracy_arr

import warnings

#Load multiple cluster labels
output_folder = "./output_unbalanced_10000_5000/"
#An array of the number of copies of each digit to use
n_digits_array = [100,250,500,750,1000,2500,5000,7500,10000,25000,50000,70000]


columns = ['acc_mean','acc_stdev','acc_mode','acc_pla','time_mode_s','time_pla_s']

#Process results for 10k iterations
df_agg_data_10k = pd.DataFrame(index=n_digits_array,columns=columns)
labels_mode_10k = []
labels_pla_10k = []

for n_digits in df_agg_data_10k.index:
    print(f"Running for {n_digits} digits")
    csv_path = output_folder + f"dec_{n_digits}.csv"
    labels_path = output_folder + f"dec_{n_digits}_labels.csv"
    
    try:
        df_dec = pd.read_csv(csv_path,index_col=0)
        df_labels = pd.read_csv(labels_path,index_col=0)
    except:
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



#%%Repeat for 100 iterations in the clustering
output_folder = "./output_unbalanced_100_50/"

df_agg_data_100 = pd.DataFrame(index=n_digits_array,columns=columns)
labels_mode_100 = []
labels_pla_100 = []

for n_digits in df_agg_data_100.index:
    print(f"Running for {n_digits} digits")
    csv_path = output_folder + f"dec_{n_digits}.csv"
    labels_path = output_folder + f"dec_{n_digits}_labels.csv"
    
    try:
        df_dec = pd.read_csv(csv_path,index_col=0)
        df_labels = pd.read_csv(labels_path,index_col=0)
    except:
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






#%%
# t_start = dt.datetime.now()
# dec_labels_pla = prob_lab_agg(df_dec)
# t_end = dt.datetime.now()
# t_delta = t_end - t_start


# accuracy_pla = cluster_acc(labels, dec_labels_pla)[0]

# #%%Compare to mode dec labels
# df_dec_aligned = pd.DataFrame(index=df_dec.index)
# #Ignore a fragmentation performance warning, it was quicker doing it this way
# with warnings.catch_warnings():
#     warnings.simplefilter(action='ignore', category=Warning)
#     for col in df_dec.columns:
#         df_dec_aligned[col] = align_cluster_labels(labels,df_dec[col])

# #Mode labels
# dec_mode_labels = modal_labels(df_dec_aligned)

# accuracy_mode_labels = cluster_acc(labels, dec_mode_labels)[0]

# #%%Accuracy of individual runs
# accuracy_array = accuracy_arr(df_dec,labels)
# accuracy_mean = np.mean(accuracy_array)

