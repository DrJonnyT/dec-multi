"""
Explore the impact of number of runs of DEC using the full MNIST dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
sys.path.append('../')  #Add parent foldet to path so imports work



#Parameters file
import params.params_full_mnist as params

from mnist.plot import plot_mnist_10x10
from multi.comparison import mean_rand_index, rand_index_arr, accuracy_arr
from keras_dec.functions import align_cluster_labels, cluster_acc, modal_labels



#Load csv
# csv_path = params.csv_folder + "dec_0.csv"
# labels_path = params.csv_folder + "dec_0_labels.csv"

# #6313 of each digit
# csv_path = "./output_no_resample_100/dec_6313.csv"
# labels_path = "./output_no_resample_100/dec_6313_labels.csv"

# #Redoing 613 of each digit
# csv_path = "./output_no_resample_again/dec_631.csv"
# labels_path = "./output_no_resample_again/dec_631_labels.csv"


# #10000 randomly sampled (ie unbalanced) digits
# csv_path = "./output_balanced/dec_10000.csv"
# labels_path = "./output_balanced/dec_10000_labels.csv"


#Not resampled, new code
csv_path = "../output_unbalanced_1000_500/dec_250.csv"
labels_path = "../output_unbalanced_1000_500/dec_250_labels.csv"


# #Old resampled data
# csv_path = "./output_resample/kmeans_6313.csv"
# labels_path = "./output_resample/kmeans_6313_labels.csv"




df_dec = pd.read_csv(csv_path,index_col=0)
df_labels = pd.read_csv(labels_path,index_col=0)


#Select only first 25 columns
# df_dec = df_dec.iloc[:,0:25]
# df_labels = df_labels.iloc[:,0:25]



#All label columns are the same
labels = df_labels['labels_1']

#Align results
df_dec_aligned = pd.DataFrame(index=df_dec.index)

#Ignore a fragmentation performance warning, it was quicker doing it this way
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=Warning)
    for col in df_dec.columns:
        df_dec_aligned[col] = align_cluster_labels(labels,df_dec[col])    
        #df_dec_aligned[col] = align_cluster_labels(df_dec.iloc[:,0],df_dec[col])    


#Mode accruacy
dec_mode_labels = modal_labels(df_dec_aligned)
acc = cluster_acc(labels, dec_mode_labels)[0]



#Array of accruacy of individual runs
ds_acc = pd.Series(index=df_dec_aligned.columns)
ds_prec = pd.Series(index=df_dec_aligned.columns)
for col in df_dec_aligned.columns:
    ds_acc[col] = cluster_acc(labels, df_dec_aligned[col])[0]
    ds_prec[col] = cluster_acc(dec_mode_labels, df_dec_aligned[col])[0]
    

#For some reason in the full mnist data you are getting way too many 3's in the clusters
#So that is limiting the accuracy.
    
    