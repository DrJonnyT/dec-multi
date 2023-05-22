import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Parameters file
import params.params_no_resample as params
#import params.testing as params

from mnist.plot import plot_mnist_10x10
from multi.comparison import mean_rand_index, rand_index_arr, accuracy_arr
from keras_dec.functions import align_cluster_labels, cluster_acc, modal_labels


#Load settings from params.py file
#An array of the number of copies of each digit to use
n10_array = params.n10_array
#The number of times to run kmeans
n_runs = params.n_runs
#Number of clusters
n_clusters = params.n_clusters
#Output folder
csv_folder = params.csv_folder
#Resample flag
resample=params.resample


#Empty arrays for results
accuracy_kmeans_mean = []
accuracy_kmeans_std = []
accuracy_dec_mean = []
accuracy_dec_std = []
accuracy_kmeans_avg_labels = []
accuracy_dec_avg_labels = []


for n10 in n10_array:
    #Load kmeans and dec data
    df_kmeans = pd.read_csv(csv_folder+f"kmeans_{n10}.csv",index_col=0)
    df_kmeans_labels = pd.read_csv(csv_folder+f"kmeans_{n10}_labels.csv",index_col=0)
    df_dec = pd.read_csv(csv_folder+f"dec_{n10}.csv",index_col=0)
    df_dec_labels = pd.read_csv(csv_folder+f"dec_{n10}_labels.csv",index_col=0)
    
    #Calculate accuracy   
    accuracy_kmeans = accuracy_arr(df_kmeans,df_kmeans_labels['labels_1'])
    accuracy_dec = accuracy_arr(df_dec,df_dec_labels['labels_1'])
    
    #Append
    accuracy_kmeans_mean.append(accuracy_kmeans.mean())
    accuracy_kmeans_std.append(accuracy_kmeans.std())
    accuracy_dec_mean.append(accuracy_dec.mean())
    accuracy_dec_std.append(accuracy_dec.std())
    
    
    #Average the cluster labels
    df_kmeans_aligned = pd.DataFrame(index=df_kmeans.index)
    df_dec_aligned = pd.DataFrame(index=df_dec.index)
    for col in df_kmeans.columns:
        df_kmeans_aligned[col] = align_cluster_labels(df_kmeans_labels.iloc[:,0],df_kmeans[col])
    for col in df_dec.columns:
        df_dec_aligned[col] = align_cluster_labels(df_dec_labels.iloc[:,0],df_dec[col])

    #Accuracy of modal labels
    kmeans_mode_labels = modal_labels(df_kmeans_aligned)
    dec_mode_labels = modal_labels(df_dec_aligned)
    kmeans_true_labels = df_kmeans_labels.iloc[:,0]
    dec_true_labels = df_kmeans_labels.iloc[:,0]
    
    accuracy_kmeans_avg_labels.append(cluster_acc(df_kmeans_labels.iloc[:,0], kmeans_mode_labels)[0])
    accuracy_dec_avg_labels.append(cluster_acc(df_dec_labels.iloc[:,0], dec_mode_labels)[0])
    
    
    
#Convert to np arrays
accuracy_kmeans_mean = np.array(accuracy_kmeans_mean)
accuracy_kmeans_std = np.array(accuracy_kmeans_std)
accuracy_dec_mean = np.array(accuracy_dec_mean)
accuracy_dec_std = np.array(accuracy_dec_std)
accuracy_kmeans_avg_labels = np.array(accuracy_kmeans_avg_labels)
accuracy_dec_avg_labels = np.array(accuracy_dec_avg_labels)

#Plot results as mean with error bar
fig,ax = plt.subplots()
ax.errorbar(n10_array,accuracy_kmeans_mean,accuracy_kmeans_std,c='k',fmt='-o')
ax.errorbar(n10_array,accuracy_dec_mean,accuracy_kmeans_std,c='tab:blue',fmt='-o')

#Plot with shaded error bar
fig,ax = plt.subplots()
ax.plot(n10_array,accuracy_kmeans_mean,c='k',label='kmeans')
ax.fill_between(n10_array, accuracy_kmeans_mean-accuracy_kmeans_std, accuracy_kmeans_mean+accuracy_kmeans_std,
                 color='k',alpha=0.25)
ax.plot(n10_array,accuracy_dec_mean,c='tab:blue',label='DEC')
ax.fill_between(n10_array, accuracy_dec_mean-accuracy_dec_std, accuracy_dec_mean+accuracy_dec_std,
                 color='tab:blue',alpha=0.25,)
ax.set_xscale('log')
ax.set_xlabel("Number of each digit")
ax.set_ylabel("Accuracy")
ax.set_title("Cluster label accuracy")
ax.legend()
ax.plot(n10_array,accuracy_kmeans_avg_labels,c='k',label='kmeans avg labels')
ax.plot(n10_array,accuracy_dec_avg_labels,c='tab:blue',label='DEC avg labels')
