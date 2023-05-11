"""Testing probabalistic labeling
Based on this paper
https://dl.acm.org/doi/abs/10.1145/1081870.1081890
Available here
https://www.researchgate.net/publication/221654189_Combining_partitions_by_probabilistic_label_aggregation
"""

import pandas as pd
import numpy as np

from keras_dec.functions import align_cluster_labels, cluster_acc, modal_labels
from multi.comparison import prob_lab_agg, accuracy_arr

import warnings

#Load multiple cluster labels
#Not resampled, new code
csv_path = "./output_full_mnist_def_100/dec_0.csv"
labels_path = "./output_full_mnist_def_100/dec_0_labels.csv"

df_dec = pd.read_csv(csv_path,index_col=0)
df_labels = pd.read_csv(labels_path,index_col=0)
labels = df_labels['labels_1']


#%%


# pla = prob_lab_agg(df_labels)
# pla2 = prob_lab_agg(df_dec)

# print(cluster_acc(labels,a)[0])
# print(cluster_acc(labels,a)[1])




def confusion_mtx(labels1,labels2):
    num = len(np.unique(labels1))
    w = np.zeros((num, num), dtype=np.int64)
    for i in range(len(labels1)):
        w[labels1[i], labels2[i]] += 1
    return w

#%%

dec_labels_pla = prob_lab_agg(df_dec)
accuracy_pla = cluster_acc(labels, dec_labels_pla)[0]

#%%Compare to mode dec labels
df_dec_aligned = pd.DataFrame(index=df_dec.index)
#Ignore a fragmentation performance warning, it was quicker doing it this way
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=Warning)
    for col in df_dec.columns:
        df_dec_aligned[col] = align_cluster_labels(labels,df_dec[col])

#Mode labels
dec_mode_labels = modal_labels(df_dec_aligned)

accuracy_mode_labels = cluster_acc(labels, dec_mode_labels)[0]

#%%Accuracy of individual runs
accuracy_array = accuracy_arr(df_dec,labels)
accuracy_mean = np.mean(accuracy_array)

