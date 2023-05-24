"""Plot aggregated cluster labels produced by process_multi_labels_csv.py"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import sys
sys.path.append('../')  #Add parent folder to path so imports work


#Load aggregated labels for 10000 iterations
output_folder = "../output_unbalanced_10000_5000/"
df_agg_data_10k = pd.read_csv(output_folder + 'df_agg_data_10k.csv',
                              index_col = 0)


#Load aggregated labels for 100 iterations
output_folder = "../output_unbalanced_100_50/"
df_agg_data_100 = pd.read_csv(output_folder + 'df_agg_data_100.csv',
                              index_col = 0)

#Load aggregated labels for kmeans
output_folder = "../output_unbalanced_kmeans/"
df_agg_data_kmeans = pd.read_csv(output_folder + 'df_agg_data_kmeans.csv',
                              index_col = 0)



#Plot each of the 3 clustering types
fig, axs = plt.subplots(3,1,figsize=(5,8),sharex=True,sharey=True)
axs=axs.ravel()
[ax.set_ylabel('Accuracy') for ax in axs]
axs[0].set_xlabel('Number of samples')
axs[0].set_xscale('log')

#Plot 10k iterations DEC data
df_agg_data_10k.plot(y='acc_mean',ax=axs[0],c='tab:blue',label='Single runs',linestyle='--')
axs[0].fill_between(df_agg_data_10k.index,
                df_agg_data_10k['acc_mean']+df_agg_data_10k['acc_stdev'], 
                df_agg_data_10k['acc_mean']-df_agg_data_10k['acc_stdev'], color='tab:blue', alpha=0.2)

df_agg_data_10k.plot(y='acc_mode',ax=axs[0],c='midnightblue', label='Mode labels',marker='o')
df_agg_data_10k.plot(y='acc_pla',ax=axs[0],c='royalblue',label='PLA',marker='s')
axs[0].set_title('DEC 10000 iterations')
axs[0].legend(loc='lower right')


#Plot 100 iterations DEC data
df_agg_data_100.plot(y='acc_mean',ax=axs[1],c='tab:red',label='Single runs',linestyle='--')
axs[1].fill_between(df_agg_data_100.index,
                df_agg_data_100['acc_mean']+df_agg_data_100['acc_stdev'], 
                df_agg_data_100['acc_mean']-df_agg_data_100['acc_stdev'], color='tab:red', alpha=0.2)

df_agg_data_100.plot(y='acc_mode',ax=axs[1],c='maroon',label='Mode labels',marker='o')
df_agg_data_100.plot(y='acc_pla',ax=axs[1],c='orange',label='PLA',marker='s')
axs[1].set_title('DEC 100 iterations')

#Plot kmeans data
df_agg_data_kmeans.plot(y='acc_mean',ax=axs[2],c='grey',label='Single runs',linestyle='--')
axs[2].fill_between(df_agg_data_kmeans.index,
                df_agg_data_kmeans['acc_mean']+df_agg_data_kmeans['acc_stdev'], 
                df_agg_data_kmeans['acc_mean']-df_agg_data_kmeans['acc_stdev'], color='grey', alpha=0.2)

df_agg_data_kmeans.plot(y='acc_mode',ax=axs[2],c='black',label='Mode labels',marker='o')
df_agg_data_kmeans.plot(y='acc_pla',ax=axs[2],c='silver',label='PLA',marker='s')
axs[2].set_title('kmeans')

axs[-1].set_xlabel("Number of samples")
#axs[-1].set_xticks([100,1000,10000])
axs[-1].get_xaxis().set_major_formatter(ScalarFormatter())

plt.tight_layout()
#plt.show()
plt.savefig("./cluster_labels_aggregation.png",dpi=300)


#Plot the mode labels together on the same plot
fig, ax = plt.subplots(figsize=(4,3))
df_agg_data_10k.plot(y='acc_mode',ax=ax,c='tab:blue', label='DEC 10k iterations',marker='o')
df_agg_data_100.plot(y='acc_mode',ax=ax,c='tab:red',label='DEC 100 iterations',marker='o')
df_agg_data_kmeans.plot(y='acc_mode',ax=ax,c='tab:grey',label='kmeans',marker='o')
ax.set_xscale('log')
ax.set_xlabel("Number of samples")
ax.get_xaxis().set_major_formatter(ScalarFormatter())
plt.tight_layout()
#plt.show()
plt.savefig("./cluster_method_comparison.png",dpi=300)