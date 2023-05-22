"""Plot aggregated cluster labels produced by process_multi_labels_csv.py"""

import pandas as pd
import matplotlib.pyplot as plt
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



#Plot data
fig, ax = plt.subplots()
ax.set_ylabel('Accuracy')
ax.set_xlabel('Number of samples')
ax.set_xscale('log')

#Plot mean and stdev of single runs
df_agg_data_10k.plot(y='acc_mean',ax=ax,c='tab:blue',label='10k iters, single runs')
ax.fill_between(df_agg_data_10k.index,
                df_agg_data_10k['acc_mean']+df_agg_data_10k['acc_stdev'], 
                df_agg_data_10k['acc_mean']-df_agg_data_10k['acc_stdev'], color='tab:blue', alpha=0.2)

df_agg_data_10k.plot(y='acc_mode',ax=ax,c='tab:purple', label='10k iters, mode')
df_agg_data_10k.plot(y='acc_pla',ax=ax,c='cyan',label='10k iters, PLA')


#Do the same for the 100 iterations data
df_agg_data_100.plot(y='acc_mean',ax=ax,c='tab:red',label='100 iters, single runs')
ax.fill_between(df_agg_data_100.index,
                df_agg_data_100['acc_mean']+df_agg_data_100['acc_stdev'], 
                df_agg_data_100['acc_mean']-df_agg_data_100['acc_stdev'], color='tab:red', alpha=0.2)

df_agg_data_100.plot(y='acc_mode',ax=ax,c='orange',label='100 iters, mode')
df_agg_data_100.plot(y='acc_pla',ax=ax,c='yellow',label='100 iters, PLA')


