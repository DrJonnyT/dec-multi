"""
Load ae_loss from different sample sizes and plot average
Also scatter plot of ae_loss vs accuracy
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap, LogNorm
from matplotlib.ticker import ScalarFormatter
import numpy as np

import sys
sys.path.append('../')  #Add parent folder to path so imports work

from multi.comparison import accuracy_arr


#An array of the number of copies of each digit to use
n_digits_array = [100,250,500,750,1000,2500,5000,7500,10000,25000,50000,70000]

#Load multiple cluster labels
output_folder = r"../output_unbalanced_10000_5000_again/"

#Series for the mean loss
ds_mean_loss_10k = pd.Series()
#Dataframe for run-by-run metrics
df_runs_10k = pd.DataFrame(columns=['accuracy','ae_loss','n_digits'])

for n_digits in n_digits_array:
    dec_file = output_folder + f"dec_{n_digits}.csv"
    labels_file = output_folder + f"dec_{n_digits}_labels.csv"
    loss_file = output_folder + f"dec_{n_digits}_aeloss.csv"
    
    try:
        df_dec_loaded = pd.read_csv(dec_file,index_col=0)
        df_labels_loaded = pd.read_csv(labels_file,index_col=0)
        ds_loss_loaded = pd.read_csv(loss_file,index_col=0).squeeze('columns')
    except:
        print(f"Unable to load files for {n_digits} digits")
        break
    
    #Append run by run accuracy, loss, and digits   
    df_runs = pd.DataFrame()
    df_runs['accuracy'] = accuracy_arr(df_dec_loaded,df_labels_loaded['labels_1'])
    df_runs['ae_loss'] = ds_loss_loaded.values
    df_runs['n_digits'] = np.ones(len(ds_loss_loaded))*n_digits
    
    df_runs_10k = pd.concat([df_runs_10k,df_runs])
        
    ds_mean_loss_10k[n_digits] = ds_loss_loaded.mean()


   
#Load multiple cluster labels for 100 iterations
output_folder = "../output_unbalanced_100_50_again/"

ds_mean_loss_100 = pd.Series()
df_runs_100 = pd.DataFrame(columns=['accuracy','ae_loss','n_digits'])
for n_digits in n_digits_array:
    dec_file = output_folder + f"dec_{n_digits}.csv"
    labels_file = output_folder + f"dec_{n_digits}_labels.csv"
    loss_file = output_folder + f"dec_{n_digits}_aeloss.csv"
    
    try:
        df_dec_loaded = pd.read_csv(dec_file,index_col=0)
        df_labels_loaded = pd.read_csv(labels_file,index_col=0)
        ds_loss_loaded = pd.read_csv(loss_file,index_col=0).squeeze('columns')
    except:
        print(f"Unable to load files for {n_digits} digits")
        break
    
    #Append run by run accuracy, loss, and digits   
    df_runs = pd.DataFrame()
    df_runs['accuracy'] = accuracy_arr(df_dec_loaded,df_labels_loaded['labels_1'])
    df_runs['ae_loss'] = ds_loss_loaded.values
    df_runs['n_digits'] = np.ones(len(ds_loss_loaded))*n_digits
    
    df_runs_100 = pd.concat([df_runs_100,df_runs])
        
    ds_mean_loss_100[n_digits] = ds_loss_loaded.mean()
    
    
#First, plot the mean ae_loss for each n_digits
    
fig,ax = plt.subplots(figsize=(5,4))
ds_mean_loss_10k.plot(ax=ax,label="10k iterations",c='tab:blue')
ds_mean_loss_100.plot(ax=ax,label="100 iterations",c='tab:red')
ax.set_ylabel('ae_loss')
ax.set_xlabel('Number of samples')
ax.set_xscale('log')
ax.get_xaxis().set_major_formatter(ScalarFormatter())
ax.legend()
plt.tight_layout()
plt.savefig("./ae_loss.png",dpi=300)


#%%Now, plot loss vs accuracy on a run by run basis
fig,ax = plt.subplots(figsize=(4,3))

#Normalise the colour map scale between the two datasets
norm = Normalize(vmin=min(df_runs_100['n_digits'].min(),df_runs_10k['n_digits'].min()),
                 vmax=max(df_runs_100['n_digits'].max(),df_runs_10k['n_digits'].max()))

# Create a custom colormap with discrete colors
cmap_orig = plt.cm.RdYlBu_r
unique_values = np.unique(np.concatenate([df_runs_100['n_digits'],df_runs_10k['n_digits']]))

norm = LogNorm(vmin=np.min(unique_values), vmax=np.max(unique_values))
cmap_discrete = ListedColormap([cmap_orig(norm(val)) for val in unique_values])


scatter = ax.scatter(df_runs_10k['ae_loss'],df_runs_10k['accuracy'],c=df_runs_10k['n_digits'],
           norm=norm,marker='o',cmap=cmap_discrete,label='10k iterations')
ax.scatter(df_runs_100['ae_loss'],df_runs_100['accuracy'],c=df_runs_100['n_digits'],
           norm=norm,marker='^',cmap=cmap_discrete,label='100 iterations')

# Add a colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Num digits')
cbar.ax.yaxis.set_major_formatter(ScalarFormatter())

ax.set_xlabel('AE loss')
ax.set_ylabel('Accuracy')
ax.legend()

plt.tight_layout()
plt.savefig("./ae_loss_acc_scatter.png",dpi=300)