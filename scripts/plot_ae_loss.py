"""
Load AE loss from different sample sizes and plot
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import sys
sys.path.append('../')  #Add parent folder to path so imports work


#An array of the number of copies of each digit to use
n_digits_array = [100,250,500,750,1000,2500,5000,7500,10000,25000,50000,70000]

#Load multiple cluster labels
output_folder = "../output_unbalanced_10000_5000_again/"

ds_loss_10k = pd.Series()
for n_digits in n_digits_array:
    loss_file = output_folder + f"dec_{n_digits}_aeloss.csv"    
    
    try:
        ds_loss_loaded = pd.read_csv(loss_file,index_col=0).squeeze('columns')    
    except:
        print(f"Unable to load file for {n_digits} digits")
        break
    
    ds_loss_10k[n_digits] = ds_loss_loaded.mean()


   
#Load multiple cluster labels for 100 iterations
output_folder = "../output_unbalanced_100_50_again/"

ds_loss_100 = pd.Series()
for n_digits in n_digits_array:
    loss_file = output_folder + f"dec_{n_digits}_aeloss.csv"    
    
    try:
        ds_loss_loaded = pd.read_csv(loss_file,index_col=0).squeeze('columns')    
    except:
        print(f"Unable to load file for {n_digits} digits")
        break
    
    ds_loss_100[n_digits] = ds_loss_loaded.mean()
    
    
    
    
fig,ax = plt.subplots(figsize=(5,4))
ds_loss_10k.plot(ax=ax,label="10k iterations",c='tab:blue')
ds_loss_100.plot(ax=ax,label="100 iterations",c='tab:red')
ax.set_ylabel('AE loss')
ax.set_xlabel('Number of samples')
ax.set_xscale('log')
ax.get_xaxis().set_major_formatter(ScalarFormatter())
ax.legend()
plt.tight_layout()
plt.savefig("./ae_loss.png",dpi=300)