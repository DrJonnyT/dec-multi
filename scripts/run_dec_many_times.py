#Set tensorflow gpu memory to be able to grow, which makes it less likely to crash
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import shutil
import os
import sys
sys.path.append('../')  #Add parent folder to path so imports work


from multi.dec import dec_mnist_n_times_csv

#Run Deep embedded clustering lots of times and save the results to csv files
#Load settings from params.py file
#An array of the number of copies of each digit to use
n_digits_array = [100,250,500,750,1000,2500,5000,7500,10000,25000,50000,70000]
#n_digits_array.reverse()
#The number of times to run DEC
n_runs = 25
#Number of clusters
n_clusters = 10
#Output folder
output_folder = "../output_unbalanced_100_50/"


#Copy this file and params file into the output folder
os.makedirs(os.path.dirname(output_folder), exist_ok=True)
shutil.copyfile("./run_dec_many_times.py", output_folder+"run_dec_many_times.py")


#Loop through different sized datasets
for n_digits in n_digits_array:
        csv_file = output_folder + f"dec_{n_digits}.csv"
        #Full version
        # dec_mnist_n_times_csv(n10, n_runs, n_clusters,csv_file,newcsv=False,
        #             fail_tolerance=10000)
        dec_mnist_n_times_csv(n_digits, n_runs, n_clusters,csv_file,newcsv=False,
                    iter_max=1000,
                    finetune_iters=100,layerwise_pretrain_iters=50,resample=False,fail_tolerance=1000,balanced=False)
