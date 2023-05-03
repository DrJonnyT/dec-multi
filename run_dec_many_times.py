#Set tensorflow gpu memory to be able to grow, which makes it less likely to crash
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from multi.dec import dec_mnist_n_times_csv
import params.params_full_mnist as params

#Run Deep embedded clustering lots of times and save the results to csv files
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

#Loop through different sized datasets
for n10 in n10_array:
        csv_file = csv_folder + f"dec_{n10}.csv"
        dec_mnist_n_times_csv(n10, n_runs, n_clusters,csv_file,newcsv=False,
                    iter_max=100,
                    finetune_iters=10000,layerwise_pretrain_iters=5000,resample=resample,fail_tolerance=1000)
