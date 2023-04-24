#Set tensorflow gpu memory to be able to grow, which makes it less likely to crash
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from multi.dec import dec_mnist_n_times_csv

#Setup what sizes of dataset we would like
#The maximum is 6313, so base it off that
#This number is the number of each digit we will use
n10_array = [6,63,631,6313]


#Run kmeans lots of times
n_runs = 100
n_clusters = 10
csv_folder = "./output/"
#Loop through different sized datasets
for n10 in n10_array:
    csv_file = csv_folder + f"dec_{n10}.csv"
    dec_mnist_n_times_csv(n10, n_runs, n_clusters,csv_file,newcsv=False,
                    finetune_iters=1000,layerwise_pretrain_iters=500,iter_max=10,
                    verbose=0)

