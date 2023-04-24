#Set tensorflow gpu memory to be able to grow
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from multi.kmeans import kmeans_mnist_n_times_csv




#Setup what sizes of dataset we would like
#The maximum is 6313, so base it off that
#This number is the number of each digit we will use
n10_array = [6,63]
#n10_array = [6,63,631,6313]


#Run kmeans lots of times
n_runs = 5
num_clusters = 10
csv_folder = "./output/"
#Loop through different sized datasets
for n10 in n10_array:
    kmeans_mnist_n_times_csv(n10, n_runs, num_clusters,csv_folder+f"kmeans_{n10}.csv")

