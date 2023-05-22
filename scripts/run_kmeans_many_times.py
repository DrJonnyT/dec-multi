from multi.kmeans import kmeans_mnist_n_times_csv
import params.params_no_resample as params


#Run kmeans lots of times and save the results to csv files
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
    print(f"Running with {n10} of each digit")
    kmeans_mnist_n_times_csv(n10, n_runs, n_clusters,csv_folder+f"kmeans_{n10}.csv",resample=resample)