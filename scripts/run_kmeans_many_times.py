import shutil
import os
import sys
sys.path.append('../')  #Add parent foldet to path so imports work


from multi.kmeans import kmeans_mnist_n_times_csv


#Run Kmeans clustering lots of times and save the results to csv files

#An array of the number of copies of each digit to use
n_digits_array = [100,250]#,500,750,1000,2500,5000,7500,10000,25000,50000,70000]

#The number of times to run kmeans
n_runs = 25
#Number of clusters
n_clusters = 10
#Output folder
output_folder = "../output_unbalanced_kmeans/"


#Copy this file and params file into the output folder
os.makedirs(os.path.dirname(output_folder), exist_ok=True)
shutil.copyfile("./run_dec_many_times.py", output_folder+"run_dec_many_times.py")


#Loop through different sized datasets
for n_digits in n_digits_array:
    print(f"Running with {n_digits} of each digit")
    csv_file = output_folder + f"dec_{n_digits}.csv"
    kmeans_mnist_n_times_csv(n_digits, n_runs, n_clusters,csv_file,balanced=False)