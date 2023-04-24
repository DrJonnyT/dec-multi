from mnist.mnist import get_mnist
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import warnings
from pathlib import Path
from os.path import splitext




def kmeans_n_times_csv(X, n, num_clusters, csv_file, newcsv=True, **kwargs):
    """
    Run k-means clustering `n` times on data `X`, with `num_clusters` clusters,
    and append the resulting cluster assignments to a CSV file `csv_file`.
    The reason the clustering and file writing are tied into the same function
    is so you still save the data if it crashes halfway through, which may
    not happen for kmeans but is much more likely for other similar functions
    with other clustering, e.g. deep embedded clustering.
    In the saved csv file, each sample is a row and each run of kmeans is a
    column.
    
    Parameters
    ----------
    X : array
        The data to be clustered. Each row is a set of data
    n : int
        The number of times to run kmeans
    num_clusters : int
        The number of clusters
    csv_file : string
        Path to CSV output file
    newcsv : bool, default = True
        If this is true, create a new csv and overwrite any that was there
        before. If false, appends to existing csv if one exists.
            
    **kwargs: 
        labels : Array of labels for the data. If these are provided, it adds
        these as the first column to the csv.
        
    Returns
    -------
    df_kmeans : pandas DataFrame
        DataFrame of the same data from the csv. Rows are samples and columns
        are cluster labels from different runs of kmeans.

    """
      
    
    #Make empty dataframe
    df_kmeans = pd.DataFrame(index=[f'sample_{i}' for i in range(len(X))])
    
    #Add the corresponding labels
    if 'labels' in kwargs:
        df_kmeans['labels'] = kwargs.get('labels')

    
    #Check if file exists- if not it needs creating
    #Note that newcsv can also be true from the input arguments
    csv_path = Path(csv_file)
    if csv_path.exists() is False:
        newcsv = True
    
    
    if newcsv:
        #Make a new csv. Make the directory first.
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_kmeans.to_csv(csv_file)
        
    
    for i in range(n):
        
        #Control the number of threads in kmeans
        with threadpool_limits(limits=1, user_api='blas'):
            #Ignore the warning about the memory leak
            warnings.filterwarnings('ignore')
        
            # Fit the k-means model to the data
            kmeans = KMeans(n_clusters=num_clusters).fit(X)

        #Load the csv, add a column for the kmeans cluster labels, then save
        df_kmeans = pd.read_csv(csv_file,index_col=0)
        df_kmeans[f'kmeans_{i+1}'] = kmeans.labels_
        df_kmeans.to_csv(csv_file)
        

    return df_kmeans


def kmeans_mnist_n_times(n10, n_runs, num_clusters):
    """
    Run k-means clustering `n_runs` times on mnist digits data, with a random
    sample of 'n10' of each digit. With `num_clusters` clusters.
    After, append the resulting cluster assignments to a CSV file `csv_file`.
    The reason the clustering and file writing are tied into the same function
    is so you still save the data if it crashes halfway through, which may
    not happen for kmeans but is much more likely for other similar functions
    with other clustering, e.g. deep embedded clustering.
    In the saved csv file, each sample is a row and each run of kmeans is a
    column.
    
    Parameters
    ----------
    n10 : int
        The number of each digits to sample
    n_runs : int
        The number of times to resample and run kmeans
    num_clusters : int
        The number of clusters
        
    Returns
    -------
    df_kmeans : pandas DataFrame
        DataFrame of the same data from the csv. Rows are samples and columns
        are cluster labels from different runs of kmeans.
    df_labels : pandas DataFrame
        Dataframe of the correct labels for the equivalent column in df_kmeans

    """
    
    if n10 > 6313:
        raise Exception("n10 can only be max size of 6313 as there are only 6313 copies of 5 in the mnist data")
    
    #Get mnist dataset
    X,Y = get_mnist()
    
    #Make empty dataframes
    df_kmeans = pd.DataFrame(index=[f'sample_{i}' for i in range(n10*10)])
    df_labels = pd.DataFrame(index=[f'sample_{i}' for i in range(n10*10)])
    
       
    for run in range(n_runs):
        #Empty lists for the subsampled data
        Xsub = np.zeros((0, 784))
        Ysub = np.zeros(0,dtype='int')  
        
        
        # Select 10 instances of each digit (0-9) at random
        for digit in range(10):
            indices = np.where(Y == digit)[0]
            indices = np.random.choice(indices, size=n10, replace=False)
            Xsub = np.vstack((Xsub,X[indices]))
            Ysub = np.append(Ysub,Y[indices])

        
        #Control the number of threads in kmeans
        with threadpool_limits(limits=1, user_api='blas'):
            #Ignore the warning about the memory leak
            warnings.filterwarnings('ignore')
        
            # Fit the k-means model to the data
            kmeans = KMeans(n_clusters=num_clusters).fit(Xsub)

        #Append labels to the dataframes
        df_kmeans[f'kmeans_{run+1}'] = kmeans.labels_
        df_labels[f'labels_{run+1}'] = Ysub
        
        
    return df_kmeans, df_labels


def kmeans_mnist_n_times_csv(n10, n_runs, num_clusters,csv_file):
    """
    Run kmeans_mnist_n_times and save the results to a csv file

    Parameters
    ----------
    n10 : int
        The number of each digits to sample
    n_runs : int
        The number of times to resample and run kmeans
    num_clusters : int
        The number of clusters
    csv_file : string
        Path to CSV output file

    Returns
    -------
    None.

    """
    
    if n10 > 6313:
            raise Exception("n10 can only be max size of 6313 as there are only 6313 copies of 5 in the mnist data")
    
    #Work out the path of the labels csv file
    labels_file = splitext(csv_file)[0] + "_labels" + splitext(csv_file)[1]
    
    df_kmeans, df_labels = kmeans_mnist_n_times(n10, n_runs, num_clusters)
    
    #Make the directory if needs be and save the files
    csv_path = Path(csv_file)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_kmeans.to_csv(csv_file)
    df_labels.to_csv(labels_file)