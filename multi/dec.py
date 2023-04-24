import pandas as pd
import numpy as np
from pathlib import Path
from os.path import splitext

from keras_dec.keras_dec import DeepEmbeddingClustering
from mnist.mnist import get_mnist



def dec_n_times_csv(X,Y, n, n_clusters, csv_file, newcsv=True, **kwargs):
    """
    Run deep embedded clustering `n` times on data `X`, with `n_clusters`
    clusters, and append the resulting cluster assignments to a CSV file.
    The reason the clustering and file writing are tied into the same function
    is so you still save the data if it crashes halfway through.
    In the saved csv file, each sample is a row and each run of DEC is a
    column.
    
    Parameters
    ----------
    X : array
        The data to be clustered. Each row is a set of data.
    n : int
        The number of times to run kmeans.
    n_clusters : int
        The number of clusters.
    csv_file : string
        Path to CSV output file.
    newcsv : bool, default = True
        If this is true, create a new csv and overwrite any that was there
        before. If false, appends to existing csv if one exists.
            
    **kwargs: 
        finetune_iters : argument for DeepEmbeddingClustering.initialize
        (see keras_dec)
        layerwise_pretrain_iters : argument for 
        DeepEmbeddingClustering.initialize (see keras_dec)
        iter_max : argument for DeepEmbeddingClustering.cluster (see keras_dec)
        verbose : verbose flag for initialize step
        
        
    Returns
    -------
    df_dec : pandas DataFrame
        DataFrame of the same data from the csv. Rows are samples and columns
        are cluster labels from different runs of DEC.

    """
    
    
    #Sort through kwargs
    if "finetune_iters" in kwargs:
        finetune_iters = kwargs.get("finetune_iters")
    else:
        finetune_iters = 100000
    if "layerwise_pretrain_iters" in kwargs:
        layerwise_pretrain_iters = kwargs.get("layerwise_pretrain_iters")
    else:
        layerwise_pretrain_iters = 50000
    if "iter_max" in kwargs:
        iter_max = kwargs.get("iter_max")
    else:
        iter_max = 1000
    if "verbose" in kwargs:
        verbose = kwargs.get("verbose")
    else:
        verbose = "auto"
      
    
    #Make empty dataframe
    df_dec = pd.DataFrame(index=[f'sample_{i}' for i in range(len(X))])
    
    #Add the corresponding labels
    df_dec['labels'] = Y

    
    #Check if file exists- if not it needs creating
    #Note that newcsv can also be true from the input arguments
    csv_path = Path(csv_file)
    if csv_path.exists() is False:
        newcsv = True
    
    
    if newcsv:
        #Make a new csv. Make the directory first.
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_dec.to_csv(csv_file)
    
    
    for i in range(n):
        
        c = DeepEmbeddingClustering(n_clusters=n_clusters,
                                    input_dim=np.shape(X)[1])
        c.initialize(X, finetune_iters=finetune_iters,
                     layerwise_pretrain_iters=layerwise_pretrain_iters,
                     verbose=verbose)
        c.cluster(X, y=Y,iter_max=iter_max,save_interval=0)

        #Load the csv, add a column for the DEC cluster labels, then save
        df_dec = pd.read_csv(csv_file,index_col=0)
        df_dec[f'dec_{i+1}'] = c.q.argmax(1)
        df_dec.to_csv(csv_file)
        

    return df_dec



def dec_mnist_n_times_csv(n10, n_runs, n_clusters, csv_file, newcsv=True, **kwargs):
    """
    Run deep embedded clustering `n_runs` times on mnist data, with
    `n_clusters` clusters, and append the resulting cluster assignments to a
    CSV file. The reason the clustering and file writing are tied into the same
    function is so you still save the data if it crashes halfway through.
    In the saved csv file, each sample is a row and each run of DEC is a
    column.
    There is also a second CSV file saved, that contains the relevant labels
    from the mnist data for each run.
    
    Parameters
    ----------
    n10 : int
        The number of each digits to sample.
    n_runs : int
        The number of times to resample and run deep embedded clustering. Note
        when appending that this is the total number of runs, including any
        that are already in the files.
    n_clusters : int
        The number of clusters.
    csv_file : string
        Path to CSV output file.
    newcsv : bool, default = True
        If this is true, create a new csv and overwrite any that was there
        before. If false, appends to existing csv if one exists.
            
    **kwargs: 
        finetune_iters : argument for DeepEmbeddingClustering.initialize
        (see keras_dec)
        layerwise_pretrain_iters : argument for 
        DeepEmbeddingClustering.initialize (see keras_dec)
        iter_max : argument for DeepEmbeddingClustering.cluster (see keras_dec)
        verbose : verbose flag for initialize step
        
        
    Returns
    -------
    df_dec : pandas DataFrame
        DataFrame of the same data from the csv. Rows are samples and columns
        are cluster labels from different runs of DEC.
    df_labels : pandas DataFrame
        Dataframe of the relevant labels from the mnist dataset.

    """
    
    if n10 > 6313:
            raise Exception("n10 can only be max size of 6313 as there are only 6313 copies of 5 in the mnist data")
            
    #Sort through kwargs
    if "finetune_iters" in kwargs:
        finetune_iters = kwargs.get("finetune_iters")
    else:
        finetune_iters = 100000
    if "layerwise_pretrain_iters" in kwargs:
        layerwise_pretrain_iters = kwargs.get("layerwise_pretrain_iters")
    else:
        layerwise_pretrain_iters = 50000
    if "iter_max" in kwargs:
        iter_max = kwargs.get("iter_max")
    else:
        iter_max = 1000
    if "verbose" in kwargs:
        verbose = kwargs.get("verbose")
    else:
        verbose = "auto"
        
    
    #Get mnist dataset
    X,Y = get_mnist()
    
    #Make empty dataframes
    df_dec = pd.DataFrame(index=[f'sample_{i}' for i in range(n10*10)])
    df_labels = pd.DataFrame(index=[f'sample_{i}' for i in range(n10*10)])
      
    
    #Check if file exists- if not it needs creating
    #Note that newcsv can also be true from the input arguments
    csv_path = Path(csv_file)
    if csv_path.exists() is False:
        newcsv = True
    
    #Work out the path of the labels csv file
    labels_file = splitext(csv_file)[0] + "_labels" + splitext(csv_file)[1]
    
    if newcsv:
        #Make a new csv. Make the directory first.
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_dec.to_csv(csv_file)
        df_labels.to_csv(labels_file)
        
    
    n_runs_completed = 0
    n_runs_loop = 0
    while n_runs_completed < n_runs:
        #Load the csv file and see how many runs have been completed so far
        df_dec = pd.read_csv(csv_file,index_col=0)
        n_runs_completed = df_dec.shape[1]
        
        #Put this break in explicitly
        if n_runs_completed >= n_runs:
            break
        
        #Load the labels csv
        df_labels =  pd.read_csv(labels_file,index_col=0)
        
        #Subsample data
        #Empty lists for the subsampled data
        Xsub = np.zeros((0, 784))
        Ysub = np.zeros(0,dtype='int')  
        
        # Select 10 instances of each digit (0-9) at random
        for digit in range(10):
            indices = np.where(Y == digit)[0]
            indices = np.random.choice(indices, size=n10, replace=False)
            Xsub = np.vstack((Xsub,X[indices]))
            Ysub = np.append(Ysub,Y[indices])
        
        
        #Run deep embedded clustering
        c = DeepEmbeddingClustering(n_clusters=n_clusters,
                                    input_dim=np.shape(X)[1])
        c.initialize(Xsub, finetune_iters=finetune_iters,
                     layerwise_pretrain_iters=layerwise_pretrain_iters,
                     verbose=verbose)
        c.cluster(Xsub, y=Ysub,iter_max=iter_max,save_interval=0)

        #Add a column for the DEC cluster labels, then save
        df_dec[f'dec_{n_runs_completed+1}'] = c.q.argmax(1)
        df_dec.to_csv(csv_file)
        
        #Also save labels
        df_labels[f'labels_{n_runs_completed+1}'] = Ysub
        df_labels.to_csv(labels_file)
        
        n_runs_completed = n_runs_completed + 1
        
        #Catch in case you get stuck in an infinite loop
        n_runs_loop = n_runs_loop + 1
        if n_runs_loop > 1000:
            raise Exception("dec_mnist_n_times_csv stuck has run too many ({n_runs_loop}) times")
        
        

    return df_dec, df_labels