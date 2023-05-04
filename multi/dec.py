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
    
    
    counter = 0
    while counter < 10000:
        print(f"dec_n_times_csv Counter = {counter}")
        #Load the csv
        df_dec = pd.read_csv(csv_file,index_col=0)
        i = df_dec.shape[1]-1 #The number of times DEC has been run in the file
        print(f"dec_n_times_csv i = {i}")
        if i >n:
            break
        
        try:
            c = DeepEmbeddingClustering(n_clusters=n_clusters,
                                        input_dim=np.shape(X)[1])
            c.initialize(X, finetune_iters=finetune_iters,
                         layerwise_pretrain_iters=layerwise_pretrain_iters,
                         verbose=verbose)
            c.cluster(X, iter_max=iter_max,save_interval=0)

            #Add a column for the DEC cluster labels, then save    
            df_dec[f'dec_{i+1}'] = c.q.argmax(1)
            df_dec.to_csv(csv_file)
            counter = counter + 1
        except:
            counter = counter + 1       
        
    return df_dec



def dec_mnist_n_times_csv(n10, n_runs, n_clusters, csv_file, newcsv=True, resample=True, **kwargs):
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
        The number of each digits to sample. If set to <=0, it will run with
        the full MNIST dataset and ignore the resample flag.
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
    resample: bool, default = True
        If true then resample from mnist each time, otherwise just run 
        repeatedly on the sample sample of digits.
            
    **kwargs: 
        finetune_iters : argument for DeepEmbeddingClustering.initialize
        (see keras_dec).
        layerwise_pretrain_iters : argument for 
        DeepEmbeddingClustering.initialize (see keras_dec).
        iter_max : argument for DeepEmbeddingClustering.cluster (see keras_dec).
        verbose : verbose flag for initialize step.
        fail_tolerance : int, number of fails of DEC allowed before stopping 
        (Default is 1).

        
        
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
    if "resample" in kwargs:
        resample = kwargs.get("resample")
    else:
        resample=True
    if n10 <=0: #Flag for doing the full dataset
        resample=False
    if "fail_tolerance" in kwargs:
        fail_tolerance = kwargs.get("fail_tolerance")
    else:
        fail_tolerance = 1
        
    
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
    
        
    #Select the digits if needs be here
    if resample==False:
        if n10<=0:
            #Do not subsample data
            Xsub = X
            Ysub = Y
        else:
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
    
    
    #Main while loop running through DEC
    n_runs_completed = 0
    counter = 0
    num_fails = 0
    while counter < 10000:
        #Load the csv file and see how many runs have been completed so far
        df_dec = pd.read_csv(csv_file,index_col=0)
        n_runs_completed = df_dec.shape[1]
        
        #Put this break in explicitly
        if n_runs_completed >= n_runs:
            break
        
        #Load the labels csv
        df_labels =  pd.read_csv(labels_file,index_col=0)
        
        if resample==True:
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
        
        try:
            #Run deep embedded clustering
            c = DeepEmbeddingClustering(n_clusters=n_clusters,
                                        input_dim=np.shape(X)[1])
            c.initialize(Xsub, finetune_iters=finetune_iters,
                         layerwise_pretrain_iters=layerwise_pretrain_iters,
                         verbose=verbose)
            c.cluster(Xsub, iter_max=iter_max,save_interval=0)
    
            #Add a column for the DEC cluster labels, then save
            df_dec[f'dec_{n_runs_completed+1}'] = c.q.argmax(1)
            df_dec.to_csv(csv_file)
            
            #Also save labels
            df_labels[f'labels_{n_runs_completed+1}'] = Ysub
            df_labels.to_csv(labels_file)
            
            n_runs_completed = n_runs_completed + 1
        except:
            num_fails = num_fails + 1
        
        if num_fails > fail_tolerance:
            raise Exception(""""VRAM run out: dec_mnist_n_times_csv has failed
                            too many ({num_fails}) times, the system has
                            probably run out of VRAM""")
        
        
        counter = counter + 1
        
            
        
        

    return df_dec, df_labels