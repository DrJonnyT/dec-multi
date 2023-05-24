import pandas as pd
import numpy as np
from pathlib import Path
from os.path import splitext

from keras_dec.keras_dec import DeepEmbeddingClustering
from mnist.mnist import get_mnist, subsample_digits



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
            dec = DeepEmbeddingClustering(n_clusters=n_clusters,
                                        input_dim=np.shape(X)[1])
            dec.initialize(X, finetune_iters=finetune_iters,
                         layerwise_pretrain_iters=layerwise_pretrain_iters,
                         verbose=verbose)
            dec.cluster(X, iter_max=iter_max,save_interval=0)

            #Add a column for the DEC cluster labels, then save    
            df_dec[f'dec_{i+1}'] = dec.q.argmax(1)
            df_dec.to_csv(csv_file)
            counter = counter + 1
        except:
            counter = counter + 1       
        
    return df_dec



def dec_mnist_n_times_csv(n_digits, n_runs, n_clusters, csv_file, overwrite=False, **kwargs):
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
    n_digits : int
        The number of digits to sample. If set to <=0, it will run with
        the full MNIST dataset and ignore the resample flag.
    n_runs : int
        The number of times to resample and run deep embedded clustering. Note
        when appending that this is the total number of runs, including any
        that are already in the files.
    n_clusters : int
        The number of clusters.
    csv_file : string
        Path to CSV output file.
    overwrite : bool, default = False
        Overwrite any file that was there before. If false, appends to existing
        csv if one exists.
            
    **kwargs: 
        finetune_iters : argument for DeepEmbeddingClustering.initialize
        (see keras_dec).
        layerwise_pretrain_iters : argument for 
        DeepEmbeddingClustering.initialize (see keras_dec).
        iter_max : argument for DeepEmbeddingClustering.cluster (see keras_dec).
        verbose : verbose flag for initialize step.
        fail_tolerance : int, number of fails of DEC allowed before stopping 
        (Default is 1).
        balanced : Take a balanced sample of n_digits/10 copies of each 
        each digit, rather than a fully random sample. Default: False

        
        
    Returns
    -------
    df_dec : pandas DataFrame
        DataFrame of the same data from the csv. Rows are samples and columns
        are cluster labels from different runs of DEC.
    df_labels : pandas DataFrame
        Dataframe of the relevant labels from the mnist dataset.

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
    if "resample" in kwargs:
        resample = kwargs.get("resample")
    else:
        resample=False
    if n_digits <=0: #Flag for doing the full dataset
        resample=False
    if "fail_tolerance" in kwargs:
        fail_tolerance = kwargs.get("fail_tolerance")
    else:
        fail_tolerance = 1
    if "balanced" in kwargs:
        balanced = kwargs.get("balanced")
    else:
        balanced = False
       
    #Cannot resample if picking digits at random
    if balanced is True:
        resample = False
        
    #Check if there are enough digits in mnist dataset
    if n_digits > 63130 and balanced is True:
        raise Exception("""n_digits can only be max size of 63130 if balanced
                        is True as there are only 6313 copies of 5 in the mnist
                        data""")
    elif n_digits > 70000:
        raise Exception(""""n_digits can only be max size of 70000 as there are
                        only 70000 digits in the mnist data""")
    
    
    
    #Get mnist dataset
    X,Y = get_mnist()
    
    #Make empty dataframes
    df_dec = pd.DataFrame(index=[f'sample_{i}' for i in range(n_digits)])
    df_labels = pd.DataFrame(index=[f'sample_{i}' for i in range(n_digits)])
    df_indices = pd.DataFrame(index=[f'sample_{i}' for i in range(n_digits)])
      
    
    #Check if file exists- if not it needs creating
    #Note that newcsv can also be true from the input arguments
    csv_path = Path(csv_file)
    if csv_path.exists() is False:
        #If file doesn't exists, make a new file ('overwrite' an empty space)
        overwrite = True
    
    #Work out the path of the labels csv file, and indices csv file
    labels_file = splitext(csv_file)[0] + "_labels" + splitext(csv_file)[1]
    indices_file = splitext(csv_file)[0] + "_indices" + splitext(csv_file)[1]
    
    #Prepare the output files
    #Make sure the parent directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        #Make new CSVs for output.
        df_dec.to_csv(csv_file)
        df_labels.to_csv(labels_file)
        df_indices.to_csv(indices_file)
    
    #Subsample the digits
    Xsub, Ysub, indices = subsample_digits(X,Y,n_digits=n_digits,balanced=balanced)
    
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
        
        #Load the labels csv
        df_indices =  pd.read_csv(indices_file,index_col=0)        
        
        
        if resample==True:
            #Subsample the digits
            Xsub, Ysub, indices = subsample_digits(X,Y,n_digits=n_digits,balanced=balanced)
        
        try:
            #Run deep embedded clustering
            dec = DeepEmbeddingClustering(n_clusters=n_clusters,
                                        input_dim=np.shape(X)[1])
            dec.initialize(Xsub, finetune_iters=finetune_iters,
                         layerwise_pretrain_iters=layerwise_pretrain_iters,
                         verbose=verbose)
            dec.cluster(Xsub, iter_max=iter_max,save_interval=0)

            #Add a column for the DEC cluster labels, then save
            df_dec[f'dec_{n_runs_completed+1}'] = dec.q.argmax(1)
            df_dec.to_csv(csv_file)
            
            #Also save labels
            df_labels[f'labels_{n_runs_completed+1}'] = Ysub
            df_labels.to_csv(labels_file)
            
            #Also save indices
            df_indices[f'indices_{n_runs_completed+1}'] = indices
            df_indices.to_csv(indices_file)
            
            n_runs_completed = n_runs_completed + 1
        except:
            num_fails = num_fails + 1
        
        if num_fails > fail_tolerance:
            raise Exception(f"""VRAM run out: dec_mnist_n_times_csv has failed
                            too many ({num_fails}) times, the system has
                            probably run out of VRAM""")
        
        
        counter = counter + 1
        
            

    return df_dec, df_labels, df_indices
