import pandas as pd
import numpy as np
from pathlib import Path

from keras_dec.keras_dec import DeepEmbeddingClustering



def dec_n_times_csv(X,Y, n, num_clusters, csv_file, newcsv=True, **kwargs):
    """
    Run deep embedded clustering `n` times on data `X`, with `num_clusters`
    clusters, and append the resulting cluster assignments to a CSV file.
    The reason the clustering and file writing are tied into the same function
    is so you still save the data if it crashes halfway through.
    In the saved csv file, each sample is a row and each run of DEC is a
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
        finetune_iters : argument for DeepEmbeddingClustering.initialize
        (see keras_dec)
        layerwise_pretrain_iters : argument for 
        DeepEmbeddingClustering.initialize (see keras_dec)
        iter_max : argument for DeepEmbeddingClustering.cluster (see keras_dec)
        
        
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
        
        c = DeepEmbeddingClustering(n_clusters=num_clusters,
                                    input_dim=np.shape(X)[1])
        c.initialize(X, finetune_iters=finetune_iters,
                     layerwise_pretrain_iters=layerwise_pretrain_iters)
        c.cluster(X, y=Y,iter_max=iter_max,save_interval=0)

        #Load the csv, add a column for the DEC cluster labels, then save
        df_dec = pd.read_csv(csv_file,index_col=0)
        df_dec[f'dec_{i+1}'] = c.q.argmax(1)
        df_dec.to_csv(csv_file)
        

    return df_dec