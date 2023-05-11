import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import NMF
from itertools import combinations

from keras_dec.functions import cluster_acc


def mean_rand_index(df):
    """
    Calculate the mean Rand index for all pairs of columns in a pandas DataFrame `df`,
    where each column represents a different set of cluster labels.
    """
    
    #Remove the actual data labels
    if df.columns[0] == 'labels':
        df.drop('labels',axis=1,inplace=True)
    
    
    # Get all pairs of column names
    column_pairs = list(combinations(df.columns, 2))

    # Calculate the Rand index for each pair of columns
    rand_indices = [adjusted_rand_score(df[pair[0]], df[pair[1]]) for pair in column_pairs]

    # Return the mean Rand index for all pairs of columns
    return np.mean(rand_indices)


def rand_index_arr(df,labels):

    """
    Calculate the Rand index for all columns in a pandas DataFrame 'df'
    (where each column represents a different set of cluster labels), and a
    set of labels 'labels'

    Parameters
    ----------
    df : DataFrame
        Each column is a set of cluster labels. dtype must be int.
    labels : array
        cluster labels, of the same length as the number of rows in df.
        dtype must be int

    Returns
    -------
    rand_arr : numpy array
        Array of rand index corresponding to each column in df.

    """
    
    #Remove the actual data labels
    if df.columns[0] == 'labels':
        df.drop('labels',axis=1,inplace=True)
    
    rand_arr = [adjusted_rand_score(labels,df[col]) for col in df.columns]
    return np.array(rand_arr)


def accuracy_arr(df,labels):
    """
    Calculate the accuracy for all columns in a pandas DataFrame 'df'
    (where each column represents a different set of cluster labels), compared 
    to a set of labels 'labels'

    Parameters
    ----------
    df : DataFrame
        Each column is a set of cluster labels. dtype must be int.
    labels : array
        cluster labels, of the same length as the number of rows in df.
        dtype must be int

    Returns
    -------
    acc_arr : numpy array
        Array of accuracy corresponding to each column in df.

    """
    
    #Remove the actual data labels
    if df.columns[0] == 'labels':
        df.drop('labels',axis=1,inplace=True)
        
    acc_arr = [cluster_acc(labels,df[col])[0] for col in df.columns]
    return np.array(acc_arr)



def prob_lab_agg(df_labels,norm=True):
    """
    Probabalistic label aggregation function
    Based on the paper by Lange & Buhmann 2005, DOI:10.1145/1081870.1081890
    Available at:
    https://www.researchgate.net/publication/221654189_Combining_partitions_by_probabilistic_label_aggregation
    
    Take a dataframe where each column is a set of cluster labels. Aggregate
    these sets together based on consideration of 'what are the chances that 
    sample i and sample j have the same cluster label?'
    
    Parameters
    ----------
    df_labels : Pandas DataFrame
        Dataframe of cluster labels. Each row is a sample and each column a set
        of labels.
    norm : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    HW_labels : numpy array
        Aggregated cluster labels.

    """
    
    n_samples = np.shape(df_labels)[0]
    n_runs = np.shape(df_labels)[1]
    Z = np.zeros([n_samples,n_samples])
    
    #Loop through each sample
    for sample_i in range(n_samples):
        #Work out the number of times that other samples are in the same cluster
        zarr_sample_i = np.zeros(n_samples)
        
        #Loop through all runs
        for run in df_labels.columns:
            #Get an array that's 1 if they are the same and 0 if different, and add
            zarr_sample_i = zarr_sample_i + (df_labels[run]==df_labels[run].iloc[sample_i]).astype(int)
        
        Z[sample_i] = zarr_sample_i
    
    if norm:
        p_i = np.sum(Z,axis=0) / np.sum(Z)
        Z = Z / p_i
        #Z = Z / n_runs
    
    #Now run NMF
    #model = NMF(n_components=10, init='random', beta_loss='kullback-leibler',solver='mu')  
    model = NMF(n_components=10, init='nndsvd', beta_loss='frobenius')  
    

    W = model.fit_transform(Z)
    H = model.components_
    #H_labels = np.argmax(H,axis=0)
    #W_labels = np.argmax(W,axis=1)
    HW_labels = np.argmax(H*W.T,axis=0)
    return HW_labels